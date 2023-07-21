# Copyright 2022 Tao Jiang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
API: parse_data
    src_dir : tfrecord_dir/*.tfrecord
    dst_dir : output_dir/seq_dict_scratch_XXX.pkl
    func: parse points/labels/bbox and pose for each frame

API: cal_icp_transformation
    src_dir : output_dir/seq_dict_scratch_XXX.pkl
    dst_dir : output_dir/seq_dict_icp_XXX.pkl
    func: calulate icp transformation for each frame

'''
import argparse
import os
from pathlib import Path
import sys
import math
import numpy as np
import torch
import itertools
from collections import OrderedDict
from tqdm import tqdm
from glob import glob
import os.path as osp
# tf.enable_eager_execution()
import pickle as pkl
import time
import open3d as o3d
import multiprocessing
multiprocessing.set_start_method('forkserver', force=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # wtf tensorflow
import tensorflow as tf
from typing import List
from spconv.pytorch.hash import HashTable
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from utils.points_in_bbox import points_in_rbbox
from utils.config import *


DEBUG = False
VIS = False

def _expand_dim(array):
    return np.concatenate((array, np.ones_like(array)[:, :1]), axis=1)

def to_stride(shape: np.ndarray):
    stride = np.ones_like(shape)
    stride[:shape.shape[0] - 1] = np.cumprod(shape[::-1])[::-1][1:]
    return stride

def _indice_to_scalar(indices: torch.Tensor, shape: List[int]):
    assert indices.shape[1] == len(shape)
    stride = to_stride(np.array(shape, dtype=np.int64))
    scalar_inds = indices[:, -1].clone()
    for i in range(len(shape) - 1):
        scalar_inds += stride[i] * indices[:, i]
    return scalar_inds.contiguous()

def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print('current Function [%s] run time: %.2f s.' % (func.__name__ ,time.time() - local_time))
    return wrapper

def compute_box_3d(dim, location, yaw):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(yaw), np.sin(yaw)
  R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [-l / 2, -l / 2, -l / 2, -l / 2, l / 2, l / 2, l / 2, l / 2]
  y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
  z_corners = [-w / 2, -w / 2, -w / 2, -w / 2, w / 2, w / 2, w / 2, w / 2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
  return corners_3d.transpose(1, 0)

def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0,
                                              nlz_masks=True):
    """Convert segmentation labels from range images to point clouds.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      segmentation_labels: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for i, c in enumerate(calibrations):
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
            if nlz_masks:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask_ = range_image_mask & nlz_mask
                sl_points_tensor_ = tf.gather_nd(sl_tensor, tf.where(range_image_mask_==False))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels

def get_bbox(frame):
    # bboxes  = OrderedDict()
    # box_lidar_coord = np.zeros((0, 7))
    bboxes = []
    labels = []
    bboxes_id = []
    for box_idx, box in enumerate(frame.laser_labels):
        box_pos_info = box.box
        # bboxes[box_idx] = {
        #     'center': [box_pos_info.center_x, box_pos_info.center_y, box_pos_info.center_z],
        #     'dim': [box_pos_info.height, box_pos_info.width, box_pos_info.length],
        #     'heading': box_pos_info.heading,
        #     'id': box.id,
        #     'type': box.type,
        # }
        bbox = np.array([[box_pos_info.center_x, box_pos_info.center_y, box_pos_info.center_z-box_pos_info.height/2, box_pos_info.length, box_pos_info.width, box_pos_info.height, -box_pos_info.heading]])
        label = np.array([box.type])
        box_id = np.array([box.id])

        # corners = compute_box_3d(bboxes[box_idx]['dim'], bboxes[box_idx]['center'], bboxes[box_idx]['heading'])
        # bboxes[box_idx]['bbox'] = box_lidar_coord
        # bboxes[box_idx]['corners'] = corners
        bboxes.append(bbox)
        labels.append(label)
        bboxes_id.append(box_id)
    # indices = points_in_rbbox(pcd[:, :3], box_lidar_coord)
    if len(bboxes) > 0:
        bboxes = np.concatenate(bboxes)
        labels = np.concatenate(labels)
        bboxes_id = np.concatenate(bboxes_id)
    return bboxes, labels, bboxes_id

def my_remove_close(points, x_radius: float, y_radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_front_filt = points[:, 0] < x_radius
    x_rear_filt = points[:, 0] > -x_radius * 2
    x_filt = np.logical_and(x_front_filt, x_rear_filt)
    #x_filt = np.abs(points[:, 0]) < x_radius
    y_filt = np.abs(points[:, 1]) < y_radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    #points = points[:, not_close]
    return not_close

def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    w, l, h = self.wlh * wlh_factor

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(self.orientation.rotation_matrix, corners)

    # Translate
    x, y, z = self.center
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners

def filter_points(points, points_labels_all):
    #points_labels_all = data['labels']
    pcds = torch.tensor(points - points.min(axis=0))
    voxel_size = 0.05
    pcds_voxel = torch.div(pcds, voxel_size, rounding_mode='floor').long()
    _device = pcds_voxel.device
    k_type = torch.int64
    table_size = pcds_voxel.shape[0] * 2
    _max_voxel = pcds_voxel.max(axis=0)[0]
    spatial_shape = [_max_voxel[0].item(), _max_voxel[1].item(), _max_voxel[2].item()]
    table = HashTable(_device, k_type, torch.int64, table_size)
    scalar = _indice_to_scalar(pcds_voxel, spatial_shape)
    index = torch.arange(0, scalar.shape[0], device=_device).long()
    # insert labeled points first
    semantic_labels = points_labels_all
    mask = semantic_labels != 0
    table.insert(scalar[mask], index[mask])
    # insert all points
    table.insert(scalar, index)
    # filter points in single voxel
    keeped_index = table.items()[1]
    keeped_index = keeped_index.numpy()
    points = points[keeped_index]
    points_labels_all = points_labels_all[keeped_index]
    return points, points_labels_all


# @print_run_time
def cal_icp_transformation(seq_dict, previous_k = 50, max_iteration=1000, max_correspondence_distance=1.0, seq_idx=-1):
    # ATTENTION: set max_correspondence_distance to small value, 0.1 eg, ICP not converage and road not in same height
    # import in head cause parse_data error
    #from mmdet3d.core.visualizer.open3d_vis import Visualizer
    from utils import process

    seq_pcd = None
    previous_frame_list = []
    for frame_idx, data in seq_dict.items():
        # if frame_idx != 156: continue
        start = time.time()
        pcd = data['points_all']
        lidar2vehicle = data['lidar2vehicle']
        vehicle2global = data['vehicle2global']
        bboxes = data['bboxes'] # vehicle coordinate
        labels = data['labels']

        # remove motion points by bbox
        pcd_vehicle = (_expand_dim(pcd[:, :3]) @ lidar2vehicle.T)[:, :3]
        if len(bboxes) > 0:
            inbox = points_in_rbbox(pcd_vehicle[:, :3], bboxes)
            inbox = inbox.sum(axis=-1).reshape(-1) > 0
            pcd = pcd[inbox == False]
        if DEBUG and frame_idx%50==0:
            
            points = pcd_vehicle
            _min = points.min(axis=0)
            points = points - _min
            bboxes[:,:3] -= _min.reshape(-1, 3)
            pts_color = np.zeros((points.shape[0], 3))
            pts_color[inbox,0] = 255
            points_with_colors = np.concatenate([points[:, :3], pts_color], axis=-1)
            vis = Visualizer(points_with_colors, mode='xyzrgb')
            gt_bboxes_3d = bboxes
            vis.add_bboxes(bbox3d=gt_bboxes_3d, bbox_color=(0, 1, 0))
            vis.show()
            del vis

        
        # ICP
        trans_pcd = (_expand_dim(pcd[:, :3]) @ lidar2vehicle.T @ vehicle2global.T)[:, :3]
        trans_pcd = np.concatenate((trans_pcd, pcd[:,3:]), axis=-1)
        if seq_pcd is None:
            seq_pcd = trans_pcd
            icp_transformation = np.eye(4)
        else:
            min_ = trans_pcd[:,:3].min(axis=0)
            max_ = trans_pcd[:,:3].max(axis=0)
            _previous_frame_list = previous_frame_list
            if len(_previous_frame_list) > previous_k:
                _previous_frame_list = previous_frame_list[-previous_k:]
            pcd_t = np.concatenate(_previous_frame_list, axis=0)
            pcd_t_inrange = np.logical_and.reduce((pcd_t[:,0]>min_[0], pcd_t[:,1]>min_[1], pcd_t[:,0]<max_[0], pcd_t[:,1]<max_[1]))
            pcd_t = pcd_t[pcd_t_inrange]
            pcd_s = process.np2pcd(trans_pcd[:,:3])
            pcd_t = process.np2pcd(pcd_t[:,:3])
            # data['pcd_s'] = process.pcd2np(pcd_s)
            # data['pcd_t'] = process.pcd2np(pcd_t)
            # if frame_idx == 181:
            #     breakpoint()
            pcd_t.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.2, max_nn=50))
            pcd_t.normalize_normals()
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd_s, pcd_t, max_correspondence_distance, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
            )
            print('seq_idx: {} frameid:{} time cost:{}, icp fitness:{}'.format(seq_idx, frame_idx, time.time()-start, reg_p2p.fitness))
            if reg_p2p.fitness > 0.9:
                trans_pcd = pcd_s.transform(reg_p2p.transformation)
                trans_pcd = process.pcd2np(trans_pcd)
                icp_transformation = reg_p2p.transformation
            else:
                icp_transformation = np.eye(4)
                # DO NOT KNOW WHY ICP FAIL
                # raise NotImplementedError
                # disable all icp_transformation
                print('WARNING icp fail, file ', seq_idx)
                for frame_idx, data in seq_dict.items():
                    data['icp_transformation'] = np.eye(4)
                return seq_dict

            seq_pcd = np.concatenate((seq_pcd, trans_pcd), axis=0)

        data['icp_transformation'] = icp_transformation
        previous_frame_list.append(trans_pcd)


    return seq_dict

def split_tfrecord(waymo_format_path, split):
    """
    return train, val splits of waymo row dataset
    """
    splits = ['training', 'validation']
    if split == 'training':
        return sorted(glob(osp.join(waymo_format_path, 'training', '*.tfrecord')))
    else:
        return sorted(glob(osp.join(waymo_format_path, 'validation', '*.tfrecord')))

def handle_scene_single_core(samples_set, train_split,split):
    if split =='training':
        for train_seq in samples_set:

            # step2: fuse sequence, icp inside
            print(f'train_seq:{train_seq}')
            if UPDATE or not os.path.exists(osp.join(output_dir, f'seq_dict_icp_{str(train_seq).zfill(3)}.pkl')):
                with open(osp.join(output_dir, f'seq_dict_scratch_{str(train_seq).zfill(3)}.pkl'), 'rb') as f:
                    seq_dict = pkl.load(f)
                seq_dict = cal_icp_transformation(seq_dict, seq_idx=train_seq)
                with open(osp.join(output_dir, f'seq_dict_icp_{str(train_seq).zfill(3)}.pkl'), 'wb') as f:
                    pkl.dump(seq_dict, f)
            
            # Check if pkl data was truncated at the same time, if so, delete it and retry
            elif Path(output_dir + f'seq_dict_icp_{str(train_seq).zfill(3)}.pkl').exists():

                try:

                    file_path = os.path.join(output_dir + f'seq_dict_icp_{str(train_seq).zfill(3)}.pkl')

                    with open(file_path, 'rb') as file:

                        pkl.load(file)

                except pkl.UnpicklingError:

                    print(f"seq {train_seq} is truncated, retrying")

                    os.remove(os.path.join(output_dir + f'seq_dict_icp_{str(train_seq).zfill(3)}.pkl'))
                    with open(osp.join(output_dir, f'seq_dict_scratch_{str(train_seq).zfill(3)}.pkl'), 'rb') as f:
                        seq_dict = pkl.load(f)
                    seq_dict = cal_icp_transformation(seq_dict, seq_idx=train_seq)
                    with open(osp.join(output_dir, f'seq_dict_icp_{str(train_seq).zfill(3)}.pkl'), 'wb') as f:
                        pkl.dump(seq_dict, f)

                except Exception as e:

                    print(f"seq {train_seq} error, retrying")

                    os.remove(os.path.join(output_dir + f'seq_dict_icp_{str(train_seq).zfill(3)}.pkl'))

                    with open(osp.join(output_dir, f'seq_dict_scratch_{str(train_seq).zfill(3)}.pkl'), 'rb') as f:
                        seq_dict = pkl.load(f)
                    seq_dict = cal_icp_transformation(seq_dict, seq_idx=train_seq)
                    with open(osp.join(output_dir, f'seq_dict_icp_{str(train_seq).zfill(3)}.pkl'), 'wb') as f:
                        pkl.dump(seq_dict, f)

            else:
                print('skip')
    elif split =='validation':
        for train_seq in samples_set:

            # step2: fuse sequence, icp inside
            print(f'train_seq:{train_seq}')
            if UPDATE or not os.path.exists(osp.join(output_dir, f'seq_dict_icp_{str(train_seq+798).zfill(3)}.pkl')):
                with open(osp.join(output_dir, f'seq_dict_scratch_{str(train_seq+798).zfill(3)}.pkl'), 'rb') as f:
                    seq_dict = pkl.load(f)
                seq_dict = cal_icp_transformation(seq_dict, seq_idx=train_seq)
                with open(osp.join(output_dir, f'seq_dict_icp_{str(train_seq+798).zfill(3)}.pkl'), 'wb') as f:
                    pkl.dump(seq_dict, f)
            # Check if pkl data was truncated at the same time, if so, delete it and retry
            elif Path(output_dir + f'seq_dict_icp_{str(train_seq+798).zfill(3)}.pkl').exists():

                try:

                    file_path = os.path.join(output_dir + f'seq_dict_icp_{str(train_seq+798).zfill(3)}.pkl')

                    with open(file_path, 'rb') as file:

                        pkl.load(file)

                except pkl.UnpicklingError:

                    print(f"seq {train_seq+798} is truncated, retrying")

                    os.remove(os.path.join(output_dir + f'seq_dict_icp_{str(train_seq+798).zfill(3)}.pkl'))
                    with open(osp.join(output_dir, f'seq_dict_scratch_{str(train_seq+798).zfill(3)}.pkl'), 'rb') as f:
                        seq_dict = pkl.load(f)
                    seq_dict = cal_icp_transformation(seq_dict, seq_idx=train_seq)
                    with open(osp.join(output_dir, f'seq_dict_icp_{str(train_seq+798).zfill(3)}.pkl'), 'wb') as f:
                        pkl.dump(seq_dict, f)

                except Exception as e:

                    print(f"seq {train_seq+798} error, retrying")

                    os.remove(os.path.join(output_dir + f'seq_dict_icp_{str(train_seq+798).zfill(3)}.pkl'))
                    with open(osp.join(output_dir, f'seq_dict_scratch_{str(train_seq+798).zfill(3)}.pkl'), 'rb') as f:
                        seq_dict = pkl.load(f)
                    seq_dict = cal_icp_transformation(seq_dict, seq_idx=train_seq)
                    with open(osp.join(output_dir, f'seq_dict_icp_{str(train_seq+798).zfill(3)}.pkl'), 'wb') as f:
                        pkl.dump(seq_dict, f)

            else:
                print('skip')

def main():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--split', type=str, default='training')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=798)
    args = parser.parse_args()
    # add multi-process
    print(args.start, args.end)
    train_split = split_tfrecord(tfrecord_dir, args.split)
    print('generate aggegated point cloud for training sequences...')
    cpu_num = CPU_NUM
    samples_split = np.array_split(np.arange(args.start, args.end), cpu_num)
    print("Number of cores: {}, sequences per core: {}".format(cpu_num, len(samples_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    pbar = tqdm(total=args.end - args.start)
    def update(*a):
        pbar.update()
    for proc_id, samples_set in enumerate(samples_split):
        p = workers.apply_async(handle_scene_single_core,
                                args=(samples_set, train_split,args.split), callback=update)
        processes.append(p)
    workers.close()
    workers.join()


def debug():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--split', type=str, default='training')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=798)
    args = parser.parse_args()
    # add multi-process
    print(args.start, args.end)
    train_split = split_tfrecord(tfrecord_dir, args.split)
    handle_scene_single_core([0], train_split,args.split)


if __name__ == '__main__':
    main()
