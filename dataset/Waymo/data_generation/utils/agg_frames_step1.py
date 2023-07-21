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

import multiprocessing

multiprocessing.set_start_method('forkserver', force=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

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

def parse_data(FILENAME = os.path.join(tfrecord_dir, 'training', 'segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'), tfrecord_idx=0,split='training'):

    # if osp.exists(osp.join(output_dir, f'seq_dict_scratch_{tfrecord_idx}.pkl')): return

    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

    seq_dict = OrderedDict()

    for frame_idx, data in tqdm(enumerate(dataset)):

        # if frame_idx <  156: continue

        frame = open_dataset.Frame()

        frame.ParseFromString(bytearray(data.numpy()))

        # read pose

        lidar2vehicle = np.array(frame.context.laser_calibrations[0].extrinsic.transform).reshape(4, 4)

        vehicle2global = np.array(frame.pose.transform).reshape(4, 4)

        # read points

        (range_images, camera_projections,

        seg_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(

            frame)

        # if not seg_labels: continue

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(

            frame,

            range_images,

            camera_projections,

            range_image_top_pose)

        points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(

            frame,

            range_images,

            camera_projections,

            range_image_top_pose,

            ri_index=1)

        

        print(f'Successfully loaded seq {tfrecord_idx} frame {frame_idx}')



        # 3d points in vehicle frame.

        points_all = np.concatenate(points, axis=0)

        points_all_ri2 = np.concatenate(points_ri2, axis=0)

        # camera projection corresponding to each point.

        # cp_points_all = np.concatenate(cp_points, axis=0)

        # cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

        point_labels = convert_range_image_to_point_cloud_labels(

            frame, range_images, seg_labels)

        point_labels_all = np.concatenate(point_labels, axis=0)[:, -1]

        bboxes, labels, bboxes_id = get_bbox(frame) # bbox in vehicle frame

        # box_seg_labels = np.full_like(bboxes[:, 0], -1)

        # if len(seg_labels) > 0:

        #     indices = points_in_rbbox(points_all[:, :3], bboxes)

        #     box_indices = np.argwhere(indices==True, )

        #     for box_id in range(box_indices.shape[-1]):

        #         try:

        #             box_pcd = box_indices[box_indices[:, 1] == box_id][:, 0]

        #             seg_label = np.argmax(np.bincount(point_labels_all[box_pcd]))

        #             box_seg_labels[box_id] = seg_label

        #         except:

        #             box_seg_labels[box_id] = -1

        # convert pcd from vehicle frame to lidar frame





        print(f'seq {tfrecord_idx} frame {frame_idx} label loaded')





        points_all =  (_expand_dim(points_all[:, :3]) @ np.linalg.inv(lidar2vehicle).T)[:, :3]

        all_mask = my_remove_close(points_all, x_radius=4.0, y_radius=2.0)

        points_all = points_all[all_mask, :]

        points_all_ri2 =  (_expand_dim(points_all_ri2[:, :3]) @ np.linalg.inv(lidar2vehicle).T)[:, :3]

        all_ri2_mask = my_remove_close(points_all_ri2, x_radius=4.0, y_radius=2.0)

        points_all_ri2 = points_all_ri2[all_ri2_mask, :]

        # points_all = points_all[:, :3]

        # points_all_ri2 = points_all_ri2[:, :3]



        point_labels_ri2 = convert_range_image_to_point_cloud_labels(

            frame, range_images, seg_labels, ri_index=1)

        point_labels_all = point_labels_all[all_mask]

        point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)[:, -1]

        point_labels_all_ri2 = point_labels_all_ri2[all_ri2_mask]

        # point_labels = np.concatenate((point_labels_all[:, 1], point_labels_all_ri2[:, 1]))



        print(f'seq {tfrecord_idx} frame {frame_idx} label loaded 22')



        assert (point_labels_all.shape[0] == points_all.shape[0])

        assert (point_labels_all_ri2.shape[0] == points_all_ri2.shape[0])



        if len(seg_labels) == 0:

            point_labels_all = np.full_like(points_all[:, 0], -1)

            point_labels_all_ri2 = np.full_like(points_all_ri2[:, 0], -1)



        # points_all = np.concatenate((points_all, np.full_like(points_all[:, 0:1], frame_idx)), axis=1)

        # points_all_ri2 = np.concatenate((points_all_ri2, np.full_like(points_all_ri2[:, 0:1], frame_idx)), axis=1)

        origin = (np.array([[0, 0, 0, 1]]) @ lidar2vehicle.T @ vehicle2global.T)[:, :3]



        # points_all, point_labels_all = filter_points(points_all, point_labels_all)

        # print('test')

        seq_dict[frame_idx] = {

            'points_all': points_all,

            'points_all_ri2': points_all_ri2,

            'points_labels_all': point_labels_all,

            'points_labels_all_ri2': point_labels_all_ri2,

            'lidar2vehicle': lidar2vehicle,

            'vehicle2global': vehicle2global,

            'bboxes': bboxes,

            'labels': labels,

            'bboxes_id': bboxes_id,

            'origin': origin,

            'seg_labels': len(seg_labels) > 0,

            #'box_seg_labels': box_seg_labels,

        }

        print(f"seq:{tfrecord_idx} frame:{frame_idx} process completed")



        #if frame_idx == 30: break



    if not os.path.exists(output_dir): os.makedirs(output_dir)

    if split=='training':

        print(f'Now saving :{seq_dict}')

        with open(osp.join(output_dir, f'seq_dict_scratch_{str(tfrecord_idx).zfill(3)}.pkl'), 'wb') as f:

            pkl.dump(seq_dict, f)

    elif split=='validation':

        with open(osp.join(output_dir, f'seq_dict_scratch_{str(tfrecord_idx+798).zfill(3)}.pkl'), 'wb') as f:

            pkl.dump(seq_dict, f)

    return seq_dict



# @print_run_time





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

            # step1: parse data

            print(f'train_seq:{train_seq}')

            if UPDATE or not Path(output_dir + f'seq_dict_scratch_{str(train_seq).zfill(3)}.pkl').exists():

                seq_dict = parse_data(train_split[train_seq], train_seq)

            # Check if pkl data was truncated at the same time, if so, delete it and retry
            elif Path(output_dir + f'seq_dict_scratch_{str(train_seq).zfill(3)}.pkl').exists():

                try:

                    file_path = os.path.join(output_dir + f'seq_dict_scratch_{str(train_seq).zfill(3)}.pkl')

                    with open(file_path, 'rb') as file:

                        pkl.load(file)

                except pkl.UnpicklingError:
                    os.remove(os.path.join(output_dir + f'seq_dict_scratch_{str(train_seq).zfill(3)}.pkl'))

                    print(f"seq {train_seq} corrupted, retrying")

                    seq_dict = parse_data(train_split[train_seq], train_seq)

            else:

                print('skip')

    elif split =='validation':

        for train_seq in samples_set:

            # step1: parse data

            print(f'train_seq:{train_seq}')

            if UPDATE or not Path(output_dir + f'seq_dict_scratch_{str(train_seq+798).zfill(3)}.pkl').exists():

                seq_dict = parse_data(train_split[train_seq], train_seq,split)

            elif Path(output_dir + f'seq_dict_scratch_{str(train_seq+798).zfill(3)}.pkl').exists():

                try:

                    file_path = os.path.join(output_dir + f'seq_dict_scratch_{str(train_seq+798).zfill(3)}.pkl')

                    with open(file_path, 'rb') as file:

                        pkl.load(file)

                except pkl.UnpicklingError:

                    print(f"seq {train_seq} is truncated, retrying")

                    seq_dict = parse_data(train_split[train_seq], train_seq,split)

                except Exception as e:
                    print(f"seq {train_seq} error, retrying")

                    seq_dict = parse_data(train_split[train_seq], train_seq,split)

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


