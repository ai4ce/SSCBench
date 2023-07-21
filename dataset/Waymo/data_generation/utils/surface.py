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
API: agg_bg
    src_dir : output_dir/seq_dict_icp_XXX.pkl
    dst_dir : stuff_dir/vertices_XXX.pkl stuff_dir/seq_dict_XXX.pkl 
    func: aggregate stuff points, assign labels to unlabeled points, find general object
    others: points in global + ICP coordinate ((_expand_dim(pcds[:, :3]) @ lidar2vehicle.T @ vehicle2global.T @ icp_transformation.T)[:, :3])

API: agg_object
    src_dir : output_dir/seq_dict_icp_XXX.pkl
    dst_dir : object_dir/vertices_XXX.pkl object_dir/objects_bank_XXX.pkl
    func: extract points in each box, aggregate points belong to same instance id, calulate icp transformation cross points belong to same instance id
    others: points in box + ICP coordinate ((_expand_dim(pcds[:, :3]) @ lidar2vehicle.T @ vehicle2box.T @ icp_transformation.T)[:, :3])

'''

import argparse
import os
import os.path as osp
import sys
import pickle
from typing import List
import numpy as np
import pickle as pkl
import torch
# import vdbfusion
from sklearn.neighbors import KDTree
import multiprocessing
from functools import reduce
multiprocessing.set_start_method('forkserver', force=True)
from tqdm import tqdm, trange
from spconv.pytorch.hash import HashTable
from sklearn.neighbors import KNeighborsClassifier
# from mmdet3d.core.bbox.box_np_ops import points_in_rbbox, rotation_3d_in_axis
#from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode
#from mmdet3d.core.visualizer.open3d_vis import Visualizer
# from utils import vistool
import open3d as o3d # not known why, must import torch before open3d, othewise GPU matmul fail
# from trimesh.viewer import windowed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from utils import process
from utils.config import *
from utils.points_in_bbox import points_in_rbbox
from utils.custom import to_stride, _indice_to_scalar

DEBUG = False
VIS = False
WAYMO_THING_LABELS = [1, 2, 3, 4, 5, 6, 7, 8]
WAYMO_ROAD_LABEL_START = 17
def _expand_dim(array):
    return np.concatenate((array, np.ones_like(array)[:, :1]), axis=1)

def get_vehicle2box(box):
    # ISSUE, only checked for waymo defination
    center = box[:3]
    angles = -box[-1]

    trans = np.eye(4)
    trans[3,:3] = -center

    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                          [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    rot_mat = np.eye(4)
    rot_mat[:3,:3] = rot_mat_T

    vehicle2box = trans @ rot_mat
    # vehicle2box[:3,-1] = -center
    # vehicle2box[3,3] = 1
    return vehicle2box.T

def my_query(pcd, labels, k=150, leaf_size=10):
    pcd_tree = KDTree(pcd[:, :3], leaf_size)
    undefined_mask = labels == 0
    pcd_undefined = pcd[undefined_mask, :]
    pos, idx = pcd_tree.query(pcd_undefined[:, :3], k=k)
    #labels = np.concatenate((labels, pred_labels)).astype(int)
    #return np.concatenate((points_with_labels, points_wo_labels), axis=0), labels
    keeped_mask = np.all(labels[idx]==0, axis=1)
    return pcd_undefined[keeped_mask, :]

class WaymoDataset:
    def __init__(self, seq_dict):
        # Initialize your dataset here ..
        self.seq_dict = seq_dict

    def __len__(self):
        return len(self.seq_dict)

    def __getitem__(self, idx: int):
        data = self.seq_dict[idx]
        pcds:np.array = data['points_all'][:,:3] # lidar coordinate
        pcd_labels = data['points_labels_all']
        lidar2vehicle = data['lidar2vehicle']
        vehicle2global = data['vehicle2global']
        icp_transformation = data['icp_transformation']

        remove_dynamic = True
        if remove_dynamic:
            bboxes = data['bboxes'] # vehicle coordinate
            labels = data['labels']
            pcds_invehicle = (_expand_dim(pcds[:, :3]) @ lidar2vehicle.T)[:, :3]
            if len(bboxes) > 0:
                inbox = points_in_rbbox(pcds_invehicle[:, :3], bboxes)
                inbox = inbox.sum(axis=-1).reshape(-1) > 0
                if DEBUG and idx % 25 == 0:
                    points = pcds_invehicle
                    _min = points.min(axis=0)
                    points = points - _min
                    bboxes[:, :3] -= _min.reshape(-1, 3)
                    pts_color = np.zeros((points.shape[0], 3))
                    pts_color[inbox, 0] = 255
                    points_with_colors = np.concatenate([points[:, :3], pts_color], axis=-1)
                    vis = Visualizer(points_with_colors, mode='xyzrgb')
                    gt_bboxes_3d = bboxes
                    vis.add_bboxes(bbox3d=gt_bboxes_3d, bbox_color=(0, 1, 0))
                    vis.show()
                    del vis
                pcds = pcds[inbox == False]
                pcd_labels = pcd_labels[inbox == False]

        # lidar2vehicle = np.eye(4)
        # vehicle2global = np.eye(4)
        # icp_transformation = np.eye(4)

        origin = np.zeros((1,3))

        trans_pcds = (_expand_dim(pcds[:, :3]) @ lidar2vehicle.T @ vehicle2global.T @ icp_transformation.T)[:, :3]
        origin = (_expand_dim(origin[:, :3]) @ lidar2vehicle.T @ vehicle2global.T @ icp_transformation.T)[:, :3]
        points = trans_pcds.astype(np.float64) # global coordinate
        origin = origin[0]
        return points, origin, pcd_labels


class WaymoDatasetObject:
    def __init__(self, seq_dict):
        # Initialize your dataset here ..
        objects_bank = {} # key: object id, value: frame id
        for frameid, frame in seq_dict.items():
            bboxes = frame['bboxes']
            labels = frame['labels']
            bboxes_id = frame['bboxes_id']
            for bbox, label, bbox_id in zip(bboxes, labels, bboxes_id):
                if bbox_id not in objects_bank:
                    objects_bank[bbox_id] = []
                objects_bank[bbox_id].append(
                    {
                        'frameid': frameid,
                        'bbox': bbox,
                        'label': label,
                        'bbox_id': bbox_id,
                    }
                )
        self.objects_bank = objects_bank
        self.seq_dict = seq_dict

    def sort(self, object_id):
        # sort by inside points num
        object_infos = self.objects_bank[object_id]
        for object_info in object_infos:
            frameid = object_info['frameid']
            bbox = object_info['bbox'] # vehicle coordinate
            label = object_info['label']
            pcds:np.array = self.seq_dict[frameid]['points_all'][:,:3] # lidar coordinate
            lidar2vehicle:np.array = self.seq_dict[frameid]['lidar2vehicle']
            # vehicle2global:np.array = self.seq_dict[frameid]['vehicle2global']
            # icp_transformation:np.array = self.seq_dict[frameid]['icp_transformation']
            vehicle2box: np.array = get_vehicle2box(bbox)

            # crop object points in vehicle coordinate
            pcds_invehicle = (_expand_dim(pcds[:, :3]) @ lidar2vehicle.T)[:, :3]
            if len(bbox) > 0:
                inbox = points_in_rbbox(pcds_invehicle[:, :3],
                                        bbox.reshape(-1, 7))  # ISSUE, only check for waymo yaw define
                inbox = inbox.sum(axis=-1).reshape(-1) > 0
                object_info['points_num'] = inbox.sum()
            else:
                object_info['points_num'] = 0

        object_infos = sorted(object_infos, key=lambda x: x['points_num'], reverse=True)
        return object_infos

    def icp(self, object_id, radius=0.2, max_nn=50, max_iteration=1000, fitness=0.9, max_correspondence_distance=0.1):
        seq_pcds = None
        object_infos = self.sort(object_id) # sort frame by points num inside box
        for info_idx, object_info in enumerate(object_infos):
            # object_info = self.objects_bank[object_id][idx]
            frameid = object_info['frameid']
            bbox = object_info['bbox'] # vehicle coordinate
            label = object_info['label']

            pcds:np.array = self.seq_dict[frameid]['points_all'][:,:3] # lidar coordinate
            lidar2vehicle:np.array = self.seq_dict[frameid]['lidar2vehicle']
            # vehicle2global:np.array = self.seq_dict[frameid]['vehicle2global']
            # icp_transformation:np.array = self.seq_dict[frameid]['icp_transformation']
            vehicle2box: np.array = get_vehicle2box(bbox)
            object_info['lidar2vehicle'] = lidar2vehicle
            object_info['vehicle2box'] = vehicle2box

            # crop object points in vehicle coordinate
            pcds_invehicle = (_expand_dim(pcds[:, :3]) @ lidar2vehicle.T)[:, :3]
            if len(bbox) > 0:
                inbox = points_in_rbbox(pcds_invehicle[:, :3],
                                        bbox.reshape(-1, 7))  # ISSUE, only check for waymo yaw define
                inbox = inbox.sum(axis=-1).reshape(-1) > 0
                pcds = pcds[inbox]  # lidar coordinate

            # transfer to box coordinate
            origin = np.zeros((1,3))
            pcds_inbox = (_expand_dim(pcds[:, :3]) @ lidar2vehicle.T @ vehicle2box.T)[:, :3]
            origin = (_expand_dim(origin[:, :3]) @ lidar2vehicle.T @ vehicle2box.T)[:, :3]

            # ICP
            trans_pcds = pcds_inbox
            icp_transformation = np.eye(4)
            fitness = 1.
            if seq_pcds is None:
                seq_pcds = trans_pcds
            elif len(trans_pcds)==0:
                pass
            elif len(seq_pcds)==0:
                pass
            else:
                # TODO enable icp
                # pcds_t = process.np2pcd(seq_pcds[:, :3])
                # pcds_s = process.np2pcd(trans_pcds[:, :3])
                # pcds_t.estimate_normals(
                #     search_param=o3d.geometry.KDTreeSearchParamHybrid(
                #         radius=radius, max_nn=max_nn))
                # pcds_t.normalize_normals()
                #
                # reg_p2p = o3d.pipelines.registration.registration_icp(
                #     pcds_s, pcds_t, max_correspondence_distance, np.eye(4),
                #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
                # )
                # fitness = reg_p2p.fitness
                # # print(fitness, "#########")
                # if fitness > fitness:
                #     trans_pcds = pcds_s.transform(reg_p2p.transformation)
                #     trans_pcds = process.pcd2np(trans_pcds)
                #     icp_transformation = reg_p2p.transformation
                seq_pcds = np.concatenate((seq_pcds, trans_pcds), axis=0)
            object_info['icp_transformation'] = icp_transformation
            object_info['fitness'] = fitness
            # if info_idx == 5: break
        return seq_pcds

    def __getitem__(self, object_id):
        object_infos = self.objects_bank[object_id]
        # object_infos = self.sort(object_id)
        for info_idx, object_info in enumerate(object_infos):
            # object_info = self.objects_bank[object_id][idx]
            frameid = object_info['frameid']
            bbox = object_info['bbox'] # vehicle coordinate
            label = object_info['label']
            icp_transformation:np.array = object_info['icp_transformation']
            # fitness = object_info['fitness']
            # if fitness < 0.99: continue

            pcds:np.array = self.seq_dict[frameid]['points_all'][:,:3] # lidar coordinate
            # pcds_labels: np.array = self.seq_dict[frameid]['points_labels_all']
            lidar2vehicle:np.array = self.seq_dict[frameid]['lidar2vehicle']
            vehicle2box: np.array = get_vehicle2box(bbox)

            # crop object points in vehicle coordinate
            pcds_invehicle = (_expand_dim(pcds[:, :3]) @ lidar2vehicle.T)[:, :3]
            if len(bbox) > 0:
                inbox = points_in_rbbox(pcds_invehicle[:, :3],
                                        bbox.reshape(-1, 7))  # ISSUE, only check for waymo yaw define
                inbox = inbox.sum(axis=-1).reshape(-1) > 0
                pcds = pcds[inbox]  # lidar coordinate
            # pcds_labels = pcds_labels[inbox]

            # generate pcds and origin for vdb fusion
            # lidar2vehicle = np.eye(4)
            # vehicle2box = np.eye(4)
            # icp_transformation = np.eye(4)

            # transfer to box coordinate
            origin = np.zeros((1,3))
            trans_pcds = (_expand_dim(pcds[:, :3]) @ lidar2vehicle.T @ vehicle2box.T @ icp_transformation.T)[:, :3]
            origin = (_expand_dim(origin[:, :3]) @ lidar2vehicle.T @ vehicle2box.T @ icp_transformation.T)[:, :3]
            scan = trans_pcds.astype(np.float64) # box coordinate
            pose = origin[0]

            # pose = np.random.randn(*pose.shape)
            # pose = np.zeros_like(pose)
            if label == 1: # vehicle
                pose = np.zeros_like(pose) # fake origin : bottom center of box
            elif label == 3: # sign
                # fake origin
                pose = np.zeros_like(pose)
                pose[0] = 10
                pose[2] = bbox[5]/2
            else:
                pass # TODO check for ped/cyclist
            # if info_idx == 5: break
            yield scan, pose


def agg_object(dataset: WaymoDatasetObject, train_seq):
    sdf_voxelsize = {
        0:0.1, # TYPE_UNKNOWN
        1:0.1, # TYPE_VEHICLE
        2:0.1, # TYPE_PEDESTRIAN
        3:0.2, # TYPE_SIGN
        4:0.1, # TYPE_CYCLIST
    }
    sdf_trunc = {
        0:0.3, # TYPE_UNKNOWN
        1:0.3, # TYPE_VEHICLE
        2:0.3, # TYPE_PEDESTRIAN
        3:0.4, # TYPE_SIGN
        4:0.3, # TYPE_CYCLIST
    }
    space_carving=False
    fill_holes=True
    min_weight=5

    catch_debug = True
    scene_vertices = {}
    for object_idx, object_id in tqdm(enumerate(dataset.objects_bank.keys())):
        # if object_id == 'UaaN3iabHjJDdfH_XomoNQ':
        #     catch_debug=True
        # if not catch_debug: continue
        object_type = dataset.objects_bank[object_id][0]['label']
        print("object id: ", object_id, object_type)
        #vdb_volume = vdbfusion.VDBVolume(sdf_voxelsize[object_type], sdf_trunc[object_type], space_carving)
        dataset.icp(object_id, max_iteration=1000, fitness=0.9)  # do ICP
        seq_pcds = []
        seq_pose = []
        for scan, pose in dataset.__getitem__(object_id):
            if len(scan)>0:
                #vdb_volume.integrate(scan, pose)
                seq_pcds.append(scan)
                seq_pose.append(pose)
        if len(seq_pcds) == 0:
            print("WARNING object {} empty points".format(object_id))
            continue
        seq_pcds = np.concatenate(seq_pcds, axis=0)
        seq_pose = np.stack(seq_pose, axis=0)
        # Extract a mesh from vdbfusion
        #vertices, triangles = vdb_volume.extract_triangle_mesh(fill_holes=fill_holes, min_weight=min_weight)
        #pass  # remove isolate surface, need visulize
        #if len(vertices) == 0:  # special case: sometimes mesh is empty even have points
        vertices = seq_pcds
        triangles = np.arange(0, vertices.shape[0])
        triangles = np.stack([triangles, triangles, triangles], axis=-1)
        # mask = seq_labels < 14
        # seq_pcds = seq_pcds[mask, :]
        # seq_labels = seq_labels[mask]
        # scene_vertices[object_id] = {'vertices': seq_pcds, 'labels': seq_labels};
        scene_vertices[object_id] = {'vertices': vertices, 'labels': np.full_like(vertices[:, 0:1], object_type)}
        # How to invert vertices to each frame ?
        # since: vertices = (_expand_dim(pcds[:, :3]) @ lidar2vehicle.T @ vehicle2box.T @ icp_transformation.T)[:, :3]
        # so: pcd_inframe = (_expand_dim(vertices[:, :3]) @ np.linalg.inv(lidar2vehicle.T @ vehicle2box.T @ icp_transformation.T)
        # if object_idx == 5: break
    print(f"seq {train_seq} now saving")
    if not os.path.exists(object_dir): os.makedirs(object_dir)
    with open(os.path.join(object_dir, f'vertices_{str(train_seq).zfill(3)}.pkl'), 'wb') as f:
        pickle.dump(scene_vertices, f)
    with open(os.path.join(object_dir, f'objects_bank_{str(train_seq).zfill(3)}.pkl'), 'wb') as f:
        pickle.dump(dataset.objects_bank, f)
    # TODO: store scene_vertices & dataset.objects_bank[object_id]

def my_knn(pcd, labels):
    model = KNeighborsClassifier(n_neighbors=5)
    mask = labels != -1
    points_with_labels, seg_labels = pcd[mask, :3], labels[mask]
    points_wo_labels = pcd[mask == False, :3]
    model.fit(X=points_with_labels, y=seg_labels)
    pred_labels = model.predict(points_wo_labels)
    labels[mask == False] = pred_labels
    #labels = np.concatenate((labels, pred_labels)).astype(int)
    #return np.concatenate((points_with_labels, points_wo_labels), axis=0), labels
    return pcd, labels

def point_refinement(labels):
    obj_mask = reduce(np.logical_or, [labels == obj_label for obj_label in WAYMO_THING_LABELS])
    return np.logical_not(obj_mask)

def filter_points(points, points_labels_all):
    #points_labels_all = data['labels']
    pcds = torch.tensor(points - points.min(axis=0))
    frame_ = pcds[:, 3:4]
    pcds = pcds[:, :3]
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
    frame_ = frame_[keeped_index]
    points = np.concatenate((points, frame_), axis=1)
    points_labels_all = points_labels_all[keeped_index]
    return points, points_labels_all

def agg_bg(dataset, train_seq):
    voxel_size = 0.05
    sdf_trunc = 0.5
    space_carving = False
    #vdb_volume = vdbfusion.VDBVolume(voxel_size, sdf_trunc, space_carving)
    points = []
    point_labels = []
    for idx in trange(0, len(dataset)):
        # for idx in trange(0, 199):
        # try:
        #     scan, pose, labels = dataset.__getitem__(idx)
        # except: continue
        scan, pose, labels = dataset.__getitem__(idx)
        # keep = np.logical_and.reduce((scan[:,0]>mean[0]-_range, scan[:,0]<mean[0]+_range, scan[:,1]>mean[1]-_range, scan[:,1]<mean[1]+_range))
        # scan = scan[keep]
        # if len(scan) == 0: continue
        #vdb_volume.integrate(scan, pose)
        scan = np.concatenate((scan, np.full_like(scan[:, 0:1], idx)), axis=1)
        points.append(scan)
        point_labels.append(labels)
        # if idx != 5: break
    print(f"seq {train_seq} loading completed")
    # KNN & refine: be cautious about forth dim of pcd
    pcd = np.concatenate(points)
    pcd_labels = np.concatenate(point_labels)
    # First Step: devide keyframe and sweeps
    keyframe_mask = pcd_labels != -1
    # Second Step: refine points of general objects
    keeped_undefined_pcd = my_query(pcd[keyframe_mask], pcd_labels[keyframe_mask])
    print(f"seq {train_seq} query completed")
    # Third Step: remove outlier undefined points
    grd_height = pcd[pcd_labels >= WAYMO_ROAD_LABEL_START, 2].mean()
    valid_mask = np.logical_and(np.abs(keeped_undefined_pcd[:, 2] - grd_height) < 2, np.abs(keeped_undefined_pcd[:, 2] - grd_height > 0.2))
    keeped_undefined_pcd = keeped_undefined_pcd[valid_mask]
    # Fourth Step: remove original undefined points
    pcd = pcd[pcd_labels != 0]
    pcd_labels = pcd_labels[pcd_labels != 0]
    # Fifth Step: concat refined go points
    pcd = np.concatenate((pcd, keeped_undefined_pcd), axis=0)
    pcd_labels = np.concatenate((pcd_labels, np.zeros_like(keeped_undefined_pcd[:, 0], dtype=np.int8)))
    # Sixth Step: KNN
    frame_ = pcd[:, 3:4]
    pcd, pcd_labels = my_knn(pcd[:, :3], pcd_labels)
    pcd = np.concatenate((pcd, frame_), axis=1)
    stuff_mask = point_refinement(pcd_labels)
    pcd = pcd[stuff_mask, :]
    pcd_labels = pcd_labels[stuff_mask]
    pcd, pcd_labels = filter_points(pcd, pcd_labels)
    print(f"seq {train_seq} now saving")
    if not os.path.exists(stuff_dir): os.makedirs(stuff_dir)
    with open(Path(stuff_dir) / f'vertices_{str(train_seq).zfill(3)}.pkl', 'wb') as f:
        pickle.dump({
            'vertices': pcd[:, :4],
            'labels': pcd_labels,
        }, f)
    with open(Path(stuff_dir) / f'seq_dict_{str(train_seq).zfill(3)}.pkl', 'wb') as f:
        pickle.dump(dataset.seq_dict, f)

def handle_scene_single_core(samples_set):
    for train_seq in samples_set:
        print(f'train_seq: {train_seq}')
        done = os.path.exists(osp.join(stuff_dir, f'vertices_{str(train_seq).zfill(3)}.pkl')) and os.path.exists(osp.join(stuff_dir, f'seq_dict_{str(train_seq).zfill(3)}.pkl'))
        if UPDATE or not done:
            print('process stuff...')
            try:
                with open(os.path.join(output_dir, f'seq_dict_icp_{str(train_seq).zfill(3)}.pkl'), 'rb') as f:
                    seq_dict = pkl.load(f)
            except Exception as e:
                print("*"*20)
                print(f'seq {train_seq} icp file : error - {str(e)}')
                print("*"*20)
            dataset_stuff = WaymoDataset(seq_dict)
            agg_bg(dataset_stuff, train_seq)
        else:
            print('skip stuff', train_seq)

        done = os.path.exists(osp.join(object_dir, f'vertices_{str(train_seq).zfill(3)}.pkl')) and os.path.exists(osp.join(object_dir, f'objects_bank_{str(train_seq).zfill(3)}.pkl'))
        if UPDATE or not done:
            print('process instance...')
            try:
                with open(os.path.join(output_dir, f'seq_dict_icp_{str(train_seq).zfill(3)}.pkl'), 'rb') as f:
                    seq_dict2 = pkl.load(f)
            except Exception as e:
                print("*"*20)
                print(f'seq {train_seq} icp file : error - {str(e)}')
                print("*"*20)
            dataset = WaymoDatasetObject(seq_dict2)
            agg_object(dataset, train_seq)
        else:
            print('skip instance', train_seq)

def main():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=202)
    args = parser.parse_args()
    cpu_num = 80
    samples_split = np.array_split(np.arange(args.start, args.end), cpu_num)
    print("Number of cores: {}, sequences per core: {}".format(cpu_num, len(samples_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    pbar = tqdm(total=args.end - args.start)
    def update(*a):
        pbar.update()
    for proc_id, samples_set in enumerate(samples_split):
        p = workers.apply_async(handle_scene_single_core,
                                args=(samples_set,))
        processes.append(p)
    workers.close()
    workers.join()

def debug():
    handle_scene_single_core([0])

if __name__ == '__main__':
    main()
