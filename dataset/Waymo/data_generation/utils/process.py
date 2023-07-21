# Copyright 2022 tao.jiang
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

import argparse
import copy
import json
import os
import pickle
from typing import List

import time
import tqdm
import torch
import numpy as np
import open3d as o3d
import trimesh
# from knn_cuda import KNN
# from spconv.pytorch.hash import HashTable

os.environ['OMP_NUM_THREADS'] = '16'
from .vistool import PALETTE
from .custom import _indice_to_scalar


def np2pcd(xyz):
    assert xyz.shape[1] == 3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def pcd2np(pcd):
    return np.asarray(pcd.points)

def _expand_dim(array):
    return np.concatenate((array, np.ones_like(array)[:, :1]), axis=1)

def print_error(log):
    print(log)
    
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print('Showing outliers (red) and inliers (gray): ')
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def process_data(id, segment_name, verbose=False, visualize=False):
    if verbose:
        print('process:', id)
    pkl_path = os.path.join(args.raw_data_root, segment_name, f'{id}.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    first_frame_idx = 0
    for i, pc in enumerate(data['raw_pc']):
        if len(pc) > 50:
            first_frame_idx = i
            break
    agg_pc_obj = data['raw_pc'][first_frame_idx][:, :3]
    result_pc_list = []

    if visualize:
        raw_pcs_obj = copy.deepcopy(np.concatenate(data['raw_pc'])[:, :3])

    for i in range(first_frame_idx + 1, len(data['raw_pc'])):
        if verbose:
            begin = time.time()
        pc_obj = data['raw_pc'][i]
        if len(pc_obj) < 20 or len(data['clean_pc'][i]) < 20:
            continue
        pcd_s = np2pcd(pc_obj[:, :3])
        cl, ind = pcd_s.remove_radius_outlier(nb_points=3, radius=0.5)
        if len(ind) < 20:
            continue
        pcd_s = pcd_s.select_by_index(ind)
        pcd_t = np2pcd(agg_pc_obj[:, :3])

        pcd_t.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.2, max_nn=50))
        pcd_t.normalize_normals()

        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_s, pcd_t, 0.1, np.eye(4),
            o3d.pipelines.registration.
                TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=200))

        if reg_p2p.fitness > 0.95:
            pcd_clean = np2pcd(data['clean_pc'][i])
            pcd_clean = pcd_clean.transform(reg_p2p.transformation)
            result_pc_list.append(pcd2np(pcd_clean))
        if reg_p2p.fitness > 0.9:
            pcd_s = pcd_s.transform(reg_p2p.transformation)
            agg_pc_obj = np.concatenate([pcd2np(pcd_s), agg_pc_obj[:, :3]])
        else:
            agg_pc_obj = np.concatenate([pcd2np(pcd_s), agg_pc_obj[:, :3]])
        if verbose:
            print(i, 'cost time:', time.time() - begin)

    result_pc_obj = np.concatenate(result_pc_list)
    result_pc_pcd = np2pcd(result_pc_obj)
    cl, ind = result_pc_pcd.remove_radius_outlier(nb_points=15, radius=0.1)
    inlier_cloud = result_pc_pcd.select_by_index(ind)
    result_pc_obj = pcd2np(inlier_cloud)
    if visualize:
        # display_inlier_outlier(result_pc_pcd, ind)
        trimesh.Trimesh(result_pc_obj).export(f'/root/debug/vis_agg_pc/{id}_agg.ply')
        o3d.visualization.draw_geometries([np2pcd(result_pc_obj), np2pcd(raw_pcs_obj + np.array([0, 0, 3]))])

    output_path = os.path.join(args.agg_output_root, f'{id}.npz')
    np.savez_compressed(output_path, agg_pc=result_pc_obj)

def icp(objects_seq_dir, save_dir, do_icp=False, remove_outlier=True, visualize=False, verbose=True):
    objects_name = os.listdir(objects_seq_dir)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    for object_name in objects_name:
        print(object_name)
        points_stack = []
        object_seqs = os.path.join(objects_seq_dir, object_name)
        for file in os.listdir(object_seqs):
            file = os.path.join(object_seqs, file)
            points = torch.load(file)
            points_stack.append(points)
        
        # sort by points num
        points_stack.sort(key=lambda x: x.shape[0], reverse=True)
        agg_pc_obj = points_stack[0][:, :3]
        points_stack = points_stack[1:]
        result_pc_list = []
        for idx, pc_obj in enumerate(points_stack):
            if verbose:
                begin = time.time()
            # if len(pc_obj) < 20:
            #     continue
            pcd_s = np2pcd(pc_obj[:, :3])
            if remove_outlier:
                cl, ind = pcd_s.remove_radius_outlier(nb_points=3, radius=0.5)
                # if len(ind) < 20:
                #     continue
                pcd_s = pcd_s.select_by_index(ind)
            if do_icp: # TODO icp not help
                pcd_t = np2pcd(agg_pc_obj[:, :3])

                pcd_t.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.2, max_nn=50))
                pcd_t.normalize_normals()

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd_s, pcd_t, 0.1, np.eye(4),
                    o3d.pipelines.registration.
                        TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=200))
                if reg_p2p.fitness > 0.95:
                    pcd_clean = pcd_s
                    pcd_clean = pcd_clean.transform(reg_p2p.transformation)
                    result_pc_list.append(pcd2np(pcd_clean))
                if reg_p2p.fitness > 0.9:
                    pcd_s = pcd_s.transform(reg_p2p.transformation)
                    agg_pc_obj = np.concatenate([pcd2np(pcd_s), agg_pc_obj])
                else:
                    agg_pc_obj = np.concatenate([pcd2np(pcd_s), agg_pc_obj[:, :3]])
                if verbose:
                    print(idx, 'cost time:', time.time() - begin, "point num: {}, fitness: {}".format(pc_obj.shape[0], reg_p2p.fitness))
            else:
                agg_pc_obj = np.concatenate([pcd2np(pcd_s), agg_pc_obj[:, :3]])

        # result_pc_obj = np.concatenate(result_pc_list)
        result_pc_obj = agg_pc_obj
        if remove_outlier:
            result_pc_pcd = np2pcd(result_pc_obj)
            cl, ind = result_pc_pcd.remove_radius_outlier(nb_points=10, radius=0.3)
            inlier_cloud = result_pc_pcd.select_by_index(ind)
            result_pc_obj = pcd2np(inlier_cloud)
            
        save_file = os.path.join(save_dir, object_name+'.pt')
        torch.save(torch.tensor(result_pc_obj), save_file)
        
        if visualize:
            points = torch.cat(points_stack, dim=0)
            points_icp = torch.tensor(np.copy(agg_pc_obj))
            points_icp[:,0] += (1+points[:,0].max()-points[:,0].min())
            points = torch.cat([points, points_icp], dim=0)
            vis = Visualizer(points, mode='xyz')
            vis.show()
            del vis

def knn_label(pcds, pcds_labels, voxel_size = 0.1, max_iter=10, ignore_labels=[0,17,18,19,20,21,22],crop_labels=[1,2,3,4,5,6,7,12,13,16], visualize=False, verbose=False):
    pcds = pcds[:,:3].clone().cuda()
    pcds_labels = torch.tensor(pcds_labels).cuda().int()
    min_ = pcds.min(dim=0)[0]
    pcds = pcds - min_ # to postive

    for i in range(max_iter):
        pcds_voxel = torch.div(pcds, voxel_size, rounding_mode='floor').long()
        with_label_mask = pcds_labels != -1
        lack_label_mask = pcds_labels == -1
        pcds_voxel_with_label = pcds_voxel[with_label_mask].contiguous()
        pcds_label_with_label = pcds_labels[with_label_mask].contiguous()

        _device = pcds_voxel.device
        k_type = torch.int64
        _max_voxel = pcds_voxel.max(axis=0)[0]
        spatial_shape = [_max_voxel[0].item(), _max_voxel[1].item(), _max_voxel[2].item()]

        pcds_voxel_lack_label = pcds_voxel[lack_label_mask].contiguous()
        pcds_label_lack_label = pcds_labels[lack_label_mask].contiguous()

        table_size = pcds_voxel_with_label.shape[0] * 2

        scalar = _indice_to_scalar(pcds_voxel_with_label, spatial_shape)
        table = HashTable(_device, k_type, torch.int32, table_size)
        table.insert(scalar, pcds_label_with_label)
        scalar = _indice_to_scalar(pcds_voxel_lack_label, spatial_shape)
        values, is_empty = table.query(scalar)
        mask = is_empty==False
        pcds_label_lack_label[mask] = values[mask]
        pcds_labels[lack_label_mask] = pcds_label_lack_label

        voxel_size = voxel_size * 2
        lack_ratio = (pcds_labels==-1).sum()/pcds_labels.shape[0]
        print(voxel_size, lack_ratio)
        if(lack_ratio==0): break
    
    keeped = torch.ones_like(pcds_labels).bool()
    # not keep road
    for label in ignore_labels:
        keeped[pcds_labels==label] = False
    # to fix box crop not clean
    for label in crop_labels:
        keeped[pcds_labels==label] = False

    if visualize:
        points = pcds.cpu().numpy()
        semantic_labels = pcds_labels.cpu().numpy()
        pts_color = np.zeros((points.shape[0], 3))
        color_idx = 0
        print(np.unique(semantic_labels))
        for id in np.unique(semantic_labels):
            # if id == -1: continue
            pts_color[semantic_labels==id] = PALETTE[color_idx]
            color_idx += 1
        points_with_colors = np.concatenate([points[:, :3], pts_color], axis=-1)
        vis = Visualizer(points_with_colors, mode='xyzrgb')
        vis.show()
        del vis

    pcds_labels = pcds_labels.cpu()
    keeped = keeped.cpu()
    return pcds_labels, keeped

# clone from mmdet3d
class PointSegLearningMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int): The max possible cat_id in input segmentation mask.
            Defaults to 40.
    """

    def __init__(self, learning_map):
        self.learning_map = np.vectorize(learning_map.__getitem__)

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids. \
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']
        converted_pts_sem_mask = self.learning_map(pts_semantic_mask)

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.learning_map}, '
        return repr_str


class PointsRangeFilter(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def in_range_3d(self, tensor, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            torch.Tensor: A binary vector indicating whether each point is \
                inside the reference range.
        """
        in_range_flags = np.logical_and.reduce((
            tensor[:, 0] > point_range[0], tensor[:, 1] > point_range[1], tensor[:, 2] > point_range[2], 
            tensor[:, 0] < point_range[3], tensor[:, 1] < point_range[4], tensor[:, 2] < point_range[5]))
        return in_range_flags
    
    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        points_mask = self.in_range_3d(points, self.pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask
        #print(np.sum(points_mask) / points.tensor.size(0))

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str
    
if __name__ == '__main__':
    start = time.time()
    pcds = torch.load("/mnt/truenas/scratch/tao.jiang/data/waymo_occ/training_del/point/0000000/raw_stuff_pc.pt")
    pcds_labels = torch.load("/mnt/truenas/scratch/tao.jiang/data/waymo_occ/training_del/point/0000000/raw_stuff_label.pt")
    pcds_labels, keeped = knn_label(pcds, pcds_labels, voxel_size = 0.1, max_iter=10, ignore_labels=[0,17,18,19,20,21,22],crop_labels=[1,2,3,4,5,6,7,12,13,16], visualize=False, verbose=True)
    print(time.time() - start)
    