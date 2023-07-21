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



'''

API: generate_voxel

    src_dir : 

        cam_infos_dir/XXX/XXX_cam.pkl

        stuff_dir/vertices_XXX.pkl

        stuff_dir/seq_dict_XXX.pkl

        object/vertices_XXX.pkl

        object/objects_bank_XXX.pkl


    func: ray traversal, convert labelmap, generate lidar/camera/fov mask

    others: results in ego coordinate



'''





from typing import List, Tuple

# import open3d as o3d



import os

import sys

import numpy as np

import time

import math

import argparse

from concurrent.futures import ProcessPoolExecutor

import multiprocessing

import pickle as pkl

import matplotlib.pyplot as plt

import PIL

import torch

from glob import glob

from tqdm import tqdm

from spconv.pytorch.hash import HashTable

from collections import OrderedDict

import pccm

from cumm.inliner import NVRTCInlineBuilder

from cumm.common import TensorViewNVRTCKernel, TensorViewArrayLinalg



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from utils.config import *



INLINER = NVRTCInlineBuilder([TensorViewNVRTCKernel, TensorViewArrayLinalg])

# IMPORTANT: not use tf32 which case pose error

torch.backends.cuda.matmul.allow_tf32 = False

torch.backends.cudnn.allow_tf32 = False



# nuScenes-lidarseg dataset comes with annotations for 32 classes, convert to 16 classes for the lidar segmentation challenge, see https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval/lidarseg

# label_remap = {0: -1, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0, 12: 8, 13: 0, 14: 2, 15: 3,

#                16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0,

#                30: 16, 31: -1}

# see https://github.com/waymo-research/waymo-open-dataset/blob/bae19fa0a36664da18b691349955b95b29402713/waymo_open_dataset/protos/segmentation.proto#L20

# already remap vehicle/ped/cyc/sign in box

label_remap = {0: 0,1: 10,2: 10,3: 10,4: 10,5: 31,6: 31,7: 30,8: 81,9: 81,10: 80,

               11: 99,12: 11,13: 15,14: 50,15: 70,16: 71,17: 49,18: 40,19: 60,20: 49,21: 49,22: 48}



NOT_OBSERVED = -1

FREE = 0

OCCUPIED = 1

FREE_LABEL = 0



MAX_POINT_NUM = 10

ROAD_LABEL_START_BEFORE_REMAP = 24

ROAD_LABEL_STOP_BEFORE_REMAP = 27

ROAD_LABEL_START = 13

ROAD_LABEL_STOP = 14

BINARY_OBSERVED = 1

BINARY_NOT_OBSERVED = 0

STUFF_START = 9  # 0-10 thing 11-17 stuff

# DO NOT CHANGE

FLT_MAX = 1e9

RAY_STOP_DISTANCE_VOXEL = 1

DISTANCE_THESHOLD_IGNORE = 1.

RAY_ROAD_IGNORE_DISTANCE = 1.



VOXEL_SIZE=[0.2, 0.2, 0.2]

POINT_CLOUD_RANGE=[0, -25.6, -2, 51.2, 25.6, 4.4]

SPTIAL_SHAPE=[256, 256, 32]


VIS = False



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



def _expand_dim(array):

    return np.concatenate((array, np.ones_like(array)[:, :1]), axis=1)



def pixel2global(uvs, depth, intrinsics, sensor2ego, ego2global, normalize=True):

    nbr_points = uvs.shape[1]

    viewpad = np.eye(4)

    viewpad[:intrinsics.shape[0], :intrinsics.shape[1]] = intrinsics



    uvs_pad = np.concatenate([uvs, np.ones((1, nbr_points))])  # (2, N)=>(3, n)

    uvs_depth = uvs_pad * depth.repeat(3, 0)  # (3, n)

    uvs_depth_pad = np.concatenate([uvs_depth, np.ones((1, nbr_points))])  # (4, n)

    points = np.linalg.inv(viewpad) @ uvs_depth_pad  # (4, n)



    points = points.T @ sensor2ego.T @ ego2global.T

    points = points[:, :3]

    return points



def global2pixel(points, intrinsics, sensor2ego, ego2global, normalize=True):

    nbr_points = points.shape[0]

    viewpad = np.eye(4)

    viewpad[:intrinsics.shape[0], :intrinsics.shape[1]] = intrinsics



    points = np.concatenate([points, np.ones((nbr_points, 1))], axis=-1)  # (n, 3)=>(n, 4)

    points_sensor = np.linalg.inv(ego2global) @ np.linalg.inv(sensor2ego) @ points.T

    uvs = viewpad @ points_sensor

    uvs = uvs[:3, :]

    uvs = uvs.T



    uvs[:, 2] = np.clip(uvs[:, 2], a_min=1e-1, a_max=99999)

    uvs[:, 0] /= uvs[:, 2]

    uvs[:, 1] /= uvs[:, 2]

    

    return uvs



def global2pixel_torch(points, intrinsics, sensor2ego, ego2global, normalize=True):

    _device = points.device

    nbr_points = points.shape[0]

    viewpad = torch.eye(4).to(_device)

    viewpad[:intrinsics.shape[0], :intrinsics.shape[1]] = intrinsics

    points = torch.cat([points, torch.ones((nbr_points, 1), device=_device)], dim=-1)  # (n, 3)=>(n, 4)

    pose = torch.linalg.inv(ego2global) @ torch.linalg.inv(sensor2ego)

    points_sensor = pose.float() @ points.T

    uvs = viewpad @ points_sensor



    del points, points_sensor

    torch.cuda.empty_cache()



    uvs = uvs[:3, :]

    uvs = uvs.T



    uvs[:, 2] = torch.clip(uvs[:, 2], min=1e-1, max=99999)

    uvs[:, 0] /= uvs[:, 2]

    uvs[:, 1] /= uvs[:, 2]

    

    return uvs



def np2pcd(xyz):

    assert xyz.shape[1] == 3

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd





def pcd2np(pcd):

    return np.asarray(pcd.points)





def _init(q):

    gpu_id = q.get()

    # DO NOT "import open3d as o3d" since o3d init gpu stream

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)





def gpu_warpper(handle, jobs, gpu_num=8):

    # DO NOT import open3d before

    num_proc = min(gpu_num, len(jobs))

    ctx = multiprocessing.get_context("spawn")

    q = ctx.Queue()

    for i in range(num_proc):

        q.put(i % gpu_num)

    cnt = 0

    total = 0

    with ProcessPoolExecutor(num_proc, ctx, _init, (q,)) as ex:

        for res in ex.map(handle, jobs):

            # if res:

            #     break

            pass

            # cnt += res

            # total += 1

            # print(cnt, total)





def ray_traversal(

        points_origin, points, points_label,

        point_cloud_range, voxel_size, spatial_shape,

        free_n_theshold=10, occupied_n_theshold=1,  # TODO try different values

):

    '''

    inputs:

        points_origin: [N, 3], x/y/z order, in global coordinate

        points: [N, 3+], x/y/z order, in global coordinate

        points_label: [N, ], labels

        point_cloud_range: list[6], xmin/ymin/zmin/xmax/ymax/zmax, in global coordinate

        voxel_size: list[3], x/y/z order

        spatial_shape: list[3], x/y/z order

        free_n_theshold: traversal num exceed this as free

    outputs:

        voxel_coors: [H,W,Z,3]

        voxel_state: [H,W,Z], -1/0/1 for not observed/free/occupied

        voxel_label: [H,W,Z], from point label

        voxel_occ_count: [H,W,Z], point num in this voxel

        voxel_free_count: [H,W,Z], free traversal in this voxel

    '''

    # start_time = time.time()

    _device = points.device

    voxel_size_numpy = np.asarray(voxel_size)

    point_cloud_range_numpy = np.asarray(point_cloud_range)

    spatial_shape_numpy = np.asarray(spatial_shape)

    assert np.alltrue(

        voxel_size_numpy * spatial_shape_numpy == point_cloud_range_numpy[3:6] - point_cloud_range_numpy[:3])

    voxel_size_device = torch.tensor(voxel_size).to(_device)

    point_cloud_range_device = torch.tensor(point_cloud_range).to(_device)



    # TODO only keep ray intersect with point_cloud_range, using Liang-Barsky algorithm

    # now we only filter points not in point_cloud_range

    inrange_x = torch.logical_and(points[:, 0] > point_cloud_range[0], points[:, 0] < point_cloud_range[3])

    inrange_y = torch.logical_and(points[:, 1] > point_cloud_range[1], points[:, 1] < point_cloud_range[4])

    inrange_z = torch.logical_and(points[:, 2] > point_cloud_range[2], points[:, 2] < point_cloud_range[5])

    inrange = torch.logical_and(inrange_x, torch.logical_and(inrange_y, inrange_z))

    points_inrange = points[inrange]

    points_origin_inrange = points_origin[inrange]

    points_labels_inrange = points_label[inrange]
    


    # voxel traversal

    voxel_occ_count = torch.full(spatial_shape, fill_value=0, device=_device, dtype=torch.int)

    voxel_free_count = torch.full(spatial_shape, fill_value=0, device=_device, dtype=torch.int)

    ray_start = points_inrange.contiguous()

    ray_end = points_origin_inrange[:, :3].contiguous()

    ray_start_stride = ray_start.stride()

    ray_end_stride = ray_end.stride()

    voxel_stride = voxel_occ_count.stride()



    # debug_tensor = torch.full((1024, 3), fill_value=-1).to(_device)

    # nn = 7533802 # np.random.randint(0, ray_start.shape[0])

    # idx_tensor = torch.Tensor([nn]).to(_device)

    assert ray_start.shape == ray_end.shape

    # ray start as points, ray end as end

    assert torch.all(torch.logical_and(ray_start[:, 0] >= point_cloud_range_device[0],

                                       ray_start[:, 0] <= point_cloud_range_device[3]))

    assert torch.all(torch.logical_and(ray_start[:, 1] >= point_cloud_range_device[1],

                                       ray_start[:, 1] <= point_cloud_range_device[4]))

    assert torch.all(torch.logical_and(ray_start[:, 2] >= point_cloud_range_device[2],

                                       ray_start[:, 2] <= point_cloud_range_device[5]))

    INLINER.kernel_1d("voxel_traversal", ray_start.shape[0], 0, f"""

    auto ray_start_p = $ray_start + i*$(ray_start_stride[0]);

    auto ray_end_p = $ray_end + i*$(ray_end_stride[0]);

    // int idx = $idx_tensor[0];

    // int count = 0;

    // if(i==idx){{

    //     $debug_tensor[count*3+0] = current_voxel[0];

    //     $debug_tensor[count*3+1] = current_voxel[1];

    //     $debug_tensor[count*3+2] = current_voxel[2];

    //     count ++;

    // }}

    // Bring the ray_start_p and ray_end_p in voxel coordinates

    float new_ray_start[3];

    float new_ray_end[3];

    float voxel_size_[3];

    new_ray_start[0] = ray_start_p[0] - $(point_cloud_range[0]);

    new_ray_start[1] = ray_start_p[1] - $(point_cloud_range[1]);

    new_ray_start[2] = ray_start_p[2] - $(point_cloud_range[2]);

    new_ray_end[0] = ray_end_p[0] - $(point_cloud_range[0]);

    new_ray_end[1] = ray_end_p[1] - $(point_cloud_range[1]);

    new_ray_end[2] = ray_end_p[2] - $(point_cloud_range[2]);

    voxel_size_[0] = $(voxel_size[0]);

    voxel_size_[1] = $(voxel_size[1]);

    voxel_size_[2] = $(voxel_size[2]);



    // Declare some variables that we will need

    float ray[3]; // keeep the ray

    int step[3];

    float tDelta[3];

    int current_voxel[3];

    int last_voxel[3];

    int target_voxel[3];

    float _EPS = 1e-9;

    for(int k=0; k<3; k++) {{

        // Compute the ray

        ray[k] = new_ray_end[k] - new_ray_start[k];



        // Get the step along each axis based on whether we want to move

        // left or right

        step[k] = (ray[k] >=0) ? 1:-1;



        // Compute how much we need to move in t for the ray to move bin_size

        // in the world coordinates

        tDelta[k] = (ray[k] !=0) ? (step[k] * voxel_size_[k]) / ray[k]: {FLT_MAX};



        // Move the start and end points just a bit so that they are never

        // on the boundary

        new_ray_start[k] = new_ray_start[k] + step[k]*voxel_size_[k]*_EPS;

        new_ray_end[k] = new_ray_end[k] - step[k]*voxel_size_[k]*_EPS;



        // Compute the first and the last voxels for the voxel traversal

        current_voxel[k] = (int) floor(new_ray_start[k] / voxel_size_[k]);

        last_voxel[k] = (int) floor(new_ray_end[k] / voxel_size_[k]);

        target_voxel[k] = (int) floor(new_ray_start[k] / voxel_size_[k]); // ray start as point, ray end as origin

    }}



    // Make sure that the starting voxel is inside the voxel grid

    // if (

    //     ((current_voxel[0] >= 0 && current_voxel[0] < $grid_x) &&

    //     (current_voxel[1] >= 0 && current_voxel[1] < $grid_y) &&

    //     (current_voxel[2] >= 0 && current_voxel[2] < $grid_z)) == 0

    // ) {{

    //     return;

    // }}



    // Compute the values of t (u + t*v) where the ray crosses the next

    // boundaries

    float tMax[3];

    float current_coordinate;

    for (int k=0; k<3; k++) {{

        if (ray[k] !=0 ) {{

            // tMax contains the next voxels boundary in every axis

            current_coordinate = current_voxel[k]*voxel_size_[k];

            if (step[k] < 0 && current_coordinate < new_ray_start[k]) {{

                tMax[k] = current_coordinate;

            }}

            else {{

                tMax[k] = current_coordinate + step[k]*voxel_size_[k];

            }}

            // Now it contains the boundaries in t units

            tMax[k] = (tMax[k] - new_ray_start[k]) / ray[k];

        }}

        else {{

            tMax[k] = {FLT_MAX};

        }}

    }}



    // record point, +1

    if (

        ((target_voxel[0] >= 0 && target_voxel[0] < $(spatial_shape[0])) &&

        (target_voxel[1] >= 0 && target_voxel[1] < $(spatial_shape[1])) &&

        (target_voxel[2] >= 0 && target_voxel[2] < $(spatial_shape[2])))

    ) {{

        auto targetIdx = target_voxel[0] * $(voxel_stride[0]) + target_voxel[1] * $(voxel_stride[1]) + target_voxel[2] * $(voxel_stride[2]);

        auto old = atomicAdd($voxel_occ_count + targetIdx, 1);

    }}



    // Start the traversal

    // while (voxel_equal(current_voxel, last_voxel) == 0 && ii < $max_voxels) {{

    // while((current_voxel[0] == last_voxel[0] && current_voxel[1] == last_voxel[1] && current_voxel[2] == last_voxel[2])==0){{

    while(step[0]*(current_voxel[0] - last_voxel[0]) < {RAY_STOP_DISTANCE_VOXEL} && step[1]*(current_voxel[1] - last_voxel[1]) < {RAY_STOP_DISTANCE_VOXEL} && step[2]*(current_voxel[2] - last_voxel[2]) < {RAY_STOP_DISTANCE_VOXEL}){{ // due to traversal bias, ray may not exactly hit end voxel which cause traversal not stop

        // if tMaxX < tMaxY

        if (tMax[0] < tMax[1]) {{

            if (tMax[0] < tMax[2]) {{

                // We move on the X axis

                current_voxel[0] = current_voxel[0] + step[0];

                if (current_voxel[0] < 0 || current_voxel[0] >= $(spatial_shape[0]))

                    break;

                tMax[0] = tMax[0] + tDelta[0];

            }}

            else {{

                // We move on the Z axis

                current_voxel[2] = current_voxel[2] + step[2];

                if (current_voxel[2] < 0 || current_voxel[2] >= $(spatial_shape[2]))

                    break;

                tMax[2] = tMax[2] + tDelta[2];

            }}

        }}

        else {{

            // if tMaxY < tMaxZ

            if (tMax[1] < tMax[2]) {{

                // We move of the Y axis

                current_voxel[1] = current_voxel[1] + step[1];

                if (current_voxel[1] < 0 || current_voxel[1] >= $(spatial_shape[1]))

                    break;

                tMax[1] = tMax[1] + tDelta[1];

            }}

            else {{

                // We move on the Z axis

                current_voxel[2] = current_voxel[2] + step[2];

                if (current_voxel[2] < 0 || current_voxel[2] >= $(spatial_shape[2]))

                    break;

                tMax[2] = tMax[2] + tDelta[2];

            }}

        }}



        // set the traversed voxels

        auto currentIdx = current_voxel[0] * $(voxel_stride[0]) + current_voxel[1] * $(voxel_stride[1]) + current_voxel[2] * $(voxel_stride[2]);

        auto distance2start = abs(current_voxel[0] - target_voxel[0]) * voxel_size_[0] + abs(current_voxel[1] - target_voxel[1]) * voxel_size_[1] + abs(current_voxel[2] - target_voxel[2]) * voxel_size_[2];

        auto distance2end = abs(current_voxel[0] - last_voxel[0]) * voxel_size_[0] + abs(current_voxel[1] - last_voxel[1]) * voxel_size_[1] + abs(current_voxel[2] - last_voxel[2]) * voxel_size_[2];

        auto distance2start_height = abs(current_voxel[2] - target_voxel[2]) * voxel_size_[2];

        if(distance2start>{DISTANCE_THESHOLD_IGNORE} && distance2end>{DISTANCE_THESHOLD_IGNORE}){{

            if($points_labels_inrange[i]>={ROAD_LABEL_START} && $points_labels_inrange[i]<={ROAD_LABEL_STOP}){{

                if(distance2start_height < {RAY_ROAD_IGNORE_DISTANCE}){{

                    continue;

                }}

            }}

            auto old = atomicAdd($voxel_free_count + currentIdx, 1);

        }}

    }}

    """, verbose_path="build/nvrtc/voxel_traversal")

    torch.cuda.synchronize()



    # set default not observed

    voxel_state = torch.full(spatial_shape, fill_value=NOT_OBSERVED, device=_device, dtype=torch.long)

    voxel_label = torch.full(spatial_shape, fill_value=FREE_LABEL, device=_device,

                             dtype=torch.long)  # default semantic free

    voxel_label_squeeze = voxel_label.reshape(-1)

    # set voxel label

    pcds_voxel = torch.div(points_inrange - point_cloud_range_device[:3], voxel_size_device,

                           rounding_mode='floor').long()  # x/y/z order

    inds = _indice_to_scalar(pcds_voxel, voxel_state.shape)

    voxel_label_squeeze[inds] = points_labels_inrange

    # set free voxel

    voxel_state[voxel_free_count > free_n_theshold] = FREE

    # set occupied

    voxel_state[voxel_occ_count > occupied_n_theshold] = OCCUPIED



    xx = torch.arange(0, voxel_state.shape[0]).to(_device)

    yy = torch.arange(0, voxel_state.shape[1]).to(_device)

    zz = torch.arange(0, voxel_state.shape[2]).to(_device)

    grid_x, grid_y, grid_z = torch.meshgrid(xx, yy, zz, indexing='ij')

    voxel_coors = torch.stack([grid_x, grid_y, grid_z], axis=-1)



    # # vis ray

    # debug_tensor = debug_tensor[debug_tensor[:,0]!=-1]

    # debug_tensor = debug_tensor.long()

    # inds = _indice_to_scalar(debug_tensor[2:], voxel_state.shape)

    # voxel_label_vis = voxel_label.clone()

    # voxel_show = voxel_state==OCCUPIED

    # voxel_show_squeeze = voxel_show.reshape(-1)

    # voxel_show_squeeze[inds] = True

    # voxel_label_vis_squeeze = voxel_label_vis.reshape(-1)

    # voxel_label_vis_squeeze[inds] = 2

    # print(idx_tensor.item(), debug_tensor[0], debug_tensor[1], )

    # print(debug_tensor)

    # vis = vis_occ.main(voxel_label_vis.cpu(), voxel_show.cpu(), voxel_size=[1,1,1])

    # vis.run()

    # del vis

    # print('init voxel state:', time.time() - start_time)

    return voxel_coors, voxel_state, voxel_label, voxel_occ_count, voxel_free_count



def generate_voxel(file_idx, seq_dict, pose_file='output/010_cam.pkl', out_dir=""):

    '''

    file: dict,

        pcd: [N,4], 4: x/y/z/frameid , x/y/z in global coordinate

        pcd_labels: [N,]

        origin: [N,3], 3: x/y/z, x/y/z in global coordinate

    pose_file: dict, key: camera id

        data['ego2global'], data['sensor2ego'], data['intrinsics'], data['img']

    '''

    _device = torch.device('cuda', 0)

    occlusion_theshold = 1

    if not os.path.exists(out_dir): os.makedirs(out_dir)

    SINGLE_FRAME = False

    observe_frame_id = 0

    # file_idx = os.path.basename(file)

    # file_idx = file_idx.split('.')[0]

    # icp_fun(file, out_file)

    # with open(file, 'rb') as f:

    #     seq_dict = pkl.load(f)

    points = torch.Tensor(seq_dict['pcd'][:, :3])

    frameidx = torch.Tensor(seq_dict['pcd'][:, -1]).long()

    points_label = torch.Tensor(seq_dict['pcd_labels']).long()

    keys = [k for k in seq_dict['origin']]

    origin_list = [None for k in keys]

    for k in keys:

        origin_list[k] = torch.Tensor(seq_dict['origin'][k])

    origins = torch.cat(origin_list, dim=0)

    points_origin = origins[frameidx]

    if SINGLE_FRAME:

        keep = frameidx == observe_frame_id

        points = points[keep]

        points_label = points_label[keep]

        points_origin = points_origin[keep]

   



    points_dev = points.to(_device)

    points_label_dev = points_label.to(_device)

    points_origin_dev = points_origin.to(_device)





    # convert 32 class to 16 class

    points_label_remap_dev = torch.full_like(points_label_dev, fill_value=-1)

    for src, tgt in label_remap.items():

        mask = points_label_dev == src

        points_label_remap_dev[mask] = tgt

    assert (points_label_remap_dev == -1).sum() == 0

    points_label_dev = points_label_remap_dev





    with open(pose_file, 'rb') as f:

        camera_dict = pkl.load(f)

    # lidar2ego = camera_dict['lidar2ego']

    ego2global = camera_dict['ego2global']



    # global to ego

    points2ego_dev = torch.cat([points_dev, torch.ones((points_dev.shape[0], 1), device=_device)],

                               dim=-1) @ torch.Tensor(np.linalg.inv(ego2global).T).to(_device)

    points2ego_dev = points2ego_dev[:, :3]

    points_origin_dev = torch.cat([points_origin_dev, torch.ones((points_origin_dev.shape[0], 1), device=_device)],

                                  dim=-1) @ torch.Tensor(np.linalg.inv(ego2global).T).to(_device)

    points_origin_dev = points_origin_dev[:, :3]





    # construct local voxel state

    voxel_coors, voxel_state, voxel_label, voxel_occ_count, voxel_free_count = ray_traversal(

        points_origin_dev, points2ego_dev, points_label_dev,

        POINT_CLOUD_RANGE, VOXEL_SIZE, SPTIAL_SHAPE,

    )



    # Save in Semantic Kitti format

    # final_input_voxel = (input_voxel_state == 1).cpu().numpy().astype(np.uint8)


    final_input_voxel = (voxel_state == 1).cpu().numpy().astype(np.uint8)
    kitti_voxel_label = voxel_label.cpu().numpy().astype(np.uint16)

    kitti_invalid = torch.logical_and(voxel_state == -1, voxel_label == 0).cpu().numpy().astype(np.uint8)

    #kitti_voxel_label.tofile(os.path.join(out_dir,  f"{str(file_idx).zfill(6)}.label"))

    #kitti_invalid.tofile(os.path.join(out_dir,  f"{str(file_idx).zfill(6)}.invalid"))

    

    np.savez_compressed(os.path.join(out_dir,  f"{str(file_idx).zfill(6)}.npz"),input=final_input_voxel,labels=kitti_voxel_label,invalid=kitti_invalid)



    print('saved: ', os.path.join(out_dir,  f"{str(file_idx).zfill(6)}"))











def generate_object_id_mapping(object_infos):

    mapping = {}

    cur_id = 0

    for obj_id, _ in object_infos.items():

        mapping[obj_id] = cur_id

        cur_id += 1

    return mapping



def print_run_time(generate_voxel):

    def wrapper(*args, **kw):

        local_time = time.time()

        generate_voxel(*args, **kw)

        print('current Function [%s] run time: %.2f s.' % (generate_voxel.__name__ ,time.time() - local_time))

    return wrapper



def generate_frame_object_mapping(frame_num, object_infos):

    mapping = OrderedDict()

    # for i in range(frame_num): mapping[i] = []

    for obj_id, obj_info in object_infos.items():

        frame_ids = [dict['frameid'] for dict in obj_info]

        for frame in frame_ids:

            if not frame in mapping.keys():

                mapping[frame] = []

            mapping[frame].append(obj_id)

    return mapping



def get_object_frame_info(object_infos, obj_id, frame_id):

    for dict in object_infos[obj_id]:

        if dict['frameid'] == frame_id: return dict

    return None



class SceneDataset:

    def __init__(self, seq_idx):

        # Initialize your dataset here ..

        scene_num = str(seq_idx).zfill(3)

        # load stuff vertices and infos

        with open(os.path.join(stuff_dir, f'vertices_{scene_num}.pkl'), 'rb') as f:

            self.stuff_vertices = pkl.load(f)

        with open(os.path.join(stuff_dir, f'seq_dict_{scene_num}.pkl'), 'rb') as f:

            self.stuff_infos = pkl.load(f)

        # load object vertices and infos

        with open(os.path.join(object_dir, f'objects_bank_{scene_num}.pkl'), 'rb') as f:

            self.object_infos = pkl.load(f)

        with open(os.path.join(object_dir, f'vertices_{scene_num}.pkl'), 'rb') as f:

            self.object_vertices = pkl.load(f)

        frame_num: int = len(self.stuff_infos)

        self.frame_object_mapping = generate_frame_object_mapping(frame_num, self.object_infos)

        self.stuff_vertices_global = self.stuff_vertices['vertices']

        self.stuff_labels = self.stuff_vertices['labels']

        self.origins = {}

        for frame_idx in tqdm(list(self.stuff_infos.keys())):

            self.origins[frame_idx] = self.stuff_infos[frame_idx]['origin']

        self.seq_idx = seq_idx



    def __len__(self):

        return len(self.origins)



    def __getitem__(self, idx: int):

        start_time = time.time()

        vehicle2global = self.stuff_infos[idx]['vehicle2global']

        # lidar2vehicle = stuff_infos[frame_idx]['lidar2vehicle']

        icp_transformation = self.stuff_infos[idx]['icp_transformation']

        frame_ = self.stuff_vertices_global[:, 3:4]

        pcd = (_expand_dim(self.stuff_vertices_global[:, :3]) @ np.linalg.inv(icp_transformation).T)[:, :3]

        pcd = np.concatenate((pcd, frame_), axis=1)

        pcd_labels = self.stuff_labels





        # stuff_vertices_frame = (_expand_dim(stuff_vertices_global[:, :3]) @ np.linalg.inv(icp_transformation).T @ np.linalg.inv(vehicle2global).T)[:, :3] # ICP_inv

        for obj_id, agg_obj in self.object_vertices.items():

            if idx not in self.frame_object_mapping:

                print('Warning: no instance in frame, seq: {} frame: {}'.format(self.seq_idx, idx))

                continue

            if obj_id not in self.frame_object_mapping[idx]: continue

            object_vertices_box = agg_obj['vertices']

            if object_vertices_box.shape[0] == 0: continue

            object_frame_info = get_object_frame_info(self.object_infos, obj_id, idx)

            if object_frame_info == None: continue

            # seg_labels = agg_obj['labels']

            ins_label = object_frame_info['label']

            # lidar2vehicle = object_frame_info['lidar2vehicle']

            vehicle2box = object_frame_info['vehicle2box']

            icp_transformation = object_frame_info['icp_transformation']



            # object_vertices_frame = (_expand_dim(object_vertices_box) @ np.linalg.inv(vehicle2box.T @ icp_transformation.T))[:, :3]

            object_vertices_frame = (_expand_dim(object_vertices_box) @ np.linalg.inv(

                vehicle2box.T @ icp_transformation.T) @ vehicle2global.T)[:, :3]

            object_vertices_frame = np.concatenate(

                (object_vertices_frame, np.full_like(object_vertices_frame[:, -2:-1], idx)), axis=1)

            pcd = np.concatenate((pcd, object_vertices_frame))

            object_labels = np.full_like(object_vertices_frame[:, 0], ins_label)

            object_labels = np.where(object_labels == 2, 7, object_labels)
            object_labels = np.where(object_labels == 3, 8, object_labels)
            object_labels = np.where(object_labels == 4, 6, object_labels)

            pcd_labels = np.concatenate((pcd_labels, object_labels))

        # label_mask = pcd_labels != 0

        # pcd = pcd[label_mask, :]

        # pcd_labels = pcd_labels[label_mask]

        end_time = time.time()

        #print(f'{idx}: {end_time - start_time}')

        return {

            'pcd': pcd,

            'pcd_labels': pcd_labels,

            'origin': self.origins,

            'vehicle2global': vehicle2global,

        }



def main_fun(seq_idx):

    seq_name = f'{str(seq_idx+798).zfill(3)}'

    cam_files = glob(os.path.join(cam_infos_dir, seq_name, "*cam.pkl"))

    cam_files.sort()

    done = True

    out_dir = os.path.join(voxel_dir,"dataset",'sequences',seq_name,'voxels')

    for idx, cam_file in enumerate(cam_files):

        file_idx = os.path.basename(cam_file)

        file_idx = file_idx.split('.')[0].split('_')[0]

        name = f'{str(file_idx).zfill(6)}'

        file = os.path.join(out_dir, '{}.npz'.format(name))

        if not os.path.exists(file): done = False    




    if UPDATE or not done:

        # init dataset

        try: 

            dataset = SceneDataset(seq_idx+798)

        except Exception as e:

            print('Error {} {}'.format(seq_idx, e))

        for idx, cam_file in enumerate(cam_files):

            print("process {} ...".format(cam_file))

            file_idx = os.path.basename(cam_file)

            file_idx = file_idx.split('.')[0].split('_')[0]
            
            if int(file_idx) % 5 != 0:
                continue

            seq_dict = dataset.__getitem__(int(file_idx))

            # points = torch.Tensor(seq_dict['pcd'][:, :3])

            # frameidx = torch.Tensor(seq_dict['pcd'][:, -1]).long()

            # points_label = torch.Tensor(seq_dict['pcd_labels']).long()

            # seq_dict['origin']

            # seq_dict['vehicle2global']

            generate_voxel(file_idx, seq_dict, cam_file, out_dir)

            torch.cuda.empty_cache()

    else:

        print('skip {}'.format(seq_idx+798))





def main():

    parser = argparse.ArgumentParser(description='',

                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--start', type=int, default=0)

    parser.add_argument('--end', type=int, default=10)

    parser.add_argument('--gpu', type=int, default=4)

    args = parser.parse_args()



    jobs = [i for i in range(args.start, args.end)]

    gpu_warpper(main_fun, jobs, gpu_num=args.gpu)



def debug():

    main_fun(0)



if __name__ == "__main__":

    main()


