import os
import sys
import math
import numpy as np
import itertools
import pickle
from collections import OrderedDict
from tqdm import tqdm
from glob import glob
import os.path as osp
# tf.enable_eager_execution()
import time
import multiprocessing
multiprocessing.set_start_method('forkserver', force=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # wtf tensorflow
import tensorflow as tf
import argparse
import pathlib
from PIL import Image
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import mmcv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from utils.config import *

def get_image_transform(camera_calibration):
    """ For a given camera calibration, compute the transformation matrix
        from the vehicle reference frame to the image space.
    """
    # TODO: Handle the camera distortions
    extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4,4)
    intrinsic = camera_calibration.intrinsic

    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0  0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0],
        [0, 0,                       0, 1]])

    # Swap the axes around
    axes_transformation = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]])

    # Compute the projection matrix from the vehicle space to image space.
    vehicle_to_image = np.matmul(camera_model, np.matmul(axes_transformation, np.linalg.inv(extrinsic)))
    return extrinsic, np.matmul(camera_model, axes_transformation)
    #return vehicle_to_image

def parse_data(FILENAME = os.path.join(tfrecord_dir, 'segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'), tfrecord_idx=0,split='training'):
    # if osp.exists(osp.join(output_dir, f'seq_dict_scratch_{tfrecord_idx}.pkl')): return
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    seq_dict = OrderedDict()
    param_mapping = {
        0: 0, 1: 1, 2: 3, 3: 2, 4: 4,
    }
    for frame_idx, data in tqdm(enumerate(dataset)):
        if frame_idx % 5 != 0:
            continue
        cam_infos = {}
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        lidar2vehicle = np.array(frame.context.laser_calibrations[0].extrinsic.transform).reshape(4, 4)
        vehicle2global = np.array(frame.pose.transform).reshape(4, 4)
        cam_infos['lidar2ego'] = lidar2vehicle
        cam_infos['ego2global'] = vehicle2global
        for img_idx in range(5):
            img = frame.images[img_idx]
            img = mmcv.imfrombytes(img.image)
            img = Image.fromarray(img)
            sensor2vehicle, intrinsics = get_image_transform(frame.context.camera_calibrations[param_mapping[img_idx]])
            cam_infos[img_idx] = {
                'ego2global': vehicle2global,
                'sensor2ego': sensor2vehicle,
                'intrinsics': intrinsics,
                'img'       : img,
            }

        if split =='training':
            out_dir = os.path.join(cam_infos_dir, str(tfrecord_idx).zfill(3))
        elif  split =='validation':
            out_dir = os.path.join(cam_infos_dir, str(tfrecord_idx+798).zfill(3))
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        with open(os.path.join(out_dir, f'{str(frame_idx).zfill(3)}_cam.pkl'),'wb') as f:
            pickle.dump(cam_infos, f)

        # if frame_idx == 30: break

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
    for idx in samples_set:
        if split =='training':
            out_dir = os.path.join(cam_infos_dir, str(idx).zfill(3))
        elif  split =='validation':
            out_dir = os.path.join(cam_infos_dir, str(idx+798).zfill(3))
        if os.path.exists(out_dir):
            print(f"skip {split} {idx}")
            continue
        tfrecord = train_split[idx]
        parse_data(tfrecord, idx,split)

def main():
    parser = argparse.ArgumentParser(description='visualization settings.')
    parser.add_argument('--split', default='training', type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=798, type=int)
    args = parser.parse_args()

    train_split = split_tfrecord(tfrecord_dir, args.split)
    cpu_num = CPU_NUM
    samples_split = np.array_split(np.arange(args.start, args.end), cpu_num)
    print("Number of cores: {}, sequences per core: {}".format(cpu_num, len(samples_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    pbar = tqdm(total=args.end - args.start)

    for proc_id, samples_set in enumerate(samples_split):
        p = workers.apply_async(handle_scene_single_core,
                                args=(samples_set, train_split,args.split))
        processes.append(p)
    workers.close()
    workers.join()

def debug():
    parser = argparse.ArgumentParser(description='visualization settings.')
    parser.add_argument('--split', default='training', type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=798, type=int)
    args = parser.parse_args()
    train_split = split_tfrecord(tfrecord_dir, args.split)
    handle_scene_single_core([0], train_split,args.split)

if __name__ == '__main__':
    main()
