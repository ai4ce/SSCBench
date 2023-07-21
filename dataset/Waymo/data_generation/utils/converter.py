import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from multiprocessing import Pool
from os.path import join, isdir
import argparse
from glob import glob
import os.path as osp
import mmcv
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils

class WaymoToKITTI(object):

    def __init__(self, load_dir, save_dir, num_proc,split,start,end):
        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.num_proc = int(num_proc)

        self.split = split 
        self.start = start
        self.end = end

        self.tfrecord_pathnames = self.split_tfrecord()

        self.tfrecord_pathnames = self.tfrecord_pathnames[start:end]

        self.image_save_dir       = self.save_dir + '/image_2'
        self.calib_save_dir       = self.save_dir + '/calib'
        # self.pose_save_dir = self.save_dir + '/pose'


    def convert(self):
        print("start converting ...")
        with Pool(self.num_proc) as p:
            r = list(tqdm.tqdm(p.imap(self.convert_one, range(len(self))), total=len(self)))
        print("\nfinished ...")

    def convert_one(self, file_idx):
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')
        file_idx = file_idx + self.start
        for frame_idx, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if frame_idx ==0:
                # parse calibration files
                self.save_calib(frame, file_idx)
            if frame_idx %10 ==0:
                # save images
                self.save_image(frame, file_idx, frame_idx)
                #self.save_pose(frame, file_idx, frame_idx)

    def __len__(self):
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx):
        """ parse and save the images in png format
                :param frame: open dataset frame proto
                :param file_idx: the current file number
                :param frame_idx: the current frame number
                :return:
        """
        for img in frame.images:
            if img.name ==1:
                if self.split == 'training':
                    out_dir = os.path.join(self.image_save_dir, str(file_idx).zfill(3))
                elif self.split =='validation':
                    out_dir = os.path.join(self.image_save_dir, str(file_idx+798).zfill(3))
                if not os.path.exists(out_dir): os.makedirs(out_dir)
                img_path = os.path.join(out_dir,str(frame_idx).zfill(3) + '.png')

                img = cv2.imdecode(np.frombuffer(img.image, np.uint8), cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                resized = cv2.resize(rgb_img, (960, 640), interpolation=cv2.INTER_LINEAR)
                plt.imsave(img_path, resized, format='png')
    def save_pose(self, frame, file_idx, frame_idx):
        """ Save self driving car (SDC)'s own pose

        Note that SDC's own pose is not included in the regular training of KITTI dataset
        KITTI raw dataset contains ego motion files but are not often used
        Pose is important for algorithms that takes advantage of the temporal information

        """

        pose = np.array(frame.pose.transform).reshape(4,4)
        if self.split == 'training':
            out_dir = os.path.join(self.pose_save_dir, str(file_idx).zfill(3))
        elif self.split =='validation':
            out_dir = os.path.join(self.pose_save_dir, str(file_idx+798).zfill(3))
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        np.savetxt(join(out_dir , str(frame_idx).zfill(3) + '.txt'), pose)

    def save_calib(self, frame, file_idx):
        """ parse and save the calibration data
                :param frame: open dataset frame proto
                :param file_idx: the current file number
                :param frame_idx: the current frame number
                :return:
        """
        # kitti:
        #   bbox in reference camera frame (right-down-front)
        #       image_x_coord = Px * R0_rect * R0_rot * bbox_coord
        #   lidar points in lidar frame (front-right-up)
        #       image_x_coord = Px * R0_rect * Tr_velo_to_cam * lidar_coord
        #   note:   R0_rot is caused by bbox rotation
        #           Tr_velo_to_cam projects lidar points to cam_0 frame
        # waymo:
        #   bbox in vehicle frame, hence, use a virtual reference frame
        #   since waymo camera uses frame front-left-up, the virtual reference frame (right-down-front) is
        #   built on a transformed front camera frame, name this transform T_front_cam_to_ref
        #   and there is no rectified camera frame
        #       image_x_coord = intrinsics_x * Tr_front_cam_to_cam_x * inv(T_front_cam_to_ref) * R0_rot * bbox_coord(now in ref frame)
        #   lidar points in vehicle frame
        #       image_x_coord = intrinsics_x * Tr_front_cam_to_cam_x * inv(T_front_cam_to_ref) * T_front_cam_to_ref * Tr_velo_to_front_cam * lidar_coord
        # hence, waymo -> kitti:
        #   set Tr_velo_to_cam = T_front_cam_to_ref * Tr_vehicle_to_front_cam = T_front_cam_to_ref * inv(Tr_front_cam_to_vehicle)
        #       as vehicle and lidar use the same frame after fusion
        #   set R0_rect = identity
        #   set P2 = front_cam_intrinsics * Tr_waymo_to_conv * Tr_front_cam_to_front_cam * inv(T_front_cam_to_ref)
        #   note: front cam is cam_0 in kitti, whereas has name = 1 in waymo
        #   note: waymo camera has a front-left-up frame,
        #       instead of the conventional right-down-front frame
        #       Tr_waymo_to_conv is used to offset this difference. However, Tr_waymo_to_conv is the same as
        #       T_front_cam_to_ref, hence,
        #   set P2 = front_cam_intrinsics


        calib_context = ''

        # front-left-up -> right-down-front
        # T_front_cam_to_ref = np.array([
        #     [0.0, -1.0, 0.0],
        #     [-1.0, 0.0, 0.0],
        #     [0.0, 0.0, 1.0]
        # ])
        T_front_cam_to_ref = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0]
        ])
        # T_ref_to_front_cam = np.array([
        #     [0.0, 0.0, 1.0],
        #     [-1.0, 0.0, 0.0],
        #     [0.0, -1.0, 0.0]
        # ])

        # print('context\n',frame.context)

        for camera in frame.context.camera_calibrations:
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                T_front_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(4, 4)
                # print('T_front_cam_to_vehicle\n', T_front_cam_to_vehicle)
                T_vehicle_to_front_cam = np.linalg.inv(T_front_cam_to_vehicle)

                front_cam_intrinsic = np.zeros((3, 4))
                front_cam_intrinsic[0, 0] = camera.intrinsic[0]
                front_cam_intrinsic[1, 1] = camera.intrinsic[1]
                front_cam_intrinsic[0, 2] = camera.intrinsic[2]
                front_cam_intrinsic[1, 2] = camera.intrinsic[3]
                front_cam_intrinsic[2, 2] = 1

                break

        # print('front_cam_intrinsic\n', front_cam_intrinsic)

        self.T_front_cam_to_ref = T_front_cam_to_ref.copy()
        self.T_vehicle_to_front_cam = T_vehicle_to_front_cam.copy()

        P2 = front_cam_intrinsic.reshape(12)
        calib_context += "P2: " + " ".join(['{}'.format(i) for i in P2]) + '\n'

        Tr_velo_to_cam = self.cart_to_homo(T_front_cam_to_ref) @ np.linalg.inv(T_front_cam_to_vehicle)
        # print('T_front_cam_to_vehicle\n', T_front_cam_to_vehicle)
        # print('np.linalg.inv(T_front_cam_to_vehicle)\n', np.linalg.inv(T_front_cam_to_vehicle))
        # print('cart_to_homo(T_front_cam_to_ref)\n', cart_to_homo(T_front_cam_to_ref))
        # print('Tr_velo_to_cam\n',Tr_velo_to_cam)
        calib_context += "Tr" + ": " + " ".join(['{}'.format(i) for i in Tr_velo_to_cam[:3, :].reshape(12)]) + '\n'
        if self.split =='training':
            out_dir = os.path.join(self.calib_save_dir,str(file_idx).zfill(3))
        elif self.split =='validation':
            out_dir = os.path.join(self.calib_save_dir,str(file_idx+798).zfill(3))
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        with open(os.path.join(out_dir,'calib.txt'), 'w+') as fp_calib:
            fp_calib.write(calib_context)


    def cart_to_homo(self, mat):
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret
    
    def split_tfrecord(self):

        if self.split == 'training':
            return sorted(glob(osp.join(self.load_dir, 'training', '*.tfrecord')))
        else:
            return sorted(glob(osp.join(self.load_dir, 'validation', '*.tfrecord')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', default='...../Waymo/waymo_format/',help='Directory to load Waymo Open Dataset tfrecords')
    parser.add_argument('--save_dir', default='...../Waymo/training/convert',help='Directory to save converted KITTI-format data')
    parser.add_argument('--num_proc', default=1, help='Number of processes to spawn')
    parser.add_argument('--split', default='training', type=str)

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)

    args = parser.parse_args()

    converter = WaymoToKITTI(args.load_dir, args.save_dir, args.num_proc,args.split,args.start,args.end)
    converter.convert()
