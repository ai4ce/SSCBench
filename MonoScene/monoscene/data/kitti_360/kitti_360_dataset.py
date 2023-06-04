import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
from torchvision import transforms
from monoscene.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)

class Kitti360Dataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        project_scale=2,
        frustum_size=4,
        color_jitter=None,
        fliplr=0.0,
    ):
        super().__init__()
        self.root = root
        self.label_root = os.path.join(preprocess_root, "labels")
        # splits = {
        #     "train": ["2013_05_28_drive_0003_sync"],
        #     "val": ["2013_05_28_drive_0003_sync"],
        #     "test": ["2013_05_28_drive_0003_sync"],
        # }
        splits = {
            "train": ["2013_05_28_drive_0004_sync", "2013_05_28_drive_0000_sync", "2013_05_28_drive_0010_sync","2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0007_sync"],
            "val": ["2013_05_28_drive_0006_sync"],
            "test": ["2013_05_28_drive_0009_sync"],
        }
        self.split = split
        self.sequences = splits[split]
        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        self.voxel_size = 0.2
        self.vox_origin = np.array([0, -25.6, -2])
        self.scene_size = (51.2, 51.2, 6.4)
        self.img_H = 376
        self.img_W = 1408

        self.fliplr = fliplr

        # self.V2C = self.get_velo2cam()

        # self.V2C = self.V2C[:3, :]

        # self.P = self.get_cam_k()

        # self.T_velo_2_cam = np.identity(4)  # 4x4 matrix
        # self.T_velo_2_cam[:3, :4] = self.V2C
    
        # self.proj_matrix = self.P @ self.T_velo_2_cam
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.scans = []
        for sequence in self.sequences:
            # glob_path = os.path.join(
            #     self.root, "data_2d_raw", sequence, "image_00/data_rect", "*.png"
            # )
            calib = self.read_calib()
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            v_path = os.path.join(
                self.root, "data_2d_raw", sequence, "voxels", "*.bin"
            )
            # for img_path in glob.glob(glob_path):
            #     self.scans.append({"img_path": img_path, "sequence": sequence})
            for voxel_path in glob.glob(v_path):
                self.scans.append(
                    {
                        "sequence": sequence,
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": voxel_path,
                    }
                )

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


    # def get_cam_k(self):
    #     cam_k = np.array(
    #         [
    #             552.554261,
    #             0.000000,
    #             682.049453,
    #             0.000000,
    #             0.000000,
    #             552.554261,
    #             238.769549,
    #             0.000000,
    #             0.000000,
    #             0.000000,
    #             1.000000,
    #             0.000000,
    #         ]
    #     ).reshape(3, 4)
    #     return cam_k

    # def get_velo2cam(self):
    #     cam2velo = np.array(
    #         [
    #             0.04307104361,
    #             -0.08829286498,
    #             0.995162929,
    #             0.8043914418,
    #             -0.999004371,
    #             0.007784614041,
    #             0.04392796942,
    #             0.2993489574,
    #             -0.01162548558,
    #             -0.9960641394,
    #             -0.08786966659,
    #             -0.1770225824,
    #         ]
    #     ).reshape(3, 4)
    #     cam2velo = np.concatenate(
    #         [cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0
    #     )
    #     return np.linalg.inv(cam2velo)

    def __getitem__(self, index):
        scan = self.scans[index]
        voxel_path = scan["voxel_path"]
        sequence = scan["sequence"]
        P = scan["P"]
        T_velo_2_cam = scan["T_velo_2_cam"]
        proj_matrix = scan["proj_matrix"]

        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        rgb_path = os.path.join(
            self.root,  "data_2d_raw", sequence, "image_00/data_rect", frame_id + ".png"
        )


        data = {
            "frame_id": frame_id,
            "sequence": sequence,
            "P": P,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix": proj_matrix,
        }


        scale_3ds = [self.output_scale, self.project_scale]
        data["scale_3ds"] = scale_3ds
        cam_k = P[0:3, 0:3]
        data["cam_k"] = cam_k

        for scale_3d in scale_3ds:
            projected_pix, fov_mask, pix_z = vox2pix(
                T_velo_2_cam,
                cam_k,
                self.vox_origin,
                self.voxel_size * scale_3d,
                self.img_W,
                self.img_H,
                self.scene_size,
            )
            data["projected_pix_{}".format(scale_3d)] = projected_pix
            data["fov_mask_{}".format(scale_3d)] = fov_mask
            data["pix_z_{}".format(scale_3d)] = pix_z



        target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
        target = np.load(target_1_path)
        data["target"] = target
        target_8_path = os.path.join(self.label_root, sequence, frame_id + "_1_8.npy")
        target_1_8 = np.load(target_8_path)
        CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
        data["CP_mega_matrix"] = CP_mega_matrix

        # Compute the masks, each indicate the voxels of a local frustum
        if self.split != "test":
            projected_pix_output = data["projected_pix_{}".format(self.output_scale)]
            pix_z_output = data[
                "pix_z_{}".format(self.output_scale)
            ]
            frustums_masks, frustums_class_dists = compute_local_frustums(
                projected_pix_output,
                pix_z_output,
                target,
                self.img_W,
                self.img_H,
                dataset="kitti",
                n_classes=19,
                size=self.frustum_size,
            )
        else:
            frustums_masks = None
            frustums_class_dists = None
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        img = Image.open(rgb_path).convert("RGB")

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:376, :1408, :]  # crop image

        # Fliplr the image
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            for scale in scale_3ds:
                key = "projected_pix_" + str(scale)
                data[key][:, 0] = img.shape[1] - 1  -data[key][:, 0]

        data["img"] = self.normalize_rgb(img)

        return data
    


    def __len__(self):
        return len(self.scans)




    @staticmethod
    def read_calib():
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        P = np.array(
                [
                    552.554261,
                    0.000000,
                    682.049453,
                    0.000000,
                    0.000000,
                    552.554261,
                    238.769549,
                    0.000000,
                    0.000000,
                    0.000000,
                    1.000000,
                    0.000000,
                ]
            ).reshape(3, 4)

        cam2velo = np.array(
                [   
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
                ]
        ).reshape(3, 4)
        C2V = np.concatenate(
            [cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0
        )
        # print("C2V: ", C2V)
        V2C = np.linalg.inv(C2V)
        # print("V2C: ", V2C)
        V2C = V2C[:3, :]
        # print("V2C: ", V2C)
  
        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = P
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = V2C
        return calib_out