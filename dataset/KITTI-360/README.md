# SSCBench-KITTI-360

## Change Log
* 2023/06: Initial release.

## Overview
KITTI-360 covers a driving distance of 73.7km, comprising 300K images and 80K laser scans. Leveraging the open-source training and validation set, we build SSCBench-KITTI-360 consisting of 9 long sequences. To reduce redundancy, we sample every 5 frames following the SemanticKITTI SSC benchmark. The training set includes 8,487 frames from scenes 00, 02-05, 07, and 10, while the validation set comprises 1,812 frames from scene 06. The testing set comprises 2,566 frames from scene 09. In total, the dataset contains 12,865 frames

## License
The KITTI-360 dataset is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/) License ("CC BY-NC-SA 3.0"). Please check their [website](https://www.cvlibs.net/datasets/kitti-360/index.php#:~:text=com/dcmlr/kitti360_ros_player-,Copyright,-All%20datasets%20and) for more details.

We release SSCBench-KITTI-360 under the same license. When you download or use the SSCBench-KITTI-360 from the website or elsewhere, you are agreeing to comply with the terms of the "CC BY-NC-SA 3.0" License.

## Folder Structure and format
The folder structure of the SSCBench-KITTI-360 dataset is as follows:
```
dataset/kitti360/
|-- KITTI-360 
|   |-- data_2d_raw
|   |   |-- 2013_05_28_drive_0000_sync # train:[0, 2, 3, 4, 5, 7, 10] + val:[6] + test:[9]
|   |   |   |-- image_00
|   |   |   |   |-- data_rect # RGB images for left camera
|   |   |   |   |   |-- 000000.png
|   |   |   |   |   |-- 000001.png
|   |   |   |   |   |-- ...
|   |   |   |   |-- timestamps.txt
|   |   |   |-- image_01
|   |   |   |   |-- data_rect # RGB images for right camera
|   |   |   |   |   |-- 000000.png
|   |   |   |   |   |-- 000001.png
|   |   |   |   |   |-- ...
|   |   |   |   |-- timestamps.txt
|   |   |   |-- voxels # voxelized point clouds
|   |   |   |   |-- 000000.bin # voxelized input
|   |   |   |   |-- 000000.invalid # voxelized invalid mask
|   |   |   |   |-- 000000.label  #voxelized label
|   |   |   |   |-- 000005.bin # calculate every 5 frames 
|   |   |   |   |-- 000005.invalid
|   |   |   |   |-- 000005.label
|   |   |   |   |-- ...
|   |   |   |-- cam0_to_world.txt
|   |   |   |-- pose.txt # car pose information
|   |   |-- ...
|   |   |-- 2013_05_28_drive_0010_sync 
|   |-- preprocess # preprocessed downsampled labels
|   |   |-- labels # not unified
|   |   |   |-- 2013_05_28_drive_0000_sync 
|   |   |   |   |-- 000000_1_1.npy # original labels
|   |   |   |   |-- 000000_1_8.npy # 8x downsampled labels
|   |   |   |   |-- 000005_1_1.npy
|   |   |   |   |-- 000005_1_8.npy
|   |   |   |   |-- ...
|   |   |   |-- ... 
|   |   |   |-- 2013_05_28_drive_0010_sync
|   |   |-- labels_half # not unified, downsampled 
|   |   |   |-- 2013_05_28_drive_0000_sync 
|   |   |   |   |-- 000000_1_1.npy # original labels
|   |   |   |   |-- 000000_1_8.npy # 8x downsampled labels
|   |   |   |   |-- 000005_1_1.npy
|   |   |   |   |-- 000005_1_8.npy
|   |   |   |   |-- ...
|   |   |   |-- ... 
|   |   |   |-- 2013_05_28_drive_0010_sync
|   |   |-- unified # unified
|   |   |   |-- labels
|   |   |   |   |-- 2013_05_28_drive_0000_sync 
|   |   |   |   |   |-- 000000_1_1.npy # original labels
|   |   |   |   |   |-- 000000_1_8.npy # 8x downsampled labels
|   |   |   |   |   |-- 000005_1_1.npy
|   |   |   |   |   |-- 000005_1_8.npy
|   |   |   |   |   |-- ...
|   |   |   |   |-- ... 
|   |   |   |   |-- 2013_05_28_drive_0010_sync
|   |-- calibration # preprocessed downsampled labels
|   |   |-- calib_cam_to_pose.txt
|   |   |-- calib_cam_to_velo.txt
|   |   |-- calib_sick_to_velo.txt
|   |   |-- image_02.yaml
|   |   |-- image_03.yaml
|   |   |-- perspective.txt
```

For each frame in the dataset, we provide the following information:
* `image_00`: RGB image of size 376x1408.
* `voxels`: Voxelized point cloud of size 256x256x32. Each voxel is a 0.2x0.2x0.2 meter cube.
    * `*.bin`: voxelized point cloud in binary format 
    * `*.label`: voxelized point cloud label in binary format
    * `*.invalid` = voxelized point cloud invalid mask in binary format.

For MonoScene and VoxFormer, a preprocessed downsampled version of the dataset is provided in the `preprocess` folder. We provide two scales of downsampled point clouds: 1/1 and 1/8. The downsampled point clouds are stored in the `labels` folder, stored as `.npy` files.

## Data Download
The dataset can be downloaded from [google drive](https://drive.google.com/drive/folders/1UReVAtWiOKOQRMa6TG19knGHRdVF6MXB). The dataset is provided in the form of squashed file system for easy use for singularity containers. 
* If you want to use the dataset on a singularity container, you can mount each squashed file system to the container using the `--overlay` option.
* If you want to use the dataset on a normal system, you can unsquash the file system using the `unsquashfs` command (more details [here](https://manpages.ubuntu.com/manpages/focal/man1/unsquashfs.1.html)). Then, please organize the data as the folder structure described above.
