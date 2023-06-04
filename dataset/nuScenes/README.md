# SSCBench-nuScenes

## Change Log
* 2023/06: Initial release.

## Overview
The [nuScenes dataset](https://www.nuscenes.org/nuscenes) consists of one thousand 20-second scenes with labels provided only for the training and validation set, totaling 850 scenes. From the publicly available 850 scenes, we allocate 500 scenes for training, 200 scenes for validation, and 150 scenes for testing. This distribution results in 20,064 frames for training, 8,050 frames for validation, and 5,949 frames for testing, totaling 34,078 frames (~34K). 

## Folder Structure and format
The folder structure of the nuScenes dataset is as follows:
```
dataset/nuScenes/
|-- trainval
|   |-- sequences
|   |   |-- 000000
|   |   |   |-- image_2
|   |   |   |   |-- 0000.png
|   |   |   |   |-- 0001.png
|   |   |   |   |-- ...
|   |   |   |-- voxels
|   |   |   |   |-- 0000.bin
|   |   |   |   |-- 0000.label
|   |   |   |   |-- 0000.invalid
|   |   |   |   |-- 0001.bin
|   |   |   |   |-- 0001.label
|   |   |   |   |-- 0001.invalid
|   |   |   |   |-- ...
|   |   |-- 000001
|   |   |-- 000002
|   |   |-- ...
|   |-- calib.txt
|   |-- ...
|-- test
|   |-- ...
|-- preprocess
|   |-- trainval
|   |   |-- labels
|   |   |   |-- 000000
|   |   |   |   |-- 0000_1_1.npy
|   |   |   |   |-- 0000_1_2.npy
|   |   |   |   |-- 0000_1_8.npy
|   |   |   |   |-- 0001_1_1.npy
|   |   |   |   |-- 0001_1_2.npy
|   |   |   |   |-- 0001_1_8.npy
|   |   |   |   |-- ...
|   |   |   |-- 000001
|   |   |   |-- ...
|   |-- test
|   |   |-- ...
```

For each frame in the dataset, we provide the following information:
* `image_2`: RGB image of size 1600x900.
* `voxels`: Voxelized point cloud of size 256x256x32. Each voxel is a 0.2x0.2x0.2 meter cube.
    * `*.bin`: Voxelized point cloud in binary format.
    * `*.label`: Voxelized point cloud label in binary format.
    * `*.invalid`: Voxelized point cloud invalid mask in binary format.

For MonoScene and VoxFormer, a preprocessed downsampled version of the dataset is provided in the `preprocess` folder. We provide two scales of downsampled point clouds: 1/1 and 1/8. The downsampled point clouds are stored in the `labels` folder, stored as `.npy` files.

## Data Download
The dataset can be downloaded from [goole drive](https://drive.google.com/drive/folders/1cERh6BgWM457t0pg08hLIKYMbsuandbv?usp=drive_link). The dataset is provided in the form of squashed file system for easy use for singularity containers. 
* If you want to use the dataset on a singularity container, you can mount each squashed file system to the container using the `--overlay` option.
* If you want to use the dataset on a normal system, you can unsquash the file system using the `unsquashfs` command (more details [here](https://manpages.ubuntu.com/manpages/focal/man1/unsquashfs.1.html)). Then, please organize the data as the folder structure described above.