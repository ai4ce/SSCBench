# SSCBench-nuScenes

## Change Log
* 2023/06: Initial release.

## Overview
The [nuScenes dataset](https://www.nuscenes.org/nuscenes) consists of one thousand 20-second scenes with labels provided only for the training and validation set, totaling 850 scenes. From the publicly available 850 scenes, we allocate 500 scenes for training, 200 scenes for validation, and 150 scenes for testing. This distribution results in 20,064 frames for training, 8,050 frames for validation, and 5,949 frames for testing, totaling 34,078 frames (~34K). 

## License
The nuScenes dataset is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) License (“CC BY-NC-SA 4.0”). Please check their [Terms of Use](https://www.nuscenes.org/terms-of-use) for more details.

We release SSCBench-nuScenes under the same license. When you download or use the SSCBench-KITTI-360 from the website or elsewhere, you are agreeing to comply with the terms of the "CC BY-NC-SA 4.0" Licnese.

## Folder Structure and format
The folder structure of the nuScenes dataset is as follows:
```
dataset/nuScenes/
|-- trainval # training and validation set
|   |-- sequences # 500 (train) + 200 (validation) scenes
|   |   |-- 000000
|   |   |   |-- image_2 # RGB images
|   |   |   |   |-- 0000.png
|   |   |   |   |-- 0001.png
|   |   |   |   |-- ...
|   |   |   |-- voxels # voxelized point clouds
|   |   |   |   |-- 0000.bin # voxelized input
|   |   |   |   |-- 0000.label # voxelized label
|   |   |   |   |-- 0000.invalid # voxelized invalid mask
|   |   |   |   |-- 0001.bin
|   |   |   |   |-- 0001.label
|   |   |   |   |-- 0001.invalid
|   |   |   |   |-- ...
|   |   |-- 000001
|   |   |-- 000002
|   |   |-- ...
|   |-- calib.txt # calibration information
|   |-- ... # other files (not needed)
|-- test # testing set
|   |-- ... # same as trainval
|-- preprocess # preprocessed downsampled labels
|   |-- trainval
|   |   |-- labels
|   |   |   |-- 000000
|   |   |   |   |-- 0000_1_1.npy # original labels
|   |   |   |   |-- 0000_1_2.npy # 2x downsampled labels
|   |   |   |   |-- 0000_1_8.npy # 8x downsampled labels
|   |   |   |   |-- 0001_1_1.npy
|   |   |   |   |-- 0001_1_2.npy
|   |   |   |   |-- 0001_1_8.npy
|   |   |   |   |-- ...
|   |   |   |-- 000001
|   |   |   |-- ...
|   |-- test
|   |   |-- ... # same as trainval
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