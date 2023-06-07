# SSCBench-KITTI-360

## SSCBench-KITTI-360 Overview
KITTI-360 covers a driving distance of 73.7km, comprising 300K images and 80K laser scans. Leveraging the open-source training and validation set, we build SSCBench-KITTI-360 consisting of 9 long sequences. To reduce redundancy, we sample every 5 frames following the SemanticKITTI SSC benchmark. The training set includes 8,487 frames from scenes 00, 02-05, 07, and 10, while the validation set comprises 1,812 frames from scene 06. The testing set comprises 2,566 frames from scene 09. In total, the dataset contains 12,865 frames
## Folder Structure and format
The folder structure of the Waymo dataset is as follows:
```
dataset/Waymo/
|-- final_2 # training and validation set
|   |-- dataset
|   |   |-- sequences # 500 (train) + 298 (validation) + 202 (test) scenes
|   |   |   |-- 000
|   |   |   |   |-- image_2 # RGB images
|   |   |   |   |   |-- 0000.png
|   |   |   |   |   |-- 0001.png
|   |   |   |   |   |-- ...
|   |   |   |   |-- voxels # voxelized point clouds
|   |   |   |   |   |-- 000.npz # dictionary, ['input'] = voxelized input, ['labels'] = voxelized label, ['invalid'] = voxelized invalid mask
|   |   |   |   |   |-- 015.npz # sampled every 15 frames, we will provide waymo sampled every 10 frames in the future.
|   |   |   |   |   |-- 030.npz
|   |   |   |   |   |-- 045.npz
|   |   |   |   |   |-- ...
|   |   |   |-- 001
|   |   |   |-- 002
|   |   |   |-- ...
|   |   |   |-- 999
|   |-- convert # contains images and calibration information
|   |   |-- image_2 # 500 (train) + 298 (validation) + 202 (test) scenes
|   |   |   |-- 000
|   |   |   |   |-- 0000.png #RGB images
|   |   |   |   |-- 0001.png
|   |   |   |   |-- ...
|   |   |   |-- 001
|   |   |   |-- 002
|   |   |   |-- ...
|   |   |   |-- 999
|   |   |-- calib # 500 (train) + 298 (validation) + 202 (test) scenes
|   |   |   |-- 000
|   |   |   |   |-- calib.txt # calibration information
|   |   |   |-- 001
|   |   |   |-- 002
|   |   |   |-- ...
|   |   |   |-- 999
|-- preprocess # preprocessed downsampled labels
|   |-- labels
|   |   |-- 000
|   |   |   |-- 0000_1_1.npy # original labels
|   |   |   |-- 0000_1_8.npy # 8x downsampled labels
|   |   |   |-- 0001_1_1.npy
|   |   |   |-- 0001_1_8.npy
|   |   |   |-- ...
|   |   |-- 001
|   |   |-- ...
```

For each frame in the dataset, we provide the following information:
* `image_2`: RGB image of size 1280x1960.
* `voxels`: Voxelized point cloud of size 256x256x32. Each voxel is a 0.2x0.2x0.2 meter cube.
    * `*.npz`: A dictionary, ['input'] = voxelized point cloud in binary format, ['labels'] = voxelized point cloud label in binary format, ['invalid'] = voxelized point cloud invalid mask in binary format.

For MonoScene and VoxFormer, a preprocessed downsampled version of the dataset is provided in the `preprocess` folder. We provide two scales of downsampled point clouds: 1/1 and 1/8. The downsampled point clouds are stored in the `labels` folder, stored as `.npy` files.

## Data Download
The dataset can be downloaded from [goole drive](https://drive.google.com/drive/folders/1OnCsqzwLS6VQFtoI3l77JTUsqbwZrjIU?usp=drive_link). The dataset is provided in the form of squashed file system for easy use for singularity containers. 
* If you want to use the dataset on a singularity container, you can mount each squashed file system to the container using the `--overlay` option.
* If you want to use the dataset on a normal system, you can unsquash the file system using the `unsquashfs` command (more details [here](https://manpages.ubuntu.com/manpages/focal/man1/unsquashfs.1.html)). Then, please organize the data as the folder structure described above.
