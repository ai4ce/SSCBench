<div align="center">  
  
# SSCBench: A Large-Scale 3D Semantic Scene Completion Benchmark for Autonomous Driving
<div align="center">  
  
![](./teaser/kitti_.gif "")
  
SSCBench-KITTI-360
</div>
  
</div>
<div align="center"> 
  
![](./teaser/nuscenes_.gif "")

SSCBench-nuScenes
</div>

<div align="center">  
  
![](./teaser/waymo_.gif "")
  
SSCBench-Waymo
</div>




> SSCBench: A Large-Scale 3D Semantic Scene Completion Benchmark for Autonomous Driving
> 
> [Yiming Li*](https://scholar.google.com/citations?hl=en&user=i_aajNoAAAAJ&view_op=list_works&sortby=pubdate), [Sihang Li*], [Xinhao Liu*], [Moonjun Gong*], [Kenan Li], [Nuo Chen], [Zijun Wang], [Zhiheng Li], [Tao Jiang], [Fisher Yu], [Yue Wang], [Hang Zhao], [Zhiding Yu](https://scholar.google.com/citations?user=1VI_oYUAAAAJ&hl=en), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ&hl=en)

>  [[PDF]](https://github.com/ai4ce/SSCBench/) [[Project]](https://github.com/ai4ce/SSCBench/) 

## News
- [2023/06]: SSCBench has been submitted to NeurIPS 2023 Datasets and Benchmarks.

## Abstract
Semantic scene completion (SSC) is crucial for holistic 3D scene understanding by jointly estimating semantics and geometry from sparse observations. However, progress in SSC, particularly in autonomous driving scenarios, is hindered by the scarcity of  high-quality datasets. To overcome this challenge, we introduce SSCBench, a comprehensive benchmark that integrates scenes from widely-used automotive datasets (e.g., KITTI-360, nuScenes, and Waymo). SSCBench follows an established setup and format in the community, facilitating the easy exploration of the camera- and LiDAR-based SSC across various real-world scenarios. We present quantitative and qualitative evaluations of state-of-the-art algorithms on SSCBench and commit to continuously incorporating novel automotive datasets and SSC algorithms to drive further advancements in this field.

## SSCBench-KITTI-360
### SSCBench-KITTI-360 Overview
KITTI-360 covers a driving distance of 73.7km, comprising 300K images and 80K laser scans. Leveraging the open-source training and validation set, we build SSCBench-KITTI-360 consisting of 9 long sequences. To reduce redundancy, we sample every 5 frames following the SemanticKITTI SSC benchmark. The training set includes 8,487 frames from scenes 00, 02-05, 07, and 10, while the validation set comprises 1,812 frames from scene 06. The testing set comprises 2,566 frames from scene 09. In total, the dataset contains 12,865 frames
### Folder Structure and format
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

### Data Download
The dataset can be downloaded from [goole drive](https://drive.google.com/drive/folders/1OnCsqzwLS6VQFtoI3l77JTUsqbwZrjIU?usp=drive_link). The dataset is provided in the form of squashed file system for easy use for singularity containers. 
* If you want to use the dataset on a singularity container, you can mount each squashed file system to the container using the `--overlay` option.
* If you want to use the dataset on a normal system, you can unsquash the file system using the `unsquashfs` command (more details [here](https://manpages.ubuntu.com/manpages/focal/man1/unsquashfs.1.html)). Then, please organize the data as the folder structure described above.

## SSCBench-nuScenes
### SSCBench-nuScenes Overview
The nuScenes dataset consists of one thousand 20-second scenes with labels provided only for the training and validation set, totaling 850 scenes. From the publicly available 850 scenes, we allocate 500 scenes for training, 200 scenes for validation, and 150 scenes for testing. This distribution results in 20,064 frames for training, 8,050 frames for validation, and 5,949 frames for testing, totaling 34,078 frames (~34K).
### Folder Structure and format
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

### Data Download
The dataset can be downloaded from [goole drive](https://drive.google.com/drive/folders/1cERh6BgWM457t0pg08hLIKYMbsuandbv?usp=drive_link). The dataset is provided in the form of squashed file system for easy use for singularity containers. 
* If you want to use the dataset on a singularity container, you can mount each squashed file system to the container using the `--overlay` option.
* If you want to use the dataset on a normal system, you can unsquash the file system using the `unsquashfs` command (more details [here](https://manpages.ubuntu.com/manpages/focal/man1/unsquashfs.1.html)). Then, please organize the data as the folder structure described above.

## SSCBench-Waymo
### SSCBench-Waymo Overview
The Waymo dataset consists of 1000 scenes for training and validation, as well as 150 scenes for testing, with each scene spanning 20 seconds. We utilize the open-source training and validation scenes and redistribute them into sets of 500, 298, and 202 scenes for training, validation, and testing, respectively. In order to reduce redundancy and training time for our benchmark, we downsample the original data by a factor of 10. This downsampling results in a training set of 10,011 frames, a validation set of 5,936 frames, and a test set of 4,038 frames, totaling 19,985 frames
### Folder Structure and format
The folder structure of the Waymo dataset is as follows:
```
dataset/Waymo/
|-- final_2 # training and validation set
|   |-- dataset
|   |   |-- sequences # 500 (train) + 298 (validation) + 202 (test) scenes
|   |   |   |-- 000
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

### Data Download
The dataset can be downloaded from [goole drive](https://drive.google.com/drive/folders/1OnCsqzwLS6VQFtoI3l77JTUsqbwZrjIU?usp=drive_link). The dataset is provided in the form of squashed file system for easy use for singularity containers. 
* If you want to use the dataset on a singularity container, you can mount each squashed file system to the container using the `--overlay` option.
* If you want to use the dataset on a normal system, you can unsquash the file system using the `unsquashfs` command (more details [here](https://manpages.ubuntu.com/manpages/focal/man1/unsquashfs.1.html)). Then, please organize the data as the folder structure described above.
