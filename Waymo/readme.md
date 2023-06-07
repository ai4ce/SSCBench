# SSCBench-Waymo
## SSCBench-Waymo Overview
The Waymo dataset consists of 1000 scenes for training and validation, as well as 150 scenes for testing, with each scene spanning 20 seconds. We utilize the open-source training and validation scenes and redistribute them into sets of 500, 298, and 202 scenes for training, validation, and testing, respectively. In order to reduce redundancy and training time for our benchmark, we downsample the original data by a factor of 10. This downsampling results in a training set of 10,011 frames, a validation set of 5,936 frames, and a test set of 4,038 frames, totaling 19,985 frames
## Folder Structure and format
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

## Data Download
SSCBench-Waymo is not publicly available due to Waymo Open Dataset's terms of use. However, we will provide code for data preprocessing for easy accesss to our dataset. You can download Waymo Perception Dataset v1.4.0 directly from their [official website](https://waymo.com/intl/en_us/open/download/).
