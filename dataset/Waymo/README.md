# SSCBench-Waymo

## Change Log
* 2023/07: The code for processing is [released](https://github.com/ai4ce/SSCBench/tree/main/dataset/Waymo/data_generation).
* 2023/06: The code for processing will be released soon.

## SSCBench-Waymo Overview
The Waymo dataset consists of 1000 scenes for training and validation, as well as 150 scenes for testing, with each scene spanning 20 seconds. We utilize the open-source training and validation scenes and redistribute them into sets of 500, 298, and 202 scenes for training, validation, and testing, respectively. In order to reduce redundancy and training time for our benchmark, we downsample the original data by a factor of 10. This downsampling results in a training set of 10,011 frames, a validation set of 5,936 frames, and a test set of 4,038 frames, totaling 19,985 frames

## License
The Waymo dataset is released under the "Waymo Dataset License Agreement for Non-Commercial Use (August 2019)" License. Please check their [website](https://waymo.com/open/terms/) for more details.

We release SSCBench-Waymo under the same license. When you download or use the SSCBench-Waymo from the website or elsewhere, you are agreeing to comply with the terms of the "Waymo Dataset License Agreement for Non-Commercial Use (August 2019)" License.

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
Complying with the Waymo Dataset License Agreement for Non-Commercial Use (August 2019), we are not allowed to redistribute the original data. Please follow the instructions below to download the data.

First, please visit this [site](https://waymo.com/open/licensing/) and agree to the license agreement.

Then, you should be able to download the dataset from [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_0). Note that we only need the [training](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_0/individual_files/training?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&authuser=1&prefix=&forceOnObjectsSortingFiltering=false) and [validation](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_0/individual_files/validation?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&authuser=1&prefix=&forceOnObjectsSortingFiltering=false) dataset. There are 1000 tfrecord files in total(798 for training dataset, and 202 for validation dataset).
## Data Process
We've released the code for generating voxelized point clouds, calibration files and images in [here](https://github.com/ai4ce/SSCBench/tree/main/dataset/Waymo/data_generation).
