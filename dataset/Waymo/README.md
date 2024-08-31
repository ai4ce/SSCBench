# SSCBench-Waymo

## Change Log
* 2024/08: We released SSCBench-Waymo for academic usage.
* 2023/06: The code for processing will be released soon.

## SSCBench-Waymo Overview
To construct SSCBench-Waymo, we utilize the open-source training and validation scenes and redistribute them into sets of 500, 298, and 202 scenes for training, validation, and testing, respectively. We use only the annotated key frames, which results in a training set of 14,943 frames, a validation set of 8,778 frames, and a test set of 5,946 frames, totaling 29,667 frames

## License
The Waymo dataset is released under the "Waymo Dataset License Agreement for Non-Commercial Use (August 2019)" License. Please check their [website](https://waymo.com/open/terms/) for more details.

We release SSCBench-Waymo under the same license. When you download or use the SSCBench-Waymo from the website or elsewhere, you are agreeing to comply with the terms of the "Waymo Dataset License Agreement for Non-Commercial Use (August 2019)" License.

## Folder Structure and format
The folder structure of the Waymo dataset is as follows:
```
dataset/Waymo/ # 500 (train) + 298 (validation) + 202 (test) scenes
|-- voxels # training and validation set
|   |   |-- 000
|   |   |   |-- 000.npz # dictionary, ['input'] = voxelized input, ['labels'] = voxelized label, ['invalid'] = voxelized invalid mask
|   |   |   |-- 001.npz # labeled key frames only
|   |   |   |-- 002.npz
|   |   |   |-- 003.npz
|   |   |   |-- ...
|   |   |-- 001
|   |   |-- 002
|   |   |-- ...
|   |   |-- 999
|-- kitti_format_cam # contains images and calibration information
|   |-- image_1 # Front camera images
|   |   |-- 000
|   |   |   |-- 000.png #RGB images
|   |   |   |-- 001.png
|   |   |   |-- ...
|   |   |-- 001
|   |   |-- 002
|   |   |-- ...
|   |   |-- 999
|   |-- image_2 # Front left camera images
|   |   |-- 000
|   |   |   |-- 000.png #RGB images
|   |   |   |-- 001.png
|   |   |   |-- ...
|   |   |-- 001
|   |   |-- 002
|   |   |-- ...
|   |   |-- 999
|   |-- image_3 # Front right camera images
|   |   |-- 000
|   |   |   |-- 000.png #RGB images
|   |   |   |-- 001.png
|   |   |   |-- ...
|   |   |-- 001
|   |   |-- 002
|   |   |-- ...
|   |   |-- 999
|   |-- calib1 # Front camera calibrations in KITTI format
|   |   |-- 000
|   |   |   |-- calib.txt # calibration information
|   |   |-- 001
|   |   |-- 002
|   |   |-- ...
|   |   |-- 999
|   |-- calib2 # Front camera calibrations in KITTI format
|   |   |-- 000
|   |   |   |-- calib.txt # calibration information
|   |   |-- 001
|   |   |-- 002
|   |   |-- ...
|   |   |-- 999
|   |-- calib3 # Front camera calibrations in KITTI format
|   |   |-- 000
|   |   |   |-- calib.txt # calibration information
|   |   |-- 001
|   |   |-- 002
|   |   |-- ...
|   |   |-- 999
|-- preprocess # Preprocessed labels for camera-based methods
|   |-- unified # 11 labels in total
|   |   |-- labels
|   |   |   |-- 000
|   |   |   |   |-- 000_1_1.npy # original labels
|   |   |   |   |-- 000_1_2.npy # 2x downsampled labels
|   |   |   |   |-- 000_1_8.npy # 8x downsampled labels
|   |   |   |   |-- 001_1_1.npy
|   |   |   |   |-- 001_1_2.npy
|   |   |   |   |-- 001_1_8.npy
|   |   |   |   |-- ...
|   |   |   |-- 001
|   |   |   |-- ...
|   |-- not_unified # 15 labels in total
|   |   |-- labels
|   |   |   |-- 000
|   |   |   |   |-- 000_1_1.npy # original labels
|   |   |   |   |-- 000_1_2.npy # 2x downsampled labels
|   |   |   |   |-- 000_1_8.npy # 8x downsampled labels
|   |   |   |   |-- 001_1_1.npy
|   |   |   |   |-- 001_1_2.npy
|   |   |   |   |-- 001_1_8.npy
|   |   |   |   |-- ...
|   |   |   |-- 001
|   |   |   |-- ...
```

For each frame in the dataset, we provide the following information:
* `image_1`: Front camera RGB image of size 1280x1960.
* `image_2`: Front left camera RGB image of size 1280x1960.
* `image_3`: Front right camera RGB image of size 1280x1960.
* `voxels`: Voxelized point cloud of size 256x256x32. Each voxel is a 0.2x0.2x0.2 meter cube.
    * `*.npz`: A dictionary, ['input'] = voxelized point cloud in binary format, ['labels'] = voxelized point cloud label in binary format, ['invalid'] = voxelized point cloud invalid mask in binary format.

For MonoScene and VoxFormer, a preprocessed downsampled version of the dataset is provided in the `preprocess` folder. We provide three scales of downsampled point clouds: 1/1, 1/2 and 1/8. The downsampled point clouds are stored in the `labels` folder, stored as `.npy` files.

## Data Download
The dataset can be downloaded from [google drive](https://drive.google.com/drive/u/1/folders/1dm6O6H1pMLF9pv3RWfnULQsJ11PMpJys). The dataset is provided in the form of squashed file system for easy use for singularity containers. 
* If you want to use the dataset on a singularity container, you can mount each squashed file system to the container using the `--overlay` option.
* If you want to use the dataset on a normal system, you can unsquash the file system using the `unsquashfs` command (more details [here](https://manpages.ubuntu.com/manpages/focal/man1/unsquashfs.1.html)). Then, please organize the data as the folder structure described above.