# SSCBench: Monocular 3D Semantic Scene Completion Benchmark in Street Views

[Yiming Li*](https://roboticsyimingli.github.io/), 
[Sihang Li*](https://louis-leee.github.io/), 
[Xinhao Liu*](https://gaaaavin.github.io/), 
[Moonjun Gong*](https://moonjungong.github.io/), 
[Kenan Li](https://github.com/ai4ce/SSCBench), 
[Nuo Chen](https://github.com/ai4ce/SSCBench), 
[Zijun Wang](https://github.com/ai4ce/SSCBench), 
[Zhiheng Li](https://github.com/ai4ce/SSCBench), 
[Tao Jiang](https://github.com/ai4ce/SSCBench), 
[Fisher Yu](https://www.yf.io/), 
[Yue Wang](https://yuewang.xyz/), 
[Hang Zhao](https://hangzhaomit.github.io/), 
[Zhiding Yu](https://chrisding.github.io/), 
[Chen Feng](https://engineering.nyu.edu/faculty/chen-feng)

[[PDF]](https://arxiv.org/abs/2306.09001)

<p align="center">
<img src="./teaser/kitti.gif" width="100%"/>
<p align="center">SSCBench-KITTI-360</p>
<img src="./teaser/nuscenes.gif" width="100%"/>
<p align="center">SSCBench-nuScenes</p>
<img src="./teaser/waymo.gif" width="100%"/>
<p align="center">SSCBench-Waymo</p>
<img src="./teaser/pandaset.gif" width="100%"/>
<p align="center">🔥<strong>[New]</strong> SSCBench-PandaSet</p>
</div>


# News
- [2024/08]: We release [SSCBench-Waymo](dataset/Waymo/) for academic usage.
- [2024/06]: SSCBench is accepted at IROS 2024!
- [2023/10]: We release [OCFBench](https://github.com/ai4ce/Occ4cast#ocfbench), a large-scale dataset for OCF, derived from nuScenes, Lyft, Argoverse, and ApolloScape (Waymo is coming soon).
- [2023/08]: We add demo for SSCBench-PandaSet. We are working on incoporating the dataset
- [2023/06]: We release [SSCBench-KITTI-360](dataset/KITTI-360/) and [SSCBench-nuScenes](dataset/nuScenes/) for academic usage.
- [2023/06]: The preprint version is available on [arXiv](https://arxiv.org/abs/2306.09001).

# Abstract
Semantic scene completion (SSC) is crucial for holistic 3D scene understanding by jointly estimating semantics and geometry from sparse observations. However, progress in SSC, particularly in autonomous driving scenarios, is hindered by the scarcity of  high-quality datasets. To overcome this challenge, we introduce SSCBench, a comprehensive benchmark that integrates scenes from widely-used automotive datasets (e.g., KITTI-360, nuScenes, and Waymo). SSCBench follows an established setup and format in the community, facilitating the easy exploration of the camera- and LiDAR-based SSC across various real-world scenarios. We present quantitative and qualitative evaluations of state-of-the-art algorithms on SSCBench and commit to continuously incorporating novel automotive datasets and SSC algorithms to drive further advancements in this field.

# SSCBench Dataset
SSCBench consists of three carefully designed datasets, all based on existing data sources. For more details, please refer to the [dataset](./dataset) folder.

# Model Checkpoints
We provide the model checkpoints of the experiments reported in the paper. The checkpoints can be accessed on [google drive](https://drive.google.com/drive/folders/1583Xy0nh46vNXg_StWvIp2B8IXij92Bm?usp=sharing). 

Note that the provided checkpoints are trained with the unified class labels. Details of class mappings can be found in the [configs](./dataset/configs) folder.

# Related SSC Projects
- [Semantic Scene Completion from a Single Depth Image](https://github.com/shurans/sscnet), CVPR 2017
- [LMSCNet: Lightweight Multiscale 3D Semantic Completion](https://github.com/astra-vision/LMSCNet), 3DV 2020
- [MonoScene: Monocular 3D Semantic Scene Completion](https://github.com/astra-vision/MonoScene), CVPR 2022
- [VoxFormer: a Cutting-edge Baseline for 3D Semantic Occupancy Prediction](https://github.com/NVlabs/VoxFormer), CVPR 2023
- [TPVFormer: An academic alternative to Tesla's Occupancy Network](https://github.com/wzzheng/TPVFormer), CVPR2023
- [OccFormer: Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction](https://github.com/zhangyp15/OccFormer), ICCV 2023
- [SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving](https://github.com/weiyithu/SurroundOcc), ICCV 2023
- [S4C: Self-Supervised Semantic Scene Completion with Neural Fields](https://ahayler.github.io/publications/s4c/), arXiv 2023

## Related Dataset/Benchmark
- [Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving](https://github.com/Tsinghua-MARS-Lab/Occ3D), arXiv 2023
- [OpenOccupancy: A Large Scale Benchmark for Surrounding Semantic Occupancy Perception](https://github.com/JeffWang987/OpenOccupancy), ICCV 2023
- [Occ4cast: LiDAR-based 4D Occupancy Completion and Forecasting](https://github.com/ai4ce/Occ4cast/), arXiv 2023.

# License
Due to the license of the different original datasets, we release SSCBench under the following licenses:
- SSCBench-KITTI-360: [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/)
- SSCBench-nuScenes: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- SSCBench-Waymo: [Waymo Dataset License Agreement for Non-Commercial Use (August 2019)](https://waymo.com/open/terms/)

For more details, please refer to the [dataset](./dataset) folder file.

# Bibtex
If this work is helpful for your research, please cite the following BibTeX entry.

```
@inproceedings{li2024sscbench,
      title={SSCBench: A Large-Scale 3D Semantic Scene Completion Benchmark for Autonomous Driving}, 
      author={Li, Yiming and Li, Sihang and Liu, Xinhao and Gong, Moonjun and Li, Kenan and Chen, Nuo and Wang, Zijun and Li, Zhiheng and Jiang, Tao and Yu, Fisher and Wang, Yue and Zhao, Hang and Yu, Zhiding and Feng, Chen},
      booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2024}
}
```

# Star History
[![Star History Chart](https://api.star-history.com/svg?repos=ai4ce/SSCBench&type=Date)](https://star-history.com/#ai4ce/SSCBench&Date)
