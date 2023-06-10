# SSCBench: A Large-Scale 3D Semantic Scene Completion Benchmark for Autonomous Driving

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
[Zhiding Yu](https://scholar.google.com/citations?user=1VI_oYUAAAAJ&hl=en), 
[Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ&hl=en)

[[PDF]](https://github.com/ai4ce/SSCBench/) [[Project]](https://github.com/ai4ce/SSCBench/) 

<p align="center">
<img src="./teaser/kitti.gif" width="100%"/>
<p align="center">SSCBench-KITTI-360</p>
<img src="./teaser/nuscenes.gif" width="100%"/>
<p align="center">SSCBench-nuScenes</p>
<img src="./teaser/waymo.gif" width="100%"/>
<p align="center">SSCBench-Waymo</p>
</div>


## News
- [2023/06]: SSCBench is submitted to NeurIPS 2023 Datasets and Benchmarks.

## Abstract
Semantic scene completion (SSC) is crucial for holistic 3D scene understanding by jointly estimating semantics and geometry from sparse observations. However, progress in SSC, particularly in autonomous driving scenarios, is hindered by the scarcity of  high-quality datasets. To overcome this challenge, we introduce SSCBench, a comprehensive benchmark that integrates scenes from widely-used automotive datasets (e.g., KITTI-360, nuScenes, and Waymo). SSCBench follows an established setup and format in the community, facilitating the easy exploration of the camera- and LiDAR-based SSC across various real-world scenarios. We present quantitative and qualitative evaluations of state-of-the-art algorithms on SSCBench and commit to continuously incorporating novel automotive datasets and SSC algorithms to drive further advancements in this field.

## SSCBench Dataset
SSCBench consists of three carefully designed datasets, all based on existing data sources. For more details, please refer to the [dataset](./dataset) folder.

## License
Due to the license of the different original datasets, we release SSCBench under the following licenses:
- SSCBench-KITTI-360: [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/)
- SSCBench-nuScenes: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- SSCBench-Waymo: [Waymo Dataset License Agreement for Non-Commercial Use (August 2019)](https://waymo.com/open/terms/)

For more details, please refer to the [dataset](./dataset) folder file.

