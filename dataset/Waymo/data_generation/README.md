# Installation
Change the cuda version in ``requirement.txt`` to match your env

For example, if your cuda version is 11.7, change it like this:
cumm-cu117==0.4.7
spconv-cu117==2.3.3
```
$ pip3 install -r requirement.txt
```

# Get Started
## Prerequisites
Change split = 'training' # 'validation' in waymo_pipline.sh

Change input ``tfrecord_dir`` to tfrecord path in config.py

Change output ``output_dir``/``stuff_dir``/``object_dir``/``cam_infos_dir``/``voxel_dir`` in config.py

Change CPU_NUM  in config.py to match your machine

## Single machine
```
$ ./waymo_pipline.sh

Recommend running each command in waymo_pipline.sh one by one to deal with possible errors
```
## Multi machine
To speedup, you can split dataset, change ``start_seq`` and ``end_seq`` value in waymo_pipline.sh, then run bash file in different machine

## Debug
Each file contain ``debug()`` function, change the params and debug as you need
