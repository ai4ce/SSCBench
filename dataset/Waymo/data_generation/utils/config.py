# Copyright 2022 Tao Jiang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path



tfrecord_dir = "......./Waymo/waymo_format"

output_dir = '......./Waymo/output/'
stuff_dir = '......./Waymo/stuff/'
object_dir = '......./Waymo/object/'
cam_infos_dir = '......./Waymo/cam_infos/'
voxel_dir =  '......./Waymo/voxel/'

CPU_NUM = 80



UPDATE = False
TYPE_UNKNOWN = 0
TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_SIGN = 3
TYPE_CYCLIST = 4

SEMANTCI_CLASS_NUM = 23

label_name_map = {
    0 : "TYPE_UNDEFINED",
    1 : "TYPE_CAR",
    2 : "TYPE_TRUCK", 
    3 : "TYPE_BUS", 
    4 : "TYPE_OTHER_VEHICLE", 
    5 : "TYPE_MOTORCYCLIST", 
    6 : "TYPE_BICYCLIST",
    7 : "TYPE_PEDESTRIAN",
    8 : "TYPE_SIGN", 
    9: "TYPE_TRAFFIC_LIGHT", 
    10: "TYPE_POLE", 
    11: "TYPE_CONSTRUCTION_CONE", 
    12: "TYPE_BICYCLE", 
    13: "TYPE_MOTORCYCLE", 
    14: "TYPE_BUILDING", 
    15: "TYPE_VEGETATION",
    16: "TYPE_TREE_TRUNK",
    17: "TYPE_CURB",
    18: "TYPE_ROAD",
    19: "TYPE_LANE_MARKER",
    20: "TYPE_OTHER_GROUND",
    21: "TYPE_WALKABLE",
    22: "TYPE_SIDEWALK",
}
class_names = ['TYPE_UNDEFINED', 'TYPE_CAR', 'TYPE_TRUCK', 'TYPE_BUS', 'TYPE_OTHER_VEHICLE', 'TYPE_MOTORCYCLIST', 'TYPE_BICYCLIST', 'TYPE_PEDESTRIAN', 'TYPE_SIGN', 'TYPE_TRAFFIC_LIGHT', 'TYPE_POLE', 'TYPE_CONSTRUCTION_CONE', 'TYPE_BICYCLE', 'TYPE_MOTORCYCLE', 'TYPE_BUILDING', 'TYPE_VEGETATION', 'TYPE_TREE_TRUNK', 'TYPE_CURB', 'TYPE_ROAD', 'TYPE_LANE_MARKER', 'TYPE_OTHER_GROUND', 'TYPE_WALKABLE', 'TYPE_SIDEWALK']
learning_map = {
    0 : -1,
    1: 0, # npc
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 1, # sign/pole/TYPE_CONSTRUCTION_CONE
    9: 1,
    10: 1,
    11: 1,
    12: 0,
    13: 0,
    14: 3, # building
    15: 4, # vegetation
    16: 4,
    17: 2, # curb/road/
    18: 2,
    19: 2,
    20: 2,
    21: 2,
    22: 2,
}
