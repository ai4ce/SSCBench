#! /bin/sh

set -x

split=training # training/validation
start_seq=0
end_seq=798 # 798 for training, 202 for validation
gpu_num=4

python utils/agg_frames_step1.py --split $split --start $start_seq --end $end_seq
python utils/agg_frames_step2.py --split $split --start $start_seq --end $end_seq
python utils/generate_cam_infos.py --split $split --start $start_seq --end $end_seq
python utils/surface.py --start $start_seq --end $end_seq

if [ "$split" = "validation" ]; then
    python utils/raymodel_validation.py --start $start_seq --end $end_seq --gpu $gpu_num
else
    python utils/raymodel_train.py --start $start_seq --end $end_seq --gpu $gpu_num
fi