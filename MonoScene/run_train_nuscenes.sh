#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=monoscene
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=log/dm_%j.out
#SBATCH --error=log/dm_%j.err
#SBATCH --gres=gpu:a100:2


module purge
cd /scratch/$USER/sscbench/MonoScene

singularity exec --nv \
	    --overlay /scratch/$USER/environments/monoscene.ext3:ro \
        --overlay /scratch/$USER/dataset/nusc_kitti/trainval.sqf:ro \
        --overlay /scratch/$USER/dataset/nusc_kitti/test.sqf:ro \
        --overlay /scratch/$USER/dataset/nusc_kitti/preprocess.sqf:ro \
	    /scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; 
        conda activate monoscene; 
        python monoscene/scripts/train_monoscene.py"