#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=96GB
#SBATCH --job-name=monoscene
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=log/mono_%j.out
#SBATCH --error=log/mono_%j.err
#SBATCH --gres=gpu:a100:4


module purge
cd /scratch/$USER/sscbench/MonoScene

singularity exec --nv \
	    --overlay /scratch/$USER/environments/monoscene.ext3:ro \
        --overlay /scratch/$USER/dataset/waymo/waymo.sqf:ro \
        --overlay /scratch/$USER/dataset/waymo/preprocess.sqf:ro \
	    /scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; 
        conda activate monoscene; 
        python monoscene/scripts/train_monoscene_waymo.py"