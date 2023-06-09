#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=monoscene
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=log/dm_%j.out
#SBATCH --error=log/dm_%j.err
#SBATCH --gres=gpu:a100:1


module purge
cd /scratch/$USER/sscbench/MonoScene

singularity exec --nv \
	    --overlay /scratch/$USER/environments/monoscene.ext3:ro \
        --overlay /scratch/$USER/dataset/waymo/waymo.sqf:ro \
        --overlay /scratch/$USER/dataset/waymo/preprocess.sqf:ro \
	    /scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; 
        conda activate monoscene; 
        python monoscene/scripts/eval_waymo.py"