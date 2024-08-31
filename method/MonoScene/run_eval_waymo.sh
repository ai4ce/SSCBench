#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem=200G
#SBATCH --job-name=monoscene_eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gw2396@nyu.edu
#SBATCH --output=/scratch/gw2396/projects/SSCBench-Waymo-Code/MonoScene/slurm_log/eval_kitti2waymo_0_%j.out
#SBATCH --error=/scratch/gw2396/projects/SSCBench-Waymo-Code/MonoScene/slurm_log/eval_kitti2waymo_0_%j.err
#SBATCH --gres=gpu:a100:1


module purge
cd /scratch/gw2396/projects/SSCBench-Waymo-Code/MonoScene

singularity exec --nv \
	    --overlay /scratch/$USER/envs/monoscene.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "export SSL_CERT_DIR=/etc/ssl/certs/;
		source /ext3/env.sh; 
        conda activate kitti_ssc; 
        python monoscene/scripts/eval_waymo.py"
