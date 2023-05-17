#!/bin/bash
#SBATCH --chdir /scratch/izar/treil
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 200G
#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --qos dlav
#SBATCH --account civil-459-2023
#SBATCH --time 35:00:00

source /home/treil/anaconda3/bin/activate pyskl
conda info --envs
bash /home/treil/dlav_pyskl_2d_skeleton/tools/dist_train.sh /home/treil/dlav_pyskl_2d_skeleton/configs/posec3d/c3d_light_ntu60_xsub/joint.py 2 --validate --test-last --test-best
