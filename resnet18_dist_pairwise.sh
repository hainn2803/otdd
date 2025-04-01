#!/bin/bash -e
#SBATCH --job-name=pairwise
#SBATCH --output=/lustre/scratch/client/movian/research/users/hainn14/otdd/spp_noti/pairwise.out
#SBATCH --error=/lustre/scratch/client/movian/research/users/hainn14/otdd/spp_noti/pairwise.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=125G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=movianr
#SBATCH --mail-type=all
#SBATCH --mail-user=v.HaiNN14@vinai.io

module purge
module load python/miniconda3/miniconda3

# Corrected line
eval "$(conda shell.bash hook)"

conda activate /lustre/scratch/client/movian/research/users/hainn14/envs/otdd
cd /lustre/scratch/client/movian/research/users/hainn14/otdd

python3 resnet18_dist_pairwise.py --num_samples 5000 --num_projections 500000 --parent_dir saved_split_task_10