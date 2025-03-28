#!/bin/bash -e
#SBATCH --job-name=rebuttal9
#SBATCH --output=/lustre/scratch/client/movian/research/users/hainn14/otdd/spp_noti/finetune9.out
#SBATCH --error=/lustre/scratch/client/movian/research/users/hainn14/otdd/spp_noti/finetune9.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=125G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.HaiNN14@vinai.io

module purge
module load python/miniconda3/miniconda3

# Corrected line
eval "$(conda shell.bash hook)"

conda activate /lustre/scratch/client/movian/research/users/hainn14/envs/otdd
cd /lustre/scratch/client/movian/research/users/hainn14/otdd

python3 resnet18_finetune.py --num_epochs 50 --parent_dir saved_split_task_10  --source_task 9