#!/bin/bash -lex
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=rebuttal0
#SBATCH --output=/lustre/scratch/client/movian/research/users/hainn14/otdd/spp_noti/dist.out
#SBATCH --error=/lustre/scratch/client/movian/research/users/hainn14/otdd/spp_noti/dist.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=50GB
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hainn14@vinai.io

# (optional) debugging flags
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_SOCKET_IFNAME=bond0

   
module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"

conda deactivate
conda activate /lustre/scratch/client/movian/research/users/hainn14/envs/otdd
cd /lustre/scratch/client/movian/research/users/hainn14/otdd

python3 resnet18_dist_pairwise.py --num_samples 5000 --num_projections 500000