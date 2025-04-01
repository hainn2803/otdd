#!/bin/bash -e
#SBATCH --job-name=dist
#SBATCH --output=/lustre/scratch/client/movian/research/users/hainn14/otdd/spp_noti/dist.out
#SBATCH --error=/lustre/scratch/client/movian/research/users/hainn14/otdd/spp_noti/dist.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=125G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=movianr
#SBATCH --mail-type=all
#SBATCH --mail-user=v.HaiNN14@vinai.io

for source in {0..9}; do
  for target in $(seq $((source + 1)) 9); do
    echo "Running source=$source, target=$target"
    python3 resnet18_dist.py --source $source --target_tasks $target --num_samples 10000 --num_projections 500000 --parent_dir saved_split_task_10
  done
done

