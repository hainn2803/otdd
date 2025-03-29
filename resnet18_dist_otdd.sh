#!/bin/bash

for source in {0..9}; do
  for target in $(seq $((source + 1)) 9); do
    echo "Running source=$source, target=$target"
    python3 resnet18_dist_otdd.py --source $source --target_tasks $target --num_samples 2000 --parent_dir saved_split_task_10
  done
done

