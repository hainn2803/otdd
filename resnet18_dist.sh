#!/bin/bash

for source in {0..9}; do
  for target in $(seq $((source + 1)) 9); do
    echo "Running source=$source, target=$target"
    python3 resnet18_dist.py --source $source --target_tasks $target --num_samples 3000 --num_projections 100000
  done
done

