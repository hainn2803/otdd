parent_dir="saved_3"
exp_type="split_size"
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 200 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 500 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 1000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 2000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 5000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 8000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 10000 --num_projections 10000 --num_classes 100