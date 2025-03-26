import argparse
import torch
import numpy as np
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from otdd.pytorch.method5 import compute_pairwise_distance


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def get_task_numbers(parent_dir):
    """Get sorted list of task numbers from data files with validation"""
    task_numbers = []
    try:
        for fname in os.listdir(parent_dir):
            if fname.startswith("data_task_") and fname.endswith("_size_10000.pt"):
                parts = fname.split("_")
                if len(parts) >= 3 and parts[2].isdigit():
                    task_num = int(parts[2])
                    task_numbers.append(task_num)
        return sorted(task_numbers)
    except FileNotFoundError:
        raise ValueError(f"Directory {parent_dir} not found")
    except Exception as e:
        raise RuntimeError(f"Error scanning directory: {str(e)}")



def load_task_data(task_num, parent_dir, sample_size=400, seed=42):
    """Load task data with balanced class sampling"""

    np.random.seed(seed)
    data_path = f'{parent_dir}/data_task_{task_num}_size_10000.pt'
    labels_path = f'{parent_dir}/labels_task_{task_num}_size_10000.pt'

    # Load full dataset
    task_data = torch.load(data_path)
    task_labels = torch.load(labels_path)
    
    labels_np = task_labels.numpy()
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    num_classes = len(unique_labels)
    
    samples_per_class = sample_size // num_classes
    
    selected_indices = []
    
    # Stratified sampling with balanced classes
    for idx, label in enumerate(unique_labels):
        # Get indices for this class
        class_indices = np.where(labels_np == label)[0]
        
        selected = np.random.permutation(class_indices)[:samples_per_class]
        selected_indices.extend(selected)
    
    selected_indices = np.random.permutation(selected_indices)[:sample_size]
    
    return TensorDataset(
        task_data[selected_indices],
        task_labels[selected_indices]
    )
        


def compute_pairwise_distances(parent_dir, output_file, source_task=None, target_tasks=None, num_samples=1000, num_projections=10000):
    """Compute distances between source task and multiple target tasks"""
    total_start = time.time()
    time_metrics = {}
    
    # Phase 1: Data Preparation and Validation
    phase_start = time.time()

    # Create task pairs
    task_pairs = [(source_task, t) for t in target_tasks if t != source_task]
    unique_tasks = {source_task}.union(set(target_tasks))
    
    # Load data for needed tasks
    task_datasets = {}
    for task in unique_tasks:
        task_datasets[task] = load_task_data(task_num=task, parent_dir=parent_dir, sample_size=num_samples)
    
    time_metrics['data_loading'] = time.time() - phase_start

    # Phase 2: Distance Computation
    kwargs = {
        "dimension": 224,
        "num_channels": 3,
        "num_moments": 5,
        "use_conv": True,
        "precision": "float",
        "p": 2,
        "chunk": 10000
    }

    results = {}
    for s, t in task_pairs:
        # Create dataloaders for this pair
        pair_dataloaders = [
            DataLoader(task_datasets[s], batch_size=256, shuffle=True),
            DataLoader(task_datasets[t], batch_size=256, shuffle=True)
        ]
        
        # Compute distance
        distance_matrix, processing_time = compute_pairwise_distance(
            list_D=pair_dataloaders,
            num_projections=num_projections,
            device=DEVICE,
            evaluate_time=True,
            **kwargs
        )
        
        # Store distance for both directions
        results[f"{s}-{t}"] = distance_matrix[0].item()
        results[f"{t}-{s}"] = distance_matrix[0].item()

    time_metrics['computation'] = processing_time

    # Phase 3: Results Saving
    phase_start = time.time()
    os.makedirs(output_file, exist_ok=True)

    torch.save(distance_matrix, output_file + f"/sourcce_{s}_target_{t}.pt")
    
    with open(output_file + f"/sourcce_{s}_target_{t}.txt", "w") as f:
        f.write(f"Pairwise Distances ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"Source Task: {source_task}\n")
        f.write(f"Target Tasks: {', '.join(map(str, target_tasks))}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Time Metrics:\n")
        f.write(f"  Data Loading: {time_metrics['data_loading']:.2f}s\n")
        f.write(f"  Computation: {time_metrics['computation']:.2f}s\n\n")
        
        f.write("Task Pair\tDistance\n")
        for pair in sorted(results.keys(), key=lambda x: list(map(int, x.split('-')))):
            f.write(f"{pair}\t{results[pair]:.4f}\n")
    
    time_metrics['total'] = time.time() - total_start

    print(f"Results saved to {output_file}")
    print("\nTime Breakdown:")
    for phase, t in time_metrics.items():
        print(f"- {phase.capitalize()}: {t:.2f} seconds")
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute task distances using source-target pairs')
    parser.add_argument('--source', type=int, required=True,
                       help='Source task number')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Source task number')
    parser.add_argument('--num_projections', type=int, default=100000,
                       help='Source task number')
    parser.add_argument('--target_tasks', type=int, nargs='+', default=None, 
                       help="Target tasks for fine-tuning (0-9)")
    parser.add_argument('--output', default="dist/task_distances.txt",
                       help='Output file name')
    parser.add_argument('--parent_dir', default="saved_split_tiny_imagenet",
                       help='Parent directory with task data')
    
    args = parser.parse_args()

    if args.target_tasks is None:
        args.target_tasks = [i for i in range(10) if i != args.source]

    args.output = args.parent_dir + "/dist"

    print(f"Starting computation on {DEVICE}...")
    results = compute_pairwise_distances(
        parent_dir=args.parent_dir,
        output_file=args.output,
        source_task=args.source,
        target_tasks=args.target_tasks,
        num_samples=args.num_samples,
        num_projections=args.num_projections
    )