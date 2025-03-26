import torch
import numpy as np
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from otdd.pytorch.method5 import compute_pairwise_distance

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_task_data(task_num, parent_dir, sample_size=10000, seed=42):
    """Load task data with reproducible random sampling"""
    try:
        np.random.seed(seed)
        data_path = f'{parent_dir}/data_task_{task_num}_size_10000.pt'
        labels_path = f'{parent_dir}/labels_task_{task_num}_size_10000.pt'

        task_data = torch.load(data_path)
        task_labels = torch.load(labels_path)

        if task_data.shape[0] > sample_size:
            indices = np.random.permutation(task_data.shape[0])[:sample_size]
            task_data = task_data[indices]
            task_labels = task_labels[indices]

        return TensorDataset(task_data, task_labels)
    except Exception as e:
        raise RuntimeError(f"Failed loading task {task_num}: {str(e)}")

def compute_pairwise_distances(parent_dir, output_file="pairwise_distances.txt"):
    """Compute and save pairwise distances with full time tracking"""
    total_start = time.time()
    time_metrics = {}
    
    try:
        # Phase 1: Data Preparation
        phase_start = time.time()
        task_numbers = get_task_numbers(parent_dir)
        print(f"Found {len(task_numbers)} tasks: {task_numbers}")
        
        # Create DataLoaders
        dataloaders = []
        for task_num in task_numbers:
            dataset = load_task_data(task_num, parent_dir)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
            dataloaders.append(dataloader)
        
        time_metrics['data_loading'] = time.time() - phase_start

        # Phase 2: Distance Computation
        phase_start = time.time()
        kwargs = {
            "dimension": 224,
            "num_channels": 3,
            "num_moments": 5,
            "use_conv": True,
            "precision": "float",
            "p": 2,
            "chunk": 10000
        }

        distance_matrix, compute_time = compute_pairwise_distance(
            list_D=dataloaders,
            num_projections=100000,
            device=DEVICE,
            evaluate_time=True,
            **kwargs
        )
        time_metrics['computation'] = compute_time
        time_metrics['total'] = time.time() - total_start

        # Phase 3: Results Saving
        phase_start = time.time()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w") as f:
            # Header with metadata
            f.write(f"Pairwise Distance Matrix ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write(f"Device: {DEVICE}\n")
            f.write(f"Total Tasks: {len(task_numbers)}\n")
            f.write(f"Time Metrics:\n")
            f.write(f"  Data Loading: {time_metrics['data_loading']:.2f}s\n")
            f.write(f"  Computation: {time_metrics['computation']:.2f}s\n")
            f.write(f"  Total: {time_metrics['total']:.2f}s\n\n")
            
            # Matrix data
            f.write("Tasks\t" + "\t".join(map(str, task_numbers)) + "\n")
            for i, task_i in enumerate(task_numbers):
                row = [f"{distance_matrix[i][j]:.4f}" for j in range(len(task_numbers))]
                f.write(f"{task_i}\t" + "\t".join(row) + "\n")
        
        time_metrics['saving'] = time.time() - phase_start

        print(f"Results saved to {output_file}")
        print("\nTime Breakdown:")
        for phase, t in time_metrics.items():
            print(f"- {phase.capitalize()}: {t:.2f} seconds")
            
        return distance_matrix

    except Exception as e:
        print(f"Error occurred after {time.time() - total_start:.2f}s: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        parent_dir = "data/saved_split_tiny_imagenet"
        output_file = "task_distances.txt"
        
        print(f"Starting computation on {DEVICE}...")
        distance_matrix = compute_pairwise_distances(parent_dir, output_file)
        
        # Print sample output
        print("\nSample distance matrix:")
        if isinstance(distance_matrix, np.ndarray):
            print(distance_matrix[:3,:3])
        else:
            print([row[:3] for row in distance_matrix[:3]])
            
    except KeyboardInterrupt:
        print("\nComputation interrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")