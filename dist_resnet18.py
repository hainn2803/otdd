import torch
import numpy as np
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from otdd.pytorch.method5 import compute_pairwise_distance

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_task_numbers(parent_dir):
    """Get sorted list of task numbers from data files"""
    task_numbers = []
    for fname in os.listdir(parent_dir):
        if fname.startswith("data_task_") and fname.endswith("_size_10000.pt"):
            task_num = int(fname.split("_")[2])
            task_numbers.append(task_num)
    return sorted(task_numbers)

def load_task_data(task_num, parent_dir, sample_size=10000):
    """Load task data with random sampling"""
    data_path = f'{parent_dir}/data_task_{task_num}_size_10000.pt'
    labels_path = f'{parent_dir}/labels_task_{task_num}_size_10000.pt'

    task_data = torch.load(data_path)
    task_labels = torch.load(labels_path)

    # Randomly sample if needed
    if task_data.shape[0] > sample_size:
        indices = np.random.permutation(task_data.shape[0])[:sample_size]
        task_data = task_data[indices]
        task_labels = task_labels[indices]

    return TensorDataset(task_data, task_labels)

def compute_pairwise_distances(parent_dir, output_file="pairwise_distances.txt"):
    """Compute and save pairwise distances between all tasks"""
    # Get all task numbers
    task_numbers = get_task_numbers(parent_dir)
    print(f"Found {len(task_numbers)} tasks: {task_numbers}")

    # Create DataLoaders for all tasks
    dataloaders = []
    for task_num in task_numbers:
        dataset = load_task_data(task_num, parent_dir)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        dataloaders.append(dataloader)

    # Compute pairwise distances
    kwargs = {
        "dimension": 224,
        "num_channels": 3,
        "num_moments": 5,
        "use_conv": True,
        "precision": "float",
        "p": 2,
        "chunk": 10000
    }

    start_time = time.time()
    distance_matrix, total_time = compute_pairwise_distance(
        list_D=dataloaders,
        num_projections=100000,
        device=DEVICE,
        evaluate_time=True,
        **kwargs
    )
    
    # Save results
    with open(output_file, "w") as f:
        # Write header
        f.write("Pairwise Distance Matrix\n")
        f.write(f"Computation Time: {total_time:.2f} seconds\n\n")
        f.write("Tasks\t" + "\t".join(map(str, task_numbers)) + "\n")
        
        # Write matrix
        for i, task_i in enumerate(task_numbers):
            row = [f"{distance_matrix[i][j]:.4f}" for j in range(len(task_numbers))]
            f.write(f"{task_i}\t" + "\t".join(row) + "\n")
    
    print(f"Results saved to {output_file}")
    return distance_matrix

if __name__ == "__main__":
    # Configuration
    parent_dir = "saved_split_tiny_imagenet"
    output_file = "task_distances.txt"
    
    # Compute and save distances
    distance_matrix = compute_pairwise_distances(parent_dir, output_file)
    
    # Print sample output
    print("\nSample distance matrix:")
    print(distance_matrix[:3,:3])  # Show first 3x3 entries