import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from otdd.pytorch.datasets import load_torchvision_data, load_imagenet
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parent_dir = "saved_split_imagenet"
# datadir = "data/tiny-ImageNet/tiny-imagenet-200"
datadir = "data/imagenet"


def load_data(task_num):
    data_path = f'{parent_dir}/data_task_{task_num}_size_10000.pt'
    labels_path = f'{parent_dir}/labels_task_{task_num}_size_10000.pt'
    
    task_data = torch.load(data_path)
    task_labels = torch.load(labels_path)
    dataset = torch.utils.data.TensorDataset(task_data, task_labels)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    return train_loader

train_loader_task_0 = load_data(0)
train_loader_task_1 = load_data(1)
dataloaders = [train_loader_task_0, train_loader_task_1]
# sOTDD
kwargs = {
    "dimension": 224,
    "num_channels": 3,
    "num_moments": 5,
    "use_conv": True,
    "precision": "float",
    "p": 2,
    "chunk": 1000
}
list_pairwise_dist, sotdd_time_taken = compute_pairwise_distance(list_D=dataloaders, num_projections=100000, device=DEVICE, evaluate_time=True, **kwargs)
sotdd_dist = list_pairwise_dist[0]
total_processing_time += sotdd_time_taken
print(f"sOTDD distance: {sotdd_dist}, time taken: {sotdd_time_taken}")