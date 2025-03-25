import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
from otdd.pytorch.method5 import compute_pairwise_distance
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
from torch.utils.data.sampler import SubsetRandomSampler
import time
from trainer import train, test_func, frozen_module
from models.resnet import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from PIL import Image
import random
from otdd.pytorch.datasets import load_torchvision_data, load_imagenet


class Subset(Dataset):
    def __init__(self, original_dataset, chosen_indices, transform=None):
        self._original_dataset = original_dataset
        self._chosen_indices = chosen_indices
        self.transform = transform
        self.indices = torch.arange(start=0, end=len(self._chosen_indices), step=1)
        self.data = self._original_dataset[self._chosen_indices]
        self.targets = torch.tensor(self._original_dataset.targets)[self._chosen_indices]
        self.classes = sorted(torch.unique(torch.tensor(self._dataset.targets)).tolist())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.transform is None:
            return self.transform(self.data[idx][0]), self.targets[idx]
        else:
            return self.data[idx][0], self.targets[idx]

# (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = "cpu"
    print(f"Use CUDA or not: {DEVICE}")

    # datadir_tiny_imagenet = "data/tiny-ImageNet/tiny-imagenet-200"
    parent_dir = "saved_split_imagenet"
    datadir_tiny_imagenet = "data/imagenet"
    imagenet = load_imagenet(datadir=datadir_tiny_imagenet)

    imagenet_trainset = imagenet[1]["train"]
    imagenet_testset = imagenet[1]["test"]

    imagenet_trainloader = imagenet[0]["train"]
    imagenet_testloader = imagenet[0]["test"]

    num_classes = len(torch.unique(torch.tensor(imagenet_trainset.targets)))

    indices = np.arange(len(imagenet_trainset))

    num_tasks = 10

    for task_id in range(num_tasks):
        all_data_indices_task = list()
        for cls_id in range(num_classes):
            cls_dataset_indices = indices[imagenet_trainset.targets == cls_id]
            # shuffled_indices = np.random.permutation(cls_dataset_indices)
            num_data_of_cls = 10
            cls_data_for_this_task = cls_dataset_indices[num_data_of_cls * task_id: num_data_of_cls * (task_id + 1)]
            all_data_indices_task.extend(cls_data_for_this_task)
        print(task_id, len(all_data_indices_task))
        
        list_chosen_data = list()
        list_chosen_labels = list()
        for idx in all_data_indices_task:
            list_chosen_data.append(imagenet_trainset[idx][0].unsqueeze(0))
            list_chosen_labels.append(torch.tensor(imagenet_trainset[idx][1]).unsqueeze(0))
        list_chosen_data = torch.cat(list_chosen_data, dim=0)
        list_chosen_labels = torch.cat(list_chosen_labels, dim=0)
        print(list_chosen_data.shape, list_chosen_labels.shape)

        os.makedirs(parent_dir, exist_ok=True)
        torch.save(list_chosen_data, f"{parent_dir}/data_task_{task_id}_size_{len(all_data_indices_task)}.pt")  # Save data tensor
        torch.save(list_chosen_labels, f"{parent_dir}/labels_task_{task_id}_size_{len(all_data_indices_task)}.pt")  # Save labels tensor


if __name__ == "__main__":
    main()

