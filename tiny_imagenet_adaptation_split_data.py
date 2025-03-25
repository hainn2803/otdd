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
    def __init__(self, dataset, original_indices, transform=None):
        self._dataset = dataset
        self._original_indices = original_indices
        self.transform = transform
        self.indices = torch.arange(start=0, end=len(self._original_indices), step=1)
        self.data = self._dataset.data[self._original_indices]
        self.targets = torch.tensor(self._dataset.targets)[self._original_indices]
        self.classes = sorted(torch.unique(torch.tensor(self._dataset.targets)).tolist())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.transform is None:
            return self.transform(self.data[idx]), self.targets[idx]
        else:
            return self.data[idx], self.targets[idx]

# (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = "cpu"
    print(f"Use CUDA or not: {DEVICE}")

    datadir_tiny_imagenet = "data/tiny-ImageNet/tiny-imagenet-200"
    # datadir_tiny_imagenet = "data/imagenet"
    imagenet = load_imagenet(datadir=datadir_tiny_imagenet)

    imagenet_trainset = imagenet[1]["train"]
    imagenet_testset = imagenet[1]["test"]

    imagenet_trainloader = imagenet[0]["train"]
    imagenet_testloader = imagenet[0]["test"]

    num_classes = len(torch.unique(torch.tensor(imagenet_trainset.targets)))

    indices = np.arange(len(imagenet_trainset))

    num_tasks = 10

    list_data = list()
    list_labels = list()
    for data, label in imagenet_trainloader:
        print(data.shape, label.shape)
        list_data.append(data)
        list_labels.append(label)
    
    all_data = torch.cat(list_data, dim=0)
    all_labels = torch.cat(list_labels, dim=0)
    print(all_data.shape, all_labels.shape)
    torch.save(all_data, 'all_tiny-imagenet_data.pt')  # Save data tensor
    torch.save(all_labels, 'all_tiny-imagenet_labels.pt')  # Save labels tensor

    print('Data and labels have been saved!')


    def save_data(data_set, saved_tensor_path):
        list_images = list()
        list_labels = list()
        for img, label in data_set:
            list_images.append(img)
            list_labels.append(label)
        tensor_images = torch.stack(list_images)
        tensor_labels = torch.tensor(list_labels)
        torch.save((tensor_images, tensor_labels), saved_tensor_path)
        print(f"Number of data: {len(list_images)}, Save data into {saved_tensor_path}")

    # for task_id in range(num_tasks):
    #     all_data_indices_task = list()
    #     for cls_id in range(num_classes):
    #         cls_dataset_indices = indices[imagenet_trainset.targets == cls_id]
    #         shuffled_indices = np.random.permutation(cls_dataset_indices)
    #         num_data_of_cls = 50
    #         cls_data_for_this_task = shuffled_indices[:num_data_of_cls]
    #         all_data_indices_task.extend(cls_data_for_this_task)
    #     print(task_id, len(all_data_indices_task))
        
    #     sub = Subset(dataset=imagenet_trainset, original_indices=all_data_indices_task)
    #     save_data(data_set=sub, saved_tensor_path=f"saved_split_imagenet/task_{task_id}_size_{len(all_data_indices_task)}.pt")


if __name__ == "__main__":
    main()

