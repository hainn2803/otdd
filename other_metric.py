import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torchvision import datasets, transforms
from collections import defaultdict
import torch.nn as nn
from trainer import FeatureExtractor, FullyConnectedNetwork

import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data
import otdd.pytorch.method5 as method5
import otdd.pytorch.method_linear_gaussian as method_linear_gaussian
from otdd.pytorch.distance import DatasetDistance

from otdd.pytorch.method_gaussian import load_full_dataset
from otdd.pytorch.moments import compute_label_stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def create_dataset(maxsamples=None, maxsize_for_each_class=None):

    LIST_DATASETS = ["MNIST", "FashionMNIST", "EMNIST", "KMNIST", "USPS"]
    METADATA_DATASET = dict()
    for dataset_name in LIST_DATASETS:

        METADATA_DATASET[dataset_name] = dict()

        if dataset_name == "USPS":
            data_folders = load_torchvision_data(dataset_name, valid_size=0, resize=28, download=False, maxsize=maxsamples, datadir="data/USPS", maxsize_for_each_class=maxsize_for_each_class)
        else:
            data_folders = load_torchvision_data(dataset_name, valid_size=0, resize=28, download=False, maxsize=maxsamples, maxsize_for_each_class=maxsize_for_each_class)

        METADATA_DATASET[dataset_name]["train_loader"] = data_folders[0]['train']
        METADATA_DATASET[dataset_name]["test_loader"] = data_folders[0]['test']
        METADATA_DATASET[dataset_name]["train_set"] = data_folders[1]['train']
        METADATA_DATASET[dataset_name]["test_set"] = data_folders[1]['test']

        if dataset_name == "MNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        elif dataset_name == "KMNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        elif dataset_name == "EMNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 26
        elif dataset_name == "FashionMNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        elif dataset_name == "USPS":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        else:
            raise("Unknown src dataset")
    
    return METADATA_DATASET



METADATA_DATASET = create_dataset(maxsamples=None)


for cac in range(1, 5):

    root_path = f"saved_nist/nist{cac}/pretrained_weights"

    for target_dataset in os.listdir(root_path):
        print(target_dataset)
        target_path = os.path.join(root_path, target_dataset)

        if not os.path.isdir(target_path):
            continue

        model_path = target_path
        # for source_dataset in os.listdir(target_path):
        #     model_path = os.path.join(target_path, source_dataset)
        extractor_path = os.path.join(model_path, "extractor_layers.pth")
        fc_path = os.path.join(model_path, "fc_layers.pth")

        if not os.path.exists(extractor_path) or not os.path.exists(fc_path):
            continue

        # print(f"Evaluating: {source_dataset} → {target_dataset}")

        ft_extractor = FeatureExtractor(input_size=28).to(DEVICE)
        ft_extractor.load_state_dict(torch.load(extractor_path))

        classifier = FullyConnectedNetwork(feat_dim=ft_extractor.feat_dim, num_classes=METADATA_DATASET[target_dataset]["num_classes"]).to(DEVICE)
        classifier.load_state_dict(torch.load(fc_path))

        ft_extractor = ft_extractor.eval()
        classifier = classifier.eval()

        testloader = METADATA_DATASET[target_dataset]["test_loader"]

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in testloader:

                feats = ft_extractor(images.to(DEVICE))
                feats = feats.view(feats.shape[0], -1)
                outputs = classifier(feats)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        precision = precision_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")
        f1 = f1_score(all_labels, all_preds, average="macro")
        accuracy = accuracy_score(all_labels, all_preds)

        with open(model_path+"/accuracy.txt", "a") as f:
            f.write(f"accuracy: {accuracy} \n")
            f.write(f"precision: {precision} \n")
            f.write(f"recall: {recall} \n")
            f.write(f"f1: {f1} \n")