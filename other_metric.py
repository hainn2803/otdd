import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision import datasets, transforms
from collections import defaultdict
import torch.nn as nn



root_path = "saved_nist/nist1/finetune_weights"


def create_dataset(maxsamples=MAXSIZE_DIST, maxsize_for_each_class=None):

    METADATA_DATASET = dict()
    for dataset_name in LIST_DATASETS:

        METADATA_DATASET[dataset_name] = dict()

        if dataset_name == "USPS":
            data_folders = load_torchvision_data(dataset_name, valid_size=0, resize=28, download=False, maxsize=maxsamples, datadir="data/USPS", maxsize_for_each_class=maxsize_for_each_class)
        else:
            data_folders = load_torchvision_data(dataset_name, valid_size=0, resize=28, download=False, maxsize=maxsamples, maxsize_for_each_class=maxsize_for_each_class)

        METADATA_DATASET[dataset_name]["train_loader"] = data_folders[0]['train']
        METADATA_DATASET[dataset_name]["train_set"] = data_folders[1]['train']
        METADATA_DATASET[dataset_name]["pretrained_extractor_path"] = f'{pretrained_path}/{dataset_name}/extractor_layers.pth'
        METADATA_DATASET[dataset_name]["pretrained_classifier_path"] = f'{pretrained_path}/{dataset_name}/fc_layers.pth'

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



for source_dataset in os.listdir(root_path):
    source_path = os.path.join(root_path, source_dataset)
    if not os.path.isdir(source_path):
        continue

    for target_dataset in os.listdir(source_path):
        model_path = os.path.join(source_path, target_dataset)
        extractor_path = os.path.join(model_path, "extractor_layers.pth")
        fc_path = os.path.join(model_path, "fc_layers.pth")

        if not os.path.exists(extractor_path) or not os.path.exists(fc_path):
            continue

        print(f"Evaluating: {source_dataset} → {target_dataset}")

        # Load model
        extractor = torch.load(extractor_path).eval()
        classifier = torch.load(fc_path).eval()

        # Load test set for the target domain
        try:
            testset = load_test_dataset(target_dataset, transform)
        except ValueError as e:
            print(e)
            continue

        testloader = DataLoader(testset, batch_size=64, shuffle=False)

        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        results[f"{source_dataset}→{target_dataset}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

# Print results
print("\n--- Evaluation Results ---")
for k, v in results.items():
    print(f"{k:25s} | Precision: {v['precision']:.3f}, Recall: {v['recall']:.3f}, F1: {v['f1']:.3f}")