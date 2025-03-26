import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from otdd.pytorch.datasets import load_torchvision_data, load_imagenet
import os
import argparse

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Load training and test data
def load_data(task_num, parent_dir):
    # Load task-specific training and test data
    data_path = f'{parent_dir}/data_task_{task_num}_size_10000.pt'
    labels_path = f'{parent_dir}/labels_task_{task_num}_size_10000.pt'
    
    task_data = torch.load(data_path)  # Load training data
    task_labels = torch.load(labels_path)  # Load labels
    
    # Convert data to a dataset for DataLoader
    dataset = torch.utils.data.TensorDataset(task_data, task_labels)
    
    # Create DataLoader for the training set
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader

# Function to define the ResNet-18 model
def get_model(num_classes):
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)  # Modify the final layer for your dataset
    resnet = resnet.to(DEVICE)
    return resnet

# Training and evaluation for each task
def train_and_evaluate(task_num, parent_dir, datadir):
    print(f"Training for task {task_num}...")
    
    # Load data for this task
    train_loader = load_data(task_num=task_num, parent_dir=parent_dir)
    imagenet = load_imagenet(datadir=datadir)
    test_loader = imagenet[0]["test"]
    
    # Get model
    model = get_model()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Variables to track the best model
    best_accuracy = 0.0
    best_model_wts = model.state_dict()
    best_epoch = 0
    
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Task {task_num} - Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")


    accuracy = evaluate_model(model, test_loader)
    best_accuracy = accuracy
    best_model_wts = model.state_dict()
    best_epoch = epoch
    print(f"Task {task_num} - New best accuracy: {accuracy}%")
    model_save_path = f'{parent_dir}/model_task_{task_num}.pt'
    torch.save(best_model_wts, model_save_path)
    accuracy_save_path = f'{parent_dir}/accuracy_task_{task_num}.txt'
    with open(accuracy_save_path, 'w') as f:
        f.write(f"Best Accuracy: {best_accuracy} in epoch {best_epoch}%")
    
    print(f"Best model and accuracy for Task {task_num} saved!")


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Task {task_num} - Test Accuracy: {accuracy}%")
    return accuracy



def main():
    parser = argparse.ArgumentParser(description='Train models on specific tasks')
    # Existing arguments
    parser.add_argument('--tasks', type=int, nargs='+', default=[9],
                        help='List of task numbers to train (e.g., 7 8 9)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Input batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    
    # New directory arguments
    parser.add_argument('--parent_dir', type=str, default="saved_split_tiny_imagenet",
                        help='Parent directory for task data and models')
    parser.add_argument('--datadir', type=str, 
                        default="data/tiny-ImageNet/tiny-imagenet-200",
                        help='Base directory for dataset')

    args = parser.parse_args()

    parent_dir = args.parent_dir
    datadir = args.datadir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    
    for task_num in args.tasks:
        train_and_evaluate(task_num, parent_dir, datadir)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

# 0 1 2 3 4 5 6 7 8
# CUDA_VISIBLE_DEVICES=2 python3 train_resnet18.py 01
# CUDA_VISIBLE_DEVICES=3 python3 train_resnet18.py 01
# CUDA_VISIBLE_DEVICES=4 python3 train_resnet18.py 23
# CUDA_VISIBLE_DEVICES=5 python3 train_resnet18.py 45
# CUDA_VISIBLE_DEVICES=6 python3 train_resnet18.py 67
# CUDA_VISIBLE_DEVICES=7 python3 train_resnet18.py 89