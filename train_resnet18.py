import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from otdd.pytorch.datasets import load_torchvision_data, load_imagenet
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 256
num_epochs = 10
learning_rate = 0.001


parent_dir = "saved_split_imagenet"
# datadir = "data/tiny-ImageNet/tiny-imagenet-200"
datadir = "data/imagenet"

# Load training and test data
def load_data(task_num):
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
def get_model():
    resnet = models.resnet18(pretrained=False)
    # resnet18.fc = nn.Linear(resnet18.fc.in_features, 200)  # Modify the final layer for your dataset
    resnet = resnet.to(DEVICE)
    return resnet

# Training and evaluation for each task
def train_and_evaluate(task_num):
    print(f"Training for task {task_num}...")
    
    # Load data for this task
    train_loader = load_data(task_num)
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
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_wts = model.state_dict()
            best_epoch = epoch
            print(f"Task {task_num} - New best accuracy: {accuracy}%")
    
            model_save_path = f'saved_split_tiny_imagenet/model_task_{task_num}.pt'
            torch.save(best_model_wts, model_save_path)
            
            accuracy_save_path = f'saved_split_tiny_imagenet/accuracy_task_{task_num}.txt'
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

list_task = [8]
for task_num in list_task:
    train_and_evaluate(task_num)

# 0 1 2 3 4 5 6 7 8
# CUDA_VISIBLE_DEVICES=2 python3 train_resnet18.py 01
# CUDA_VISIBLE_DEVICES=3 python3 train_resnet18.py 01
# CUDA_VISIBLE_DEVICES=4 python3 train_resnet18.py 23
# CUDA_VISIBLE_DEVICES=5 python3 train_resnet18.py 45
# CUDA_VISIBLE_DEVICES=6 python3 train_resnet18.py 67
# CUDA_VISIBLE_DEVICES=7 python3 train_resnet18.py 89