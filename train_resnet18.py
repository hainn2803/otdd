import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from otdd.pytorch.datasets import load_imagenet
import argparse
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def load_data(task_num, parent_dir, batch_size):
    data_path = f'{parent_dir}/data_task_{task_num}_size_10000.pt'
    labels_path = f'{parent_dir}/labels_task_{task_num}_size_10000.pt'
    
    task_data = torch.load(data_path)
    task_labels = torch.load(labels_path)
    
    dataset = TensorDataset(task_data, task_labels)
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )
    return train_loader

def get_model(num_classes=200):
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet.to(DEVICE).train()

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Saved checkpoint to {filename}")

def train_and_evaluate(task_num, parent_dir, datadir, batch_size, num_epochs, learning_rate, 
                      checkpoint_freq=1, resume_from=None):
    print(f"\n{'='*50}\nTraining Task {task_num}\n{'='*50}")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(parent_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    train_loader = load_data(task_num, parent_dir, batch_size)
    imagenet = load_imagenet(datadir=datadir)
    test_loader = imagenet[0]["test"]
    
    # Initialize model and optimizer
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    # Resume training if specified
    start_epoch = 0
    accuracy = 0.0
    if resume_from:
        if os.path.isfile(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1
            accuracy = checkpoint['accuracy']
            print(f"Resumed training from epoch {start_epoch} with accuracy {accuracy:.2f}%")
        else:
            print(f"Warning: Checkpoint {resume_from} not found! Starting from scratch.")

    # Training loop
    for epoch in range(start_epoch, num_epochs + start_epoch):
        model.train()
        epoch_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Scaled backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        # Evaluation
        epoch_loss /= len(train_loader)        
        print(f"Task {task_num} | Epoch {epoch+1:02d}/{num_epochs} |  Loss: {epoch_loss:.4f}")
        
        
        # Save checkpoint
        if ((checkpoint_freq == 0) and (epoch + 1 == num_epochs + start_epoch)) or ((checkpoint_freq > 0) and ((epoch + 1) % checkpoint_freq == 0)):
            accuracy = evaluate_model(model, test_loader, task_num)
            checkpoint_path = os.path.join(checkpoint_dir, f'task_{task_num}_epoch_{epoch+1}.pt')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'accuracy': accuracy,
                'loss': epoch_loss,
            }, checkpoint_path)

    print(f"Accuracy for Task {task_num}: {accuracy:.2f}%")

def evaluate_model(model, test_loader, task_num):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Multi-task ImageNet Training with Checkpoints')
    # Existing arguments
    parser.add_argument('--tasks', type=int, nargs='+', default=[0],
                        help='Task numbers to train (e.g., 7 8 9)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--parent_dir', type=str, default="saved_split_tiny_imagenet")
    parser.add_argument('--datadir', type=str, default="data/tiny-ImageNet/tiny-imagenet-200")
    
    # New checkpoint arguments
    parser.add_argument('--checkpoint_freq', type=int, default=0,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    for task_num in args.tasks:
        train_and_evaluate(
            task_num=task_num,
            parent_dir=args.parent_dir,
            datadir=args.datadir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            checkpoint_freq=args.checkpoint_freq,
            resume_from=args.resume
        )

if __name__ == "__main__":
    main()

# 0 1 2 3 4 5 6 7 8
# CUDA_VISIBLE_DEVICES=2 python3 train_resnet18.py 01
# CUDA_VISIBLE_DEVICES=3 python3 train_resnet18.py 01
# CUDA_VISIBLE_DEVICES=4 python3 train_resnet18.py 23
# CUDA_VISIBLE_DEVICES=5 python3 train_resnet18.py 45
# CUDA_VISIBLE_DEVICES=6 python3 train_resnet18.py 67
# CUDA_VISIBLE_DEVICES=7 python3 train_resnet18.py 89