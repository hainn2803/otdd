import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset, random_split
from otdd.pytorch.datasets import load_imagenet
import argparse
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")




def load_data(task_num, parent_dir, batch_size, split_ratio=0.8):
    data_path = f'{parent_dir}/data_task_{task_num}_size_10000.pt'
    labels_path = f'{parent_dir}/labels_task_{task_num}_size_10000.pt'
    task_data = torch.load(data_path)
    task_labels = torch.load(labels_path)
    dataset = TensorDataset(task_data, task_labels)
    total_size = len(dataset)
    train_size = int(total_size * split_ratio)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )
    return train_loader, test_loader


def get_model(num_classes=200):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE).train()

def load_pretrained_model(model, checkpoint_path):
    """
    Load pretrained model weights from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded pretrained model from {checkpoint_path}")
    return model

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Saved checkpoint to {filename}")

def train_and_evaluate(source_task, task_num, parent_dir, datadir, batch_size, num_epochs, learning_rate, 
                      checkpoint_freq=1, resume_from=None, pretrained_model_path=None):
    print(f"\n{'='*50}\nTraining Task {task_num}\n{'='*50}")
    
    checkpoint_dir = os.path.join(parent_dir, "finetune_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_loader, test_loader = load_data(task_num, parent_dir, batch_size)
    
    model = get_model()
    
    if pretrained_model_path:
        model = load_pretrained_model(model, pretrained_model_path)
        print(f"Fine-tuning on Task {task_num} with pretrained weights.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch = 0
    best_loss = float('inf')
    best_model_wts = None
    train_history = []

    # Resume training if specified
    if resume_from and os.path.isfile(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_model_wts = checkpoint['best_model_wts']
        train_history = checkpoint['train_history']
        print(f"Resumed training from epoch {start_epoch} with best loss {best_loss:.4f}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        train_history.append(epoch_loss)
        
        # Update best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.state_dict()
            best_scaler = scaler.state_dict()
            best_optimizer = optimizer.state_dict()
            best_epoch = epoch

            print(f"New best loss: {best_loss:.4f}")
        
        print(f"Task {task_num} | Epoch {epoch+1:02d}/{num_epochs} | Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        if checkpoint_freq > 0:
            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_task_dir = os.path.join(checkpoint_dir, f"source_{source_task}")
                os.makedirs(checkpoint_task_dir, exist_ok=True)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'loss': epoch_loss,
                }, os.path.join(checkpoint_task_dir, f'target_{task_num}_epoch_{epoch+1}_ckpt.pt'))

    # Save final models
    checkpoint_task_dir = os.path.join(checkpoint_dir, f"source_{source_task}")
    os.makedirs(checkpoint_task_dir, exist_ok=True)
    final_model_path = os.path.join(checkpoint_task_dir, f'target_{task_num}_epoch_{epoch+1}_last.pt')
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'loss': best_loss,
    }, final_model_path)

    
    # Save best model
    if best_model_wts is not None:
        best_model_path = os.path.join(checkpoint_task_dir, f'target_{task_num}_epoch_{epoch+1}_best.pt')
        save_checkpoint({
            'epoch': best_epoch,
            'state_dict': best_model_wts,
            'optimizer': best_optimizer,
            'scaler': best_scaler,
            'loss': best_loss,
        }, best_model_path)
    
    # Evaluate both models
    print("\nEvaluating models...")
    # Load final model
    final_model = get_model()
    final_model.load_state_dict(torch.load(final_model_path)["state_dict"])
    final_model = final_model.to(DEVICE).eval()

    # Load best model
    best_model = get_model()
    best_model.load_state_dict(torch.load(best_model_path)["state_dict"])
    best_model = best_model.to(DEVICE).eval()
    
    final_accuracy_test = evaluate_model(final_model, test_loader)
    best_accuracy_test = evaluate_model(best_model, test_loader)
    final_accuracy_train = evaluate_model(final_model, train_loader)
    best_accuracy_train = evaluate_model(best_model, train_loader)
    
    print(f"\nTask {task_num} Evaluation Results:")
    print(f"Final Model Test Accuracy: {final_accuracy_test:.2f}%")
    print(f"Best Loss Model Test Accuracy: {best_accuracy_test:.2f}%")
    print(f"Final Model Train Accuracy: {final_accuracy_train:.2f}%")
    print(f"Best Loss Train Model Accuracy: {best_accuracy_train:.2f}%")
    
    # Save results
    with open(f'{checkpoint_task_dir}/results_source_{source_task}_target_{task_num}.txt', 'w') as f:
        f.write(f"Final Model Train Accuracy: {final_accuracy_train:.2f}%\n")
        f.write(f"Best Loss Model Train Accuracy: {best_accuracy_train:.2f}%\n")
        f.write(f"Best Training Loss: {best_loss:.4f}\n")
        f.write(f"Final Training Loss: {train_history[-1]:.4f}\n")
        f.write(f"Final Model Test Accuracy: {final_accuracy_test:.2f}%\n")
        f.write(f"Best Loss Model Test Accuracy: {best_accuracy_test:.2f}%\n")

def evaluate_model(model, test_loader):
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    parser = argparse.ArgumentParser(description='ImageNet Training with Dual Model Saving')
    parser.add_argument('--source_task', type=int, nargs='+', default=[0], help="List of source tasks for pretraining (0-9)")
    parser.add_argument('--target_tasks', type=int, nargs='+', default=None, help="Target tasks for fine-tuning (0-9)")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--parent_dir', type=str, default="saved_split_tiny_imagenet")
    parser.add_argument('--datadir', type=str, default="data/tiny-ImageNet/tiny-imagenet-200")
    parser.add_argument('--checkpoint_freq', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()

    for source_id in args.source_task:
        source_task_checkpoint = os.path.join(args.parent_dir, "checkpoints", f"task_{source_id}/task_{source_id}_epoch_100_best.pt")

        if args.target_tasks is None:
            target_tasks = [i for i in range(10) if i != source_id]

        for target_task in target_tasks:
            print(f"Pretraining on {source_id}, Fine-tuning on target task {target_task}...")
            train_and_evaluate(
                source_task = source_id,
                task_num=target_task,
                parent_dir=args.parent_dir,
                datadir=args.datadir,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                checkpoint_freq=args.checkpoint_freq,
                resume_from=args.resume,
                pretrained_model_path=source_task_checkpoint
            )

if __name__ == "__main__":
    main()

