import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from datasets import load_dataset
import wandb
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Custom Dataset implementation
class STL10SubsetDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
   
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    # ResNet-18 expects 224x224 images, normalized via ImageNet stats
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="STL-10 ResNet-18 Training Pipeline")
    parser.add_argument("--hf_repo_id", type=str, default="diwanshuydv/mlops_minor", help="Hugging Face model repo ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # Initialize weights and biases
    wandb.init(project="stl10-resnet18-assignment", config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1 & 2. Load dataset and create DataLoaders
    print("Loading dataset...")
    # Using 'train' and 'test' splits if available. We will split train into train/val if needed, 
    # or just use test as val for simplicity if it's a small subset.
    dataset = load_dataset("Chiranjeev007/STL-10_Subset")
    
    # Check what splits are available
    print("Available splits:", dataset.keys())
    
    # Assuming 'train' and 'test' exist. Let's create datasets.
    train_transform, val_transform = get_transforms()
    
    # Extract labels to know number of classes. STL-10 has 10 classes.
    num_classes = 10
    class_names = [f"Class_{i}" for i in range(num_classes)] # Fallback names if not in dataset
    if 'train' in dataset and hasattr(dataset['train'].features['label'], 'names'):
        class_names = dataset['train'].features['label'].names
    
    train_dataset = STL10SubsetDataset(dataset['train'], transform=train_transform)
    val_dataset = STL10SubsetDataset(dataset['test'], transform=val_transform) # Using test as val during training
    test_dataset = STL10SubsetDataset(dataset['test'], transform=val_transform) # Same for test
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 3. Load ResNet-18 and adapt for num_classes
    print("Initializing ResNet-18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Training Loop and WandB Logging
    best_val_acc = 0.0
    best_model_path = "best_resnet18_stl10.pth"
    
    print("Starting training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved new best model with Val Acc: {best_val_acc:.4f}")
            
    # 5. Push best model to Hugging Face
    print(f"Pushing model to Hugging Face Hub: {args.hf_repo_id}")
    try:
        api = HfApi()
        # Create repo if it doesn't exist
        api.create_repo(repo_id=args.hf_repo_id, exist_ok=True)
        api.upload_file(
            path_or_fileobj=best_model_path,
            path_in_repo="pytorch_model.bin",
            repo_id=args.hf_repo_id
        )
        print("Successfully pushed to HF.")
    except Exception as e:
        print(f"Failed to push to huggingface: {e}")
        print("Continuing with local evaluation...")
        
    # 6. Load model from Hugging Face for evaluation steps
    print("Downloading model from Hugging Face Hub for evaluation...")
    eval_model = models.resnet18(weights=None)
    eval_model.fc = nn.Linear(num_ftrs, num_classes)
    
    try:
        downloaded_model_path = hf_hub_download(repo_id=args.hf_repo_id, filename="pytorch_model.bin")
        eval_model.load_state_dict(torch.load(downloaded_model_path, map_location=device))
        print("Loaded model from HF Hub.")
    except Exception as e:
        print(f"Could not download from HF: {e}. Falling back to local best model.")
        eval_model.load_state_dict(torch.load(best_model_path, map_location=device))
        
    eval_model = eval_model.to(device)
    
    # Run evaluation on test set
    print("Running final evaluation on test set...")
    _, test_acc, test_preds, test_labels = evaluate(eval_model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 7. Confusion Matrix
    print("Generating Confusion Matrix...")
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_labels, 
            preds=test_preds,
            class_names=class_names
        )
    })
    
    # 8. Class-wise accuracy bar plot
    print("Generating Class-wise accuracy plot...")
    report = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
    # Extract just class accuracies (f1-score is often used, but we can compute exact accuracy from conf matrix)
    cm = confusion_matrix(test_labels, test_preds)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    data = [[class_names[i], acc] for i, acc in enumerate(class_accuracies)]
    table = wandb.Table(data=data, columns=["Class", "Accuracy"])
    wandb.log({"class_accuracy": wandb.plot.bar(table, "Class", "Accuracy", title="Class-wise Accuracy")})
    
    # 9. Log 20 examples with image, predicted, and actual
    print("Logging 20 examples to WandB...")
    # We need the raw images, not normalized tensors natively, so let's get them from dataset
    indices = random.sample(range(len(dataset['test'])), min(20, len(dataset['test'])))
    
    example_data = []
    
    eval_model.eval()
    with torch.no_grad():
        for idx in indices:
            item = dataset['test'][idx]
            raw_image = item['image']
            if raw_image.mode != 'RGB':
                raw_image = raw_image.convert('RGB')
            actual_label_idx = item['label']
            actual_label_str = class_names[actual_label_idx]
            
            # transform for model
            tensor_img = val_transform(raw_image).unsqueeze(0).to(device)
            out = eval_model(tensor_img)
            _, pred_idx = out.max(1)
            pred_idx = pred_idx.item()
            pred_label_str = class_names[pred_idx]
            
            example_data.append([
                wandb.Image(raw_image), 
                pred_label_str, 
                actual_label_str
            ])
            
    examples_table = wandb.Table(data=example_data, columns=["Image", "Predicted", "Actual"])
    wandb.log({"test_examples": examples_table})
    
    print("Done!")
    wandb.finish()

if __name__ == "__main__":
    main()