import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tqdm
# ---------------------------------------------------------
# 1. Dataset & DataLoader
# ---------------------------------------------------------
class SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transforms.Compose([
            transforms.ToTensor() # Converts to [0, 1] and (C, H, W)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read RGB image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        
        # Read Mask (Keep as absolute class integer values, no scaling)
        mask = Image.open(self.mask_paths[idx])
        mask = torch.tensor(np.array(mask)[:, :, 0], dtype=torch.long)
        
        return img, mask

# Fetch file paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_files = sorted(glob.glob(os.path.join(base_dir, 'CameraRGB/*.png')))
mask_files = sorted(glob.glob(os.path.join(base_dir, 'CameraMask/*.png')))

# 80-20 Train-Test split with seed 42
train_imgs, test_imgs, train_masks, test_masks = train_test_split(
    image_files, mask_files, test_size=0.2, random_state=42
)

# Create DataLoaders
train_dataset = SegDataset(train_imgs, train_masks)
test_dataset = SegDataset(test_imgs, test_masks)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ---------------------------------------------------------
# 2. UNet Model Architecture
# ---------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        # Interpolate in case dimensions mismatch due to pooling
        if x.shape != x1.shape:
            x = nn.functional.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up1(x)
        return self.outc(x)

# ---------------------------------------------------------
# 3. Metrics Calculation (mIoU and mDice)
# ---------------------------------------------------------
def compute_metrics(pred, target, num_classes=23):
    pred = torch.argmax(pred, dim=1)
    iou_list = []
    dice_list = []
    
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        
        intersection = (pred_inds & target_inds).sum().float().item()
        union = (pred_inds | target_inds).sum().float().item()
        
        if union == 0:
            # Ignore classes that are not present in both prediction and target
            continue
            
        iou = intersection / union
        dice = (2. * intersection) / (pred_inds.sum().float().item() + target_inds.sum().float().item())
        
        iou_list.append(iou)
        dice_list.append(dice)
        
    return np.mean(iou_list) if iou_list else 0.0, np.mean(dice_list) if dice_list else 0.0

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleUNet(num_classes=23).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 15
history = {'loss': [], 'miou': [], 'mdice': []}

print(f"Starting training on device: {device}")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_miou = 0
    epoch_mdice = 0
    
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Calculate metrics
        miou, mdice = compute_metrics(outputs, masks)
        epoch_miou += miou
        epoch_mdice += mdice

    avg_loss = epoch_loss / len(train_loader)
    avg_miou = epoch_miou / len(train_loader)
    avg_mdice = epoch_mdice / len(train_loader)
    
    history['loss'].append(avg_loss)
    history['miou'].append(avg_miou)
    history['mdice'].append(avg_mdice)
    
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - mIoU: {avg_miou:.4f} - mDice: {avg_mdice:.4f}")

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history['loss'], label='Training Loss', color='red', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('loss_curve.png')
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history['miou'], label='mIoU Score', color='blue', marker='s')
plt.xlabel('Epochs')
plt.ylabel('mIoU')
plt.title('Training mIoU Score Curve')
plt.legend()
plt.savefig('miou_curve.png')
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history['mdice'], label='mDice Score', color='green', marker='^')
plt.xlabel('Epochs')
plt.ylabel('mDice')
plt.title('Training mDice Score Curve')
plt.legend()
plt.savefig('mdice_curve.png')
plt.close()

SAVE_PATH = "unet_segmentation_weights.pth"
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model successfully saved to {SAVE_PATH}")
print("Training complete. Plots saved locally.")