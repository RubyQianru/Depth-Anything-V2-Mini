import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm

# -----------------------------
# Dataset Class for NYU Depth V2
# -----------------------------
class NYUDepthDataset(Dataset):
    def __init__(self, images, depths, input_size=(224, 224)):
        self.images = images
        self.depths = depths
        self.input_size = input_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
        depth = self.depths[idx]

        # Resize
        img_resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth, self.input_size, interpolation=cv2.INTER_NEAREST)

        # Normalize image
        img_resized = img_resized / 255.0  # Scale to [0, 1]
        img_tensor = torch.tensor(img_resized).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # Convert depth to tensor
        depth_tensor = torch.tensor(depth_resized, dtype=torch.float32)

        return img_tensor, depth_tensor


# -----------------------------
# Load Small Subset of NYU Depth V2 Dataset
# -----------------------------
with h5py.File('benchmarks/nyu_depth_v2_labeled.mat', 'r') as file:
    images = file['images'][:500]  # Use only 500 samples for training
    depths = file['depths'][:500]

    val_images = file['images'][500:600]  # Use 100 samples for validation
    val_depths = file['depths'][500:600]

# Create Dataset and DataLoader
train_dataset = NYUDepthDataset(images, depths)
val_dataset = NYUDepthDataset(val_images, val_depths)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# -----------------------------
# Model Configurations
# -----------------------------
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Loop Through All Models
# -----------------------------
for model_name, config in model_configs.items():
    print(f"\n=== Fine-Tuning {model_name.upper()} Model ===")

    # Initialize model
    model = DepthAnythingV2(**config)
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{model_name}.pth', map_location=DEVICE))
    model = model.to(DEVICE)

    # Freeze the backbone (transformer encoder)
    for name, param in model.named_parameters():
        if "pretrained" in name:  # Freeze all transformer backbone layers
            param.requires_grad = False

    # Verify which parameters are trainable
    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")

    # Define loss function and optimizer
    criterion = nn.L1Loss()  # L1 loss for depth regression
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # -----------------------------
    # Fine-Tuning Loop (Train Head Only)
    # -----------------------------
    num_epochs = 5  # Fewer epochs for head-only training

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, depths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - {model_name.upper()} - Training"):
            imgs = imgs.to(DEVICE, dtype=torch.float32)
            depths = depths.to(DEVICE, dtype=torch.float32)

            # Forward pass
            preds = model(imgs)
            preds = preds.squeeze(1)  # Remove channel dimension if necessary

            # Calculate loss
            loss = criterion(preds, depths)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss / len(train_loader):.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, depths in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - {model_name.upper()} - Validation"):
                imgs = imgs.to(DEVICE, dtype=torch.float32)
                depths = depths.to(DEVICE, dtype=torch.float32)

                preds = model(imgs)
                preds = preds.squeeze(1)  # Remove channel dimension if necessary

                # Calculate loss
                loss = criterion(preds, depths)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader):.4f}")

    # -----------------------------
    # Save the Fine-Tuned Model
    # -----------------------------
    torch.save(model.state_dict(), f'fine_tuned_{model_name}_nyu_v2_head.pth')
    print(f"Model saved as 'fine_tuned_{model_name}_nyu_v2_head.pth'")
