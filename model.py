import torch
from torch.utils.data import DataLoader

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for images, depths in train_loader:
        images, depths = images.to(device), depths.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.predicted_depth, depths)
        loss.backward()
        optimizer.step()