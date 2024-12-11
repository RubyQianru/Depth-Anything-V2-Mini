import torch
import torch.nn.utils.prune as prune
import numpy as np
from torchvision import transforms
import cv2
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2
import os
import h5py

# Define metrics
def calculate_metrics(pred, gt):
    pred = pred.squeeze()
    gt = gt.squeeze()

    mask = gt > 0  # Ignore invalid pixels (e.g., zero-depth values)

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(pred[mask] - gt[mask]))
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2))
    # Threshold Accuracy (Delta < 1.25)
    delta = np.mean((np.maximum(pred[mask] / gt[mask], gt[mask] / pred[mask]) < 1.25).astype(np.float32))

    return mae, rmse, delta

# Load dataset
def load_dataset(file_path, num_images=100, input_size=(224, 224)):

    preprocessed_images = []
    preprocessed_depths = []

    with h5py.File(file_path, 'r') as file:
        images_dataset = file['images']
        depths_dataset = file['depths']

        selected_indices = np.arange(num_images)
        for idx in selected_indices:
            img = images_dataset[idx].transpose(1, 2, 0)  # Adjust axes for RGB format
            depth = depths_dataset[idx]

            # Resize images and depth maps
            img_resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
            depth_resized = cv2.resize(depth, input_size, interpolation=cv2.INTER_NEAREST)

            preprocessed_images.append(img_resized)
            preprocessed_depths.append(depth_resized)

    return preprocessed_images, preprocessed_depths

# Apply pruning
def apply_pruning(model, pruning_percentage=0.5):

    print(f"Applying pruning with {pruning_percentage*100}% sparsity...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_percentage)
            prune.remove(module, 'weight')  # Make pruning permanent
    return model

# Evaluate pruned model
def evaluate_pruned_model(model, images, depths, device='cpu'):

    model.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    predicted_depths = []
    for img in tqdm(images, desc="Generating Depth Predictions"):
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_depth = model(img_tensor)
            pred_depth = pred_depth.squeeze().cpu().numpy()
            pred_depth = np.clip(pred_depth, 0, 10)  # Clip predictions to valid range
            predicted_depths.append(pred_depth)

    # Compute metrics
    mae_list, rmse_list, delta_list = [], [], []
    for pred_depth, gt_depth in zip(predicted_depths, depths):
        mae, rmse, delta = calculate_metrics(pred_depth, gt_depth)
        mae_list.append(mae)
        rmse_list.append(rmse)
        delta_list.append(delta)

    print("\nMetrics for Pruned Model:")
    print(f"MAE: {np.mean(mae_list):.4f}")
    print(f"RMSE: {np.mean(rmse_list):.4f}")
    print(f"Delta (<1.25): {np.mean(delta_list):.4f}")

# Main script
if __name__ == '__main__':
    DEVICE = 'cpu'  # Use 'cuda' if you have a GPU
    encoder = 'vitb'  # model variant option
    pruning_percentage = 0.5  # Prune 50% of weights (adjust to vary the sparsity)
    dataset_path = 'benchmarks/nyu_depth_v2_labeled.mat'  # Path to dataset

    # Load model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**model_configs[encoder])
    checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    # Apply pruning
    model = apply_pruning(model, pruning_percentage=pruning_percentage)

    # Save the pruned model
    pruned_checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}_pruned_{int(pruning_percentage*100)}.pth'
    torch.save(model.state_dict(), pruned_checkpoint_path)
    print(f"Pruned model saved to {pruned_checkpoint_path}")

    # Load dataset
    preprocessed_images, preprocessed_depths = load_dataset(dataset_path)

    # Evaluate pruned model
    evaluate_pruned_model(model, preprocessed_images, preprocessed_depths, device=DEVICE)
