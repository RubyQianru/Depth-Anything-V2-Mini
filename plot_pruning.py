import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import h5py
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2
import os

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
def load_dataset(file_path, num_images=200, input_size=(224, 224)):
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

# Evaluate pruned model
def evaluate_pruned_model(checkpoint_path, model_config, images, depths, device='cpu'):

    model = DepthAnythingV2(**model_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    predicted_depths = []
    for img in tqdm(images, desc=f"Evaluating {os.path.basename(checkpoint_path)}"):
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_depth = model(img_tensor)
            pred_depth = pred_depth.squeeze().cpu().numpy()
            pred_depth = np.clip(pred_depth, 0, 10)  # Clip predictions to valid range
            predicted_depths.append(pred_depth)

    mae_list, rmse_list, delta_list = [], [], []
    for pred_depth, gt_depth in zip(predicted_depths, depths):
        mae, rmse, delta = calculate_metrics(pred_depth, gt_depth)
        mae_list.append(mae)
        rmse_list.append(rmse)
        delta_list.append(delta)

    return np.mean(mae_list), np.mean(rmse_list), np.mean(delta_list)

# Main script
if __name__ == '__main__':
    DEVICE = 'cpu'  # Use 'cuda' if you have a GPU
    encoder = 'vits'  # Update as needed
    dataset_path = 'benchmarks/nyu_depth_v2_labeled.mat'  # Path to dataset

    # Model configuration for the specific vits encoder
    model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}

    # Sparsity levels and corresponding checkpoint paths
    sparsity_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # Sparsity percentages
    checkpoint_paths = [
        f'checkpoints/depth_anything_v2_{encoder}_pruned_{sparsity}.pth' for sparsity in sparsity_levels
    ]

    # Load dataset
    preprocessed_images, preprocessed_depths = load_dataset(dataset_path)

    # Collect metrics for each pruned model
    mae_results, rmse_results, delta_results = [], [], []
    for checkpoint in checkpoint_paths:
        if not os.path.exists(checkpoint):
            print(f"Checkpoint not found: {checkpoint}")
            continue

        mae, rmse, delta = evaluate_pruned_model(
            checkpoint, model_config, preprocessed_images, preprocessed_depths, device=DEVICE
        )
        mae_results.append(mae)
        rmse_results.append(rmse)
        delta_results.append(delta)

    # Plot MAE and RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(sparsity_levels[:len(mae_results)], mae_results, label='MAE', marker='o')
    plt.plot(sparsity_levels[:len(rmse_results)], rmse_results, label='RMSE', marker='o')
    plt.xlabel('Sparsity Level (%)', fontsize=14)
    plt.ylabel('Error Metrics', fontsize=14)
    plt.title('MAE and RMSE vs. Sparsity Level', fontsize=16)
    plt.xticks(fontsize=12)  # Axis tick font size
    plt.yticks(fontsize=12)  # Axis tick font size
    plt.legend(fontsize=12, loc='upper right')  # Legend font size and location
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Accuracy (Delta < 1.25)
    delta_results_percent = [delta * 100 for delta in delta_results]  # Scale to percentage
    plt.figure(figsize=(10, 6))
    plt.plot(sparsity_levels[:len(delta_results)], delta_results_percent, label='Delta (<1.25)', marker='o', color='green')
    plt.xlabel('Sparsity Level (%)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Accuracy vs. Sparsity Level', fontsize=16)
    plt.xticks(fontsize=12)  # Axis tick font size
    plt.yticks(fontsize=12)  # Axis tick font size
    plt.legend(fontsize=12, loc='upper right')  # Legend font size and location
    plt.grid(True)
    plt.tight_layout()
    plt.show()
