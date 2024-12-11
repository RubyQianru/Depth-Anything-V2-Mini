import torch
import numpy as np
from torchvision import transforms
import cv2
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2
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
with h5py.File('benchmarks/nyu_depth_v2_labeled.mat', 'r') as file:
    images_dataset = file['images']
    depths_dataset = file['depths']

    num_images = 500  # Number of images to process
    selected_indices = np.arange(num_images)

    preprocessed_images = []
    preprocessed_depths = []

    input_size = (224, 224)
    for idx in selected_indices:
        img = images_dataset[idx].transpose(1, 2, 0)
        depth = depths_dataset[idx]

        img_resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth, input_size, interpolation=cv2.INTER_NEAREST)

        preprocessed_images.append(img_resized)
        preprocessed_depths.append(depth_resized)

# Load model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vitl'    #model variant
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location=DEVICE))
model = model.to(DEVICE).eval()

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict and evaluate
predicted_depths = []
for img in tqdm(preprocessed_images, desc="Generating Depth Predictions"):
    img_tensor = transform(img).to(DEVICE)
    with torch.no_grad():
        pred_depth = model(img_tensor.unsqueeze(0))
        pred_depth = pred_depth.squeeze().cpu().numpy()

        pred_depth = np.clip(pred_depth, 0, 10)  # Clip predictions to valid range
        predicted_depths.append(pred_depth)

# Compute metrics
mae_list, rmse_list, delta_list = [], [], []
for pred_depth, gt_depth in zip(predicted_depths, preprocessed_depths):
    mae, rmse, delta = calculate_metrics(pred_depth, gt_depth)
    mae_list.append(mae)
    rmse_list.append(rmse)
    delta_list.append(delta)

# Results
print("\nMetrics:")
print(f"MAE: {np.mean(mae_list):.4f}")
print(f"RMSE: {np.mean(rmse_list):.4f}")
print(f"Delta (<1.25): {np.mean(delta_list):.4f}")
