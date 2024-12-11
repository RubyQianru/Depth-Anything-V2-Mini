import torch
import torch.quantization
import os
from depth_anything_v2.dpt import DepthAnythingV2

# Model configurations for Depth Anything V2
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

def load_model(variant, checkpoint_path):
    """
    Loads the Depth Anything V2 model with the specified variant and its state dictionary.
    :param variant: Model variant (e.g., 'vits', 'vitb', 'vitl')
    :param checkpoint_path: Path to the checkpoint file
    :return: Loaded Depth Anything V2 model
    """
    print(f"Loading model variant '{variant}' from {checkpoint_path}...")
    model = DepthAnythingV2(**model_configs[variant])
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model

def apply_quantization(model, qat=False):
    """
    Applies quantization to a model.
    :param model: PyTorch model
    :param qat: Whether to use Quantization-Aware Training (QAT)
    :return: Quantized model
    """
    # Modify the default qconfig to use per-tensor quantization
    custom_qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    )
    torch.quantization.default_qconfig = custom_qconfig

    if qat:
        print("Applying Quantization-Aware Training...")
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model = torch.quantization.prepare_qat(model)
    else:
        print("Applying Post-Training Quantization...")
        model.eval()
        model.qconfig = custom_qconfig
        model = torch.quantization.prepare(model)
    return torch.quantization.convert(model)

def save_model(model, output_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # Checkpoint directory and output path
    checkpoints = {
        "vits": "checkpoints/depth_anything_v2_vits.pth",
        "vitb": "checkpoints/depth_anything_v2_vitb.pth",
        "vitl": "checkpoints/depth_anything_v2_vitl.pth",
    }
    
    for variant, path in checkpoints.items():
        output_path = f"compressed_models/{variant}_quantized.pth"
        
        # Load the model
        model = load_model(variant, path)
        
        # Apply quantization (Post-Training Quantization is used here)
        quantized_model = apply_quantization(model, qat=False)
        
        # Save the quantized model
        save_model(quantized_model, output_path)
