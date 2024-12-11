import torch

# Specify the path to your .pth file
pth_file_path = "checkpoints/depth_anything_v2_vitl.pth"

# Load the checkpoint
model_weights = torch.load(pth_file_path)

# Check the type of the loaded object
print(f"Type of the loaded object: {type(model_weights)}\n")

# Check if it contains a state_dict
if isinstance(model_weights, dict) and 'state_dict' in model_weights:
    state_dict = model_weights['state_dict']
    print("The checkpoint contains a 'state_dict'. Inspecting its layers and parameters:\n")
else:
    state_dict = model_weights
    print("The checkpoint does not contain a separate 'state_dict'. Inspecting the raw data:\n")

# Print all layer names and their dimensions
for layer_name, param in state_dict.items():
    print(f"{layer_name}: {param.shape if hasattr(param, 'shape') else 'N/A'}")
