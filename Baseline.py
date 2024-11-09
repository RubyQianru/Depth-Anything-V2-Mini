import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

class DepthAnythingBaseline(nn.Module):
    def __init__(self, pretrained_model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        super(DepthAnythingBaseline, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.depth_anything_v2 = AutoModelForDepthEstimation.from_pretrained(pretrained_model_name)

        # Freeze the pretrained model parameters
        # for param in self.depth_anything_v2.parameters():
        #     param.requires_grad = False
        
        # Get the output dimension of the Depth-Anything-V2 model
        # with torch.no_grad():
        #     dummy_input = torch.randn(1, 3, 384, 384)  # Adjust size if needed
        #     dummy_output = self.depth_anything_v2(dummy_input)
        #     output_dim = dummy_output.predicted_depth.shape[1:]
        
        # self.custom_head = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 1, kernel_size=1)
        # )

    def forward(self, x):
        depth_output = self.depth_anything_v2(x).predicted_depth
        
        # Pass through custom head
        # refined_depth = self.custom_head(depth_output)
        
        return depth_output

def create_baseline():
    model = DepthAnythingBaseline()
    
    criterion = nn.L1Loss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    return model, criterion, optimizer
