import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

class DepthAnythingBaseline(nn.Module):
    def __init__(self, pretrained_model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        super(DepthAnythingBaseline, self).__init__()
        self.depth_anything_v2 = AutoModelForDepthEstimation.from_pretrained(pretrained_model_name)

        # Freeze the pretrained model parameters
        # for param in self.depth_anything_v2.parameters():
        #     param.requires_grad = False
        
        # self.custom_head = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 1, kernel_size=1)
        # )

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.depth_anything_v2(inputs).predicted_depth
        # refined_depth = self.custom_head(depth_output)
        outputs = nn.functional.interpolate(outputs.unsqueeze(1), size=(240, 320), mode='bilinear', align_corners=False).squeeze(1)
        return outputs

def create_baseline():
    model = DepthAnythingBaseline()
    
    criterion = nn.L1Loss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    return model, criterion, optimizer
