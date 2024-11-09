import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import utils as u

class NYUDepthV2Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file = h5py.File(file_path, 'r')
        self.images = self.file['images']
        self.depths = self.file['depths']
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx].transpose(1, 2, 0)  # Change from (3, H, W) to (H, W, 3)
        depth = self.depths[idx]

        # if self.transform:
        #     image = self.transform(image)
        #     depth = torch.from_numpy(depth).unsqueeze(0).float()  # Add channel dimension and convert to float

        return image, depth

    def __del__(self):
        self.file.close()