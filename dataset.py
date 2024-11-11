import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import util.image as u

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import util.image as u

class NYUDepthV2Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file = h5py.File(file_path, 'r')
        self.transform = transform
        self.images = []
        self.depths = []
        
        for i in range(len(self.file['images'])):
            image = self.file['images'][i].transpose(1, 2, 0)
            depth = self.file['depths'][i]
            
            image = u.resize_image(image)
            depth = u.resize_image(depth)
            
            self.images.append(image)
            self.depths.append(depth)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        depth = self.depths[idx]

        if self.transform:
            image = self.transform(image)

        return image, depth

    def __del__(self):
        self.file.close()