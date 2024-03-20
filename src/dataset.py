"""
EyeDataset class to load images and ground truth data from a directory.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class EyeDataset(Dataset):
    """
    Eye Dataset
    """

    def __init__(self, images_dir, groundtruth_dir, transform=None):
        """
        Args:
            images_dir (string): Directory with all the images.
            groundtruth_dir (string): Directory with all the ground truth CSV files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.groundtruth_dir = groundtruth_dir
        self.transform = transform

        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith(".png")]
        self.image_filenames.sort()  # Ensure matching order with groundtruth

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")

        # Convert PIL Image to NumPy array for Albumentations
        image_np = np.array(image)

        # Assuming groundtruth filenames match the image filenames but with .csv extension
        groundtruth_name = os.path.join(
            self.groundtruth_dir, self.image_filenames[idx].replace(".png", ".csv")
        )
        groundtruth = pd.read_csv(groundtruth_name)

        # Convert groundtruth DataFrame to a tensor or your required format here
        groundtruth_tensor = torch.tensor(
            groundtruth.values, dtype=torch.float
        ).reshape(-1)

        if self.transform:
            transformed = self.transform(image=image_np)
            image = transformed["image"]

        return image, groundtruth_tensor
