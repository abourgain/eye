"""
EyeDataset class to load images and ground truth data from a directory.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image


import torch
from torch.utils.data import Dataset

from src.constants import VAL_TRANSFORM


def load_images(images_dir: str) -> torch.Tensor:
    """
    Load an image from the image directory and preprocess it.
    """
    # Get the image filenames
    image_filenames = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.endswith(".png")
    ]
    image_filenames.sort()  # Ensure matching order with groundtruth

    images = []

    # Load the image
    for img_name in image_filenames:
        print(f"Loading image: {img_name}")
        image = Image.open(img_name).convert("RGB")

        # Convert PIL Image to NumPy array for Albumentations
        image_np = np.array(image)

        # Preprocess the image
        preprocess = VAL_TRANSFORM

        images.append(preprocess(image=image_np)["image"])

    images = torch.stack(images, dim=0)
    return images, image_filenames


def save_predictions(
    prediction: torch.Tensor, image_filenames: list, output_path: str
) -> None:
    """
    Save the prediction to the output path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Save the prediction in the .csv format for each image
    for i, pred in enumerate(prediction):
        image_name = os.path.basename(image_filenames[i]).replace(".png", "")
        output_filename = os.path.join(output_path, f"{image_name}.csv")

        # Header
        header = "coord_p_true_x,coord_p_true_y,radiusX_p_true,radiusY_p_true,theta_p_true,coord_i_true_x,coord_i_true_y,radiusX_i_true,radiusY_i_true,theta_i_true\n"

        # Convert the prediction to a string
        pred_str = ",".join([str(p) for p in pred.tolist()])
        pred_str = pred_str.replace("[", "").replace("]", "").replace(" ", "")

        # Write the prediction to a file
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(pred_str)


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
