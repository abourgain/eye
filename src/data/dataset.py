"""
This file contains the dataset class.
"""

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class EyeDataset(Dataset):
    """
    Eye dataset.
    """

    def __init__(self, image_dir: str, mask_dir: str, transform=None) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 127.0] = 1.0
        mask[mask == 255.0] = 2.0

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


class CarvanaDataset(Dataset):
    """
    Carvana dataset.
    """

    def __init__(self, image_dir: str, mask_dir: str, transform=None) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


class CityscapesDataset(Dataset):
    """
    Cityscapes dataset.
    """

    def __init__(self, image_dir: str, mask_dir: str, transform=None) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("leftImg8bit.png", "gtFine_labelTrainIds.png"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 20.0

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


if __name__ == "__main__":
    dataset = EyeDataset("./data/eye/train/images", "./data/eye/train/masks")
    print((dataset[0][1] == 2.0).any())
