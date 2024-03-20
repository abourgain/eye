"""
This file contains the constants used in the project.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_HEIGHT = 224  # 1280 originally
IMAGE_WIDTH = 224  # 1918 originally
TRAIN_TRANSFORM = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)
VAL_TRANSFORM = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

TRAIN_IMG_DIR = "./data/eye/train/images/"
TRAIN_GROUNDTRUTH_DIR = "./data/eye/train/groundtruth/"
VAL_IMG_DIR = "./data/eye/test/images"
VAL_GROUNDTRUTH_DIR = "./data/eye/test/groundtruth/"
