"""
Training script for the AlexNet model.
"""

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataset import EyeDataset
from src.model import AlexNet
from src.utils import (
    check_accuracy,
    get_loaders,
    save_checkpoint,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
L2 = 1e-5
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_built():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 30
NUM_WORKERS = 2
IMAGE_HEIGHT = 224  # 1280 originally
IMAGE_WIDTH = 224  # 1918 originally
PIN_MEMORY = True  # -> to explore
LOAD_MODEL = False  # -> to explore
TRAIN_IMG_DIR = "./data/eye/train/images/"
TRAIN_GROUNDTRUTH_DIR = "./data/eye/train/groundtruth/"
VAL_IMG_DIR = "./data/eye/test/images"
VAL_GROUNDTRUTH_DIR = "./data/eye/test/groundtruth/"


def train(loader, model, optimizer, loss_fn, scaler):
    """
    Training loop.
    """
    loop = tqdm(loader, leave=True)

    for _, (images, groundtruth) in enumerate(loop):
        images = images.to(device=DEVICE)
        groundtruth = groundtruth.to(DEVICE)

        assert images.shape[1] == model.in_channels, (
            f"Network has been defined with {model.in_channels} input channels, "
            f"but loaded images have {images.shape[1]} channels. Please check that "
            "the images are loaded correctly."
        )

        # forward
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():  # automatically casts floating-point operations to half-precision (float16) where it's safe to do so if CUDA is available
                preds = model(images)

                loss = loss_fn(preds, groundtruth)

                # backward
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            preds = model(images)

            loss = loss_fn(preds, groundtruth)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main(
    dataset: Dataset,
    in_channels: int = 3,
    output_dim: int = 1,
) -> None:
    """
    Main training function.
    """
    train_transform = A.Compose(
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
    val_transform = A.Compose(
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

    print(f"Using device: {DEVICE}")

    model = AlexNet(in_channels=in_channels, output_dim=output_dim).to(DEVICE)
    num_params = sum(item.numel() for item in model.parameters())
    print(f"{model.__class__.__name__} - Number of parameters: {num_params}")

    # Loss for a regression problem
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2)

    train_loader, val_loader = get_loaders(
        dataset,
        TRAIN_IMG_DIR,
        TRAIN_GROUNDTRUTH_DIR,
        VAL_IMG_DIR,
        VAL_GROUNDTRUTH_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    for _ in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main(
        dataset=EyeDataset,
        in_channels=3,
        output_dim=10,
    )
