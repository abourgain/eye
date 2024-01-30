"""
Training script for the UNet model.
"""

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.dataset import CityscapesDataset, EyeDataset
from src.model import UNet
from src.training.utils import check_accuracy, get_loaders, save_checkpoint, save_predictions_as_imgs

# Hyperparameters etc.
LEARNING_RATE = 1e-4
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.has_mps:
    DEVICE = "mps"
else:
    DEVICE = "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True  # -> to explore
LOAD_MODEL = False  # -> to explore
TRAIN_IMG_DIR = "./data/eye/train/images/"
TRAIN_MASK_DIR = "./data/eye/train/masks/"
VAL_IMG_DIR = "./data/eye/test/images"
VAL_MASK_DIR = "./data/eye/test/masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Training loop.
    """
    loop = tqdm(loader)

    for _, (images, true_masks) in enumerate(loop):
        images = images.to(device=DEVICE)
        true_masks = true_masks.float().to(DEVICE)

        assert images.shape[1] == model.n_channels, (
            f"Network has been defined with {model.n_channels} input channels, " f"but loaded images have {images.shape[1]} channels. Please check that " "the images are loaded correctly."
        )

        print(f"images.shape: {images.shape}")
        print(f"true_masks.shape: {true_masks.shape}")

        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            masks_pred = model(images)
            if model.n_classes == 1:
                loss = criterion(masks_pred.squeeze(1), true_masks.float())
                loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
            else:
                loss = criterion(masks_pred, true_masks)
                loss += dice_loss(F.softmax(masks_pred, dim=1).float(), F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(), multiclass=True)

        # forward
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                predictions = model(images)
                print(f"predictions.shape: {predictions.shape}")

                # if multi-class, use softmax + argmax
                probs = torch.softmax(predictions, dim=1)
                masks_predictions = torch.argmax(probs, dim=1).float()
                print(f"predictions.shape: {masks_predictions.shape}")

                loss = loss_fn(masks_predictions, true_masks)
                print(f"loss: {loss}")

                # backward
                optimizer.zero_grad()
                loss.requires_grad = True
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            predictions = model(data)
            print(f"predictions.shape: {predictions.shape}")

            # if multi-class, use softmax + argmax
            probs = torch.softmax(predictions, dim=1)
            masks_predictions = torch.argmax(probs, dim=1).float()
            print(f"predictions.shape: {masks_predictions.shape}")

            loss = loss_fn(masks_predictions, targets)
            print(f"loss: {loss}")

            # backward
            optimizer.zero_grad()
            loss.requires_grad = True
            loss.backward()
            optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main(
    dataset: Dataset,
    in_channels: int = 3,
    out_channels: int = 1,
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

    # -> to explore, if multi-class, use out_channels=3 (or 4, 5, etc.)
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(DEVICE)
    # -> to explore, if multi-class, use CrossEntropyLoss
    loss_fn = nn.BCEWithLogitsLoss() if out_channels == 1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        dataset,
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    for _ in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        # -> to explore, if multi-class, use IoU for score
        check_accuracy(val_loader, model, num_classes=out_channels, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, num_classes=out_channels, folder="saved_images/", device=DEVICE)


if __name__ == "__main__":
    main(
        dataset=EyeDataset,
        in_channels=3,
        out_channels=2,
    )
