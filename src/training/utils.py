"""
Utility functions for training and evaluation.
"""

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Save model checkpoint.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """
    Load model state dictionary from a checkpoint.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    dataset: Dataset,
    train_dir: str,
    train_maskdir: str,
    val_dir: str,
    val_maskdir: str,
    batch_size: int,
    train_transform: torchvision.transforms,
    val_transform: torchvision.transforms,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Function to create training and validation dataloaders.
    """
    train_ds = dataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = dataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, num_classes: int, device: str = "cpu"):
    """
    Check accuracy of a trained model for a binary classification task.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.unsqueeze(1).to(device=device)

            if num_classes == 1:  # binary classification
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
                print(f"Dice score: {dice_score/len(loader)}")
            else:  # multi-class classification
                preds = torch.softmax(model(x), dim=1)
                preds = torch.argmax(preds, dim=1).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    model.train()


def save_predictions_as_imgs(loader, model, num_classes: int, folder="./saved_images", device="cpu"):
    """
    Function to save predictions as images.
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            if num_classes == 1:  # binary classification
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            else:  # multiclass classification
                preds = torch.softmax(model(x), dim=1)
                preds = torch.argmax(preds, dim=1).float()

                # Unsqueeze the tensors to add the channel dimension
                # This is required by torchvision.utils.save_image
                preds = preds.unsqueeze(1)
                y = y.unsqueeze(1)

                # Get to distinguisable color values
                preds = preds * 127
                y = y * 127

        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()
