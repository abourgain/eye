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
    train_groundtruth_dir: str,
    val_dir: str,
    val_groundtruth_dir: str,
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
        images_dir=train_dir,
        groundtruth_dir=train_groundtruth_dir,
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
        images_dir=val_dir,
        groundtruth_dir=val_groundtruth_dir,
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


def check_accuracy(loader, model, device: str = "cpu"):
    """
    Check MSE of a trained model on a given loader for a regression problem.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    criterion = torch.nn.MSELoss()  # Define the loss function

    with torch.no_grad():  # No need to compute gradients when evaluating
        for x, y in loader:
            x = x.to(device=device)  # Move input to the specified device
            y = y.to(device=device)  # Ensure target is on the same device as input

            preds = model(x)  # Get model predictions

            loss = criterion(
                preds, y
            )  # Compute the loss between predictions and true values
            total_loss += loss.item() * x.size(
                0
            )  # Update total loss (multiplied by batch size for mean)

    mean_loss = total_loss / len(loader.dataset)  # Compute mean loss over all samples

    print(f"Mean Squared Error: {mean_loss:.4f}")
    model.train()  # Set the model back to train mode
