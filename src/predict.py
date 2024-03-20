"""
Predict script for predict the model.
"""

import torch

from src.dataset import load_images, save_predictions
from src.model import AlexNet
from src.utils import load_checkpoint


def predict(
    model: torch.nn.Module,
    images: torch.Tensor,
):
    """
    Predict the model and return the prediction.
    """
    # Set the model to evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        prediction = model(images)

    return prediction


def main(
    model_path: str,
    model: torch.nn.Module,
    image_dir: str,
    output_path: str = "./predictions",
):
    """
    Main function for predict.
    """
    # Load the model
    model = load_checkpoint(model_path, model)
    print(f"Model loaded from {model_path}: {model.__class__.__name__}")

    # Load the image and preprocess
    images, image_filenames = load_images(image_dir)

    # Make prediction
    prediction = predict(model, images)

    # Save the prediction
    save_predictions(prediction, image_filenames, output_path)


if __name__ == "__main__":
    main(
        model_path="./model.pth.tar",
        model=AlexNet(in_channels=3, output_dim=10),
        image_dir="./data/eye/test/images",
        output_path="./data/eye/test/predictions",
    )
