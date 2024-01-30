import csv
import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm


def draw_mask(
    row: list[float],
    mask_path: str,
    width: int = 224,
    height: int = 160,
) -> None:
    """
    Draws mask on image
    Args:
        row: list of floats

    Returns: None
    """
    # Create a blank image with a white background
    _, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    plt.gca().invert_yaxis()
    ax.axis("off")
    ax.set_facecolor("black")

    # Get the ellipse parameters
    coord_p_true_x, coord_p_true_y, radiusX_p_true, radiusY_p_true, theta_p_true, coord_i_true_x, coord_i_true_y, radiusX_i_true, radiusY_i_true, theta_i_true = map(float, row)

    # Set the colors
    color1 = 127
    color2 = 255

    # Draw the first ellipse with rotation, which represents the iris
    ellipse1 = Ellipse(
        xy=(coord_i_true_x, coord_i_true_y),
        width=2 * radiusX_i_true,
        height=2 * radiusY_i_true,
        angle=theta_i_true * 180 / np.pi,
        facecolor=(color1 / 255, color1 / 255, color1 / 255),
        edgecolor="none",
    )
    ellipse1.set_alpha(1.0)
    ax.add_patch(ellipse1)

    # Draw the second ellipse with rotation, which represents the pupil
    ellipse2 = Ellipse(
        xy=(coord_p_true_x, coord_p_true_y),
        width=2 * radiusX_p_true,
        height=2 * radiusY_p_true,
        angle=theta_p_true * 180 / np.pi,
        facecolor=(color2 / 255, color2 / 255, color2 / 255),
        edgecolor="none",
    )
    ellipse1.set_alpha(1.0)
    ax.add_patch(ellipse2)

    # Save the image as a PNG
    plt.savefig(mask_path, bbox_inches="tight", pad_inches=0, dpi=100, facecolor="black")


def main(
    groundtruth_dir: str,
    masks_dir: str,
    width: int = 224,
    height: int = 160,
) -> None:
    # get all files in directory
    groundtruths = glob.glob(groundtruth_dir + "/*")

    for i, groundtruth in tqdm(enumerate(groundtruths)):
        with open(groundtruth, "r", encoding="utf-8") as f:
            # Create a CSV reader object
            csv_reader = csv.reader(f)
            # Skip the header row
            next(csv_reader)

            # Loop over each row in the file
            for row in csv_reader:
                filename = groundtruth.split("/")[-1].split(".")[0]
                mask_path = f"{masks_dir}/{filename}.png"
                draw_mask(row, mask_path, width, height)


if __name__ == "__main__":
    width, height = 224, 160
    main(
        groundtruth_dir="./data/eye/test/groundtruth",
        masks_dir="./data/eye/test/masks",
        width=width,
        height=height,
    )
