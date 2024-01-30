"""
Converts json to mask images 
"""

import json

import numpy as np
from PIL import Image

from src.constants import LABEL2INT as label2int


class InputStream:
    """
    Class for reading bits from bytes array
    """

    def __init__(self, data: list[int]):
        self.data = self._bytes2bit(data)
        self.i = 0

    def read(self, size: int) -> int:
        """
        read bits from bytes array
        """
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2)

    def _access_bit(self, data: list, num: int) -> int:
        """
        from bytes array to bits by num position
        """
        base = int(num // 8)
        shift = 7 - int(num % 8)
        return (data[base] & (1 << shift)) >> shift

    def _bytes2bit(self, data: list[int]) -> str:
        """
        get bit string from bytes data
        """
        return "".join([str(self._access_bit(data, i)) for i in range(len(data) * 8)])


def rle_decode(rle: list[int], shape: tuple[int, int]) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        shape: (width, height)

    Returns: np.array
    """

    rle_input = InputStream(rle)
    width, height = shape

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    image[image > 0] = 255

    return image


def combine_masks(masks: dict[str, np.array]) -> np.array:
    """
    Combines masks into one image
    Args:
        masks: dict of masks

    Returns: np.array
    """
    combined_mask = np.zeros_like(masks[list(masks.keys())[0]], dtype=np.uint8)
    for label, mask in masks.items():
        combined_mask[mask == 255] = label2int[label]
    return combined_mask


def save_combined_mask_to_png(combined_mask: np.array, output_filename: int) -> None:
    """
    Saves combined mask to png image with 'L' mode
    Args:
        combined_mask: np.array
        output_filename: str

    Returns: None"""
    img = Image.fromarray(combined_mask, mode="L")
    img.save(output_filename)


def main(json_path: str = "mask.json", masks_dir: str = "./masks") -> None:
    """
    Converts json to mask images
    Args:
        json_path: path to json file
        masks_dir: path to masks directory

    Returns: None"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, img in enumerate(data):
        img_path = img["image"]
        mask_path = f"{masks_dir}/{img_path.split('-')[-1]}"
        print(f"Processing {i + 1}/{len(data)}: {mask_path}")

        masks = {}
        for mask in img["tag"]:
            shape = (mask["original_width"], mask["original_height"])
            label = mask["brushlabels"][0]
            rle = mask["rle"]
            masks[label] = rle_decode(rle, shape)

        combined_mask = combine_masks(masks)

        # save mask
        save_combined_mask_to_png(combined_mask, mask_path)


if __name__ == "__main__":
    main("brush_min.json", "./masks")
