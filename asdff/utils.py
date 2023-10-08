from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from diffusers.utils import BaseOutput
from PIL import Image, ImageFilter, ImageOps
import torch

@dataclass
class ADOutput(BaseOutput):
    images: list[Image.Image]
    init_images: list[Image.Image]


def mask_dilate(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    arr = np.array(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(dilated)


def mask_gaussian_blur(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    blur = ImageFilter.GaussianBlur(value)
    return image.filter(blur)


def bbox_padding(
    bbox: tuple[int, int, int, int], image_size: tuple[int, int], value: int = 32
) -> tuple[int, int, int, int]:
    if value <= 0:
        return bbox

    arr = np.array(bbox).reshape(2, 2)
    arr[0] -= value
    arr[1] += value
    arr = np.clip(arr, (0, 0), image_size)
    return tuple(arr.flatten())


def composite(
    init: Image.Image,
    mask: Image.Image,
    gen: Image.Image,
    bbox_padded: tuple[int, int, int, int],
) -> Image.Image:
    img_masked = Image.new("RGBa", init.size)
    img_masked.paste(
        init.convert("RGBA").convert("RGBa"),
        mask=ImageOps.invert(mask),
    )
    img_masked = img_masked.convert("RGBA")

    size = (
        bbox_padded[2] - bbox_padded[0],
        bbox_padded[3] - bbox_padded[1],
    )
    resized = gen.resize(size)

    output = Image.new("RGBA", init.size)
    output.paste(resized, bbox_padded)
    output.alpha_composite(img_masked)
    return output.convert("RGB")



def make_inpaint_condition(init_image, mask_image):
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

    assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image
