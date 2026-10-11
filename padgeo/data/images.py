"""Model input: one preprocessed tensor per image, shared by VGGT and DINOv2.

Both models receive the output of VGGT's own preprocessing (`vggt.utils.load_fn.load_and_preprocess_images`,
mode "crop"). It composites RGBA onto white, resizes with bicubic interpolation to width 518 and a height
rounded to a multiple of 14, and centre-crops the height only if it exceeds 518. Feeding the identical
tensor to both models makes their 14-pixel patch grids the same grid. This module also records the
original image size, which is the size of the pixel maps.

Images that VGGT would centre-crop (portrait images) are rejected: the patch grid would then cover only
part of the original image, and the pixel map could not be placed on it without extra bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from vggt.utils.load_fn import load_and_preprocess_images

from padgeo.utils.checks import check_tensor

# VGGT's preprocessing fixes the width to 518 px; 518 = 37 x 14 is also DINOv2's native resolution.
MODEL_INPUT_WIDTH = 518
PATCH_SIZE = 14


@dataclass(frozen=True)
class ModelInput:
    """One image prepared for both models.

    Attributes:
        pixels: [3, H, W] float32 RGB in [0, 1], W = 518, H a multiple of 14 (H = 518 for square images).
        original_size: (H0, W0) of the image file.
    """

    pixels: torch.Tensor
    original_size: tuple[int, int]


def model_input_height(original_size: tuple[int, int]) -> int:
    """Height VGGT's "crop" preprocessing gives an image of `original_size` (before any centre crop)."""
    height, width = original_size
    return round(height * (MODEL_INPUT_WIDTH / width) / PATCH_SIZE) * PATCH_SIZE


def load_model_input(path: Path) -> ModelInput:
    """Load one image with VGGT's preprocessing.

    Raises:
        ValueError: If VGGT's preprocessing would centre-crop the image (portrait aspect ratio).
    """
    with Image.open(path) as image:
        width, height = image.size
    expected_height = model_input_height((height, width))
    if expected_height > MODEL_INPUT_WIDTH:
        raise ValueError(f"{path}: {width}x{height} would be centre-cropped by VGGT's preprocessing")
    pixels = load_and_preprocess_images([str(path)], mode="crop")[0]
    check_tensor("pixels", pixels, (3, expected_height, MODEL_INPUT_WIDTH), torch.float32, value_range=(0.0, 1.0))
    return ModelInput(pixels=pixels, original_size=(height, width))
