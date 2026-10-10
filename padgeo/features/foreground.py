"""Foreground masks from image content only (no labels, no ground truth), identical for both methods.

MAD-Sim renders have a pure white background (every background pixel is exactly 255; see
docs/DATA_NOTES.md), so background thresholding is exact. A pixel of the model input is foreground if its
darkest RGB channel is below `pixel_threshold`. That keeps light-grey and white-ish plastic, which is
shaded below pure white, and drops the white anti-aliasing rim and see-through gaps between parts. A patch
is foreground if at least `min_patch_fraction` of its pixels are, so silhouette patches, where many
Burrs and Missing defects sit, are scored.
"""

from __future__ import annotations

import torch

from padgeo.config import ForegroundConfig
from padgeo.utils.checks import check_tensor


def pixel_foreground(pixels: torch.Tensor, config: ForegroundConfig) -> torch.Tensor:
    """[H, W] bool foreground mask of one model input [3, H, W] in [0, 1]."""
    check_tensor("pixels", pixels, (3, None, None), torch.float32, value_range=(0.0, 1.0))
    return pixels.min(dim=0).values < config.pixel_threshold


def patch_foreground(pixel_mask: torch.Tensor, patch_size: int, config: ForegroundConfig) -> torch.Tensor:
    """[h, w] bool patch mask: patch (i, j) is foreground if enough of its pixels are.

    Patch (i, j) covers pixel rows patch_size*i .. patch_size*i + patch_size - 1 and the same columns,
    which is the DINOv2 token grid. `min_patch_fraction` > 0 (enforced by the config schema), so a
    foreground patch always has at least one foreground pixel and its pooled geometry is finite.
    """
    check_tensor("pixel_mask", pixel_mask, (None, None), torch.bool)
    height, width = pixel_mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"mask {height}x{width} is not a multiple of the patch size {patch_size}")
    blocks = pixel_mask.reshape(height // patch_size, patch_size, width // patch_size, patch_size)
    fraction = blocks.to(torch.float32).mean(dim=(1, 3))
    return fraction >= config.min_patch_fraction
