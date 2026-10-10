"""Foreground masks: pixel thresholding against pure white and the patch-fraction rule."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from padgeo.config import ForegroundConfig
from padgeo.features.foreground import patch_foreground, pixel_foreground

CONFIG = ForegroundConfig(pixel_threshold=0.98, min_patch_fraction=0.1)


def test_pixel_foreground_thresholds_the_darkest_channel():
    pixels = torch.ones(3, 28, 28)
    pixels[:, 0, 0] = 0.8  # light grey plastic
    pixels[:, 0, 1] = torch.tensor([1.0, 1.0, 0.1])  # yellow: one dark channel suffices
    pixels[:, 0, 2] = 0.99  # anti-aliasing rim / highlight: background
    mask = pixel_foreground(pixels, CONFIG)
    assert mask[0, :3].tolist() == [True, True, False]
    assert mask.sum() == 2


def test_patch_foreground_fraction_rule_and_alignment():
    pixel_mask = torch.zeros(28, 42, dtype=torch.bool)  # 2 x 3 patches
    pixel_mask[0:14, 14:28] = True  # patch (0, 1) fully foreground
    pixel_mask[14:16, 28:42] = True  # patch (1, 2): 28 of 196 pixels = 14% foreground
    pixel_mask[20, 0] = True  # patch (1, 0): 1 of 196 pixels, below 10 %
    patches = patch_foreground(pixel_mask, 14, CONFIG)
    assert patches.tolist() == [[False, True, False], [False, False, True]]


def test_patch_foreground_rejects_misaligned_masks():
    with pytest.raises(ValueError, match="multiple of the patch size"):
        patch_foreground(torch.zeros(20, 28, dtype=torch.bool), 14, CONFIG)


def test_foreground_patches_always_have_foreground_pixels():
    tiny = ForegroundConfig(pixel_threshold=0.98, min_patch_fraction=1e-9)
    assert not patch_foreground(torch.zeros(14, 14, dtype=torch.bool), 14, tiny).any()
    with pytest.raises(ValidationError):
        ForegroundConfig(pixel_threshold=0.98, min_patch_fraction=0.0)  # would admit empty patches
