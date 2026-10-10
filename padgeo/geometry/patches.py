"""Join step: pool per-pixel geometry onto the patch grid shared with DINOv2.

Both models see the same [3, H, W] model input, and both use 14-pixel patches. So patch (i, j) of the
DINOv2 grid (token index i * w + j) and patch (i, j) here cover the same input pixels:
rows 14i .. 14i+13 and columns 14j .. 14j+13.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from padgeo.geometry.base import PixelGeometry
from padgeo.utils.checks import check_tensor


@dataclass(frozen=True)
class PatchGeometry:
    """Patch-level geometry of S images (what the geometry cache stores).

    Attributes:
        xyz: [S, h, w, 3] float32 mean 3D point of each patch's foreground pixels (world frame of the
            geometry pass); NaN for patches without any foreground pixel.
        confidence: [S, h, w] float32 mean confidence over the same pixels; NaN likewise.
    """

    xyz: torch.Tensor
    confidence: torch.Tensor


def pool_to_patch_grid(
    geometry: PixelGeometry, pixel_foreground: torch.Tensor, patch_size: int
) -> PatchGeometry:
    """Average points and confidence over the foreground pixels of each patch.

    Averaging only foreground pixels keeps a boundary patch's point on the object instead of pulling it
    towards the background depth. Patches with no foreground pixel get NaN, so any accidental use of a
    background patch's geometry surfaces as a failed finiteness check.

    Args:
        geometry: Per-pixel points [S, H, W, 3] and confidence [S, H, W].
        pixel_foreground: [S, H, W] bool per-pixel foreground mask of the same model inputs.
        patch_size: Patch side in pixels (14); H and W must be multiples of it.

    Returns:
        Patch-level xyz [S, h, w, 3] and confidence [S, h, w], h = H / patch_size, w = W / patch_size.
    """
    num, height, width, _ = geometry.points.shape
    check_tensor("pixel_foreground", pixel_foreground, (num, height, width), torch.bool)
    if height % patch_size or width % patch_size:
        raise ValueError(f"input {height}x{width} is not a multiple of the patch size {patch_size}")
    h, w = height // patch_size, width // patch_size
    weights = pixel_foreground.to(torch.float32).reshape(num, h, patch_size, w, patch_size)
    count = weights.sum(dim=(2, 4))
    points = geometry.points.reshape(num, h, patch_size, w, patch_size, 3)
    confidence = geometry.confidence.reshape(num, h, patch_size, w, patch_size)
    sum_points = (points * weights[..., None]).sum(dim=(2, 4))
    sum_confidence = (confidence * weights).sum(dim=(2, 4))
    has_foreground = count > 0
    safe_count = count.clamp(min=1.0)
    nan = torch.tensor(float("nan"))
    return PatchGeometry(
        xyz=torch.where(has_foreground[..., None], sum_points / safe_count[..., None], nan),
        confidence=torch.where(has_foreground, sum_confidence / safe_count, nan),
    )
