"""Patch scores to image score and pixel map. One implementation, used identically by both methods.

Both follow AnomalyDINO's code (github.com/dammsi/AnomalyDINO @ b9d1c26, `mean_top1p` and `dists2map`):
the image score is computed on the full patch grid with background patches at 0, and the pixel map is a
bilinear upsampling of that grid to the original image size followed by Gaussian smoothing.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from padgeo.config import ImageScore, MaxImageScore, TopPercentImageScore
from padgeo.utils.checks import check_tensor


def aggregate_image_score(patch_scores: torch.Tensor, config: ImageScore) -> float:
    """Image-level anomaly score from the full patch-score grid (background patches are 0).

    Args:
        patch_scores: [h, w] float32 patch scores, >= 0, background patches 0.
        config: "max": the largest patch score. "top_percent": the mean of the
            `k = max(1, floor(h * w * percent / 100))` largest patch scores. Like AnomalyDINO's `mean_top1p`,
            k depends on the grid size only, so it is the same for every image of a given input size,
            however many patches are foreground; background zeros enter only if fewer than k patches are
            foreground.

    Returns:
        The image score (0.0 for a query without foreground patches).
    """
    check_tensor("patch_scores", patch_scores, (None, None), torch.float32, finite=True)
    flat = patch_scores.flatten()
    if isinstance(config, MaxImageScore):
        return float(flat.max())
    if isinstance(config, TopPercentImageScore):
        k = max(1, math.floor(flat.numel() * config.percent / 100.0))
        return float(flat.topk(k).values.mean())
    raise TypeError(f"unknown image score config {config!r}")


def make_pixel_map(patch_scores: torch.Tensor, image_size: tuple[int, int], sigma: float) -> np.ndarray:
    """Upsample a patch-score grid to the original image size, then smooth it.

    The h x w patch grid tiles the whole model input, and the model input is the whole original image
    (resize without crop; checked when images are loaded). So bilinear interpolation with
    `align_corners=False` (identical to OpenCV's INTER_LINEAR used by AnomalyDINO) puts patch (i, j)'s
    score at the centre of original-image rows [i * H0 / h, (i + 1) * H0 / h) and columns
    [j * W0 / w, (j + 1) * W0 / w).

    Args:
        patch_scores: [h, w] float32 (background patches 0).
        image_size: (H0, W0) of the original image.
        sigma: Gaussian sigma in original-image pixels (scipy defaults: mode "reflect", truncate 4);
            0 disables smoothing.

    Returns:
        [H0, W0] float32 pixel anomaly map.
    """
    check_tensor("patch_scores", patch_scores, (None, None), torch.float32, finite=True)
    height, width = image_size
    upsampled = F.interpolate(
        patch_scores[None, None].to(torch.float64), size=(height, width), mode="bilinear", align_corners=False
    )[0, 0]
    pixel_map = upsampled.numpy()
    if sigma > 0:
        pixel_map = gaussian_filter(pixel_map, sigma=sigma)
    return pixel_map.astype(np.float32)
