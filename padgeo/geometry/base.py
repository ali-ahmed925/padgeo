"""Geometry backend interface.

A backend maps the S model-input images of one query (the query first, then its references in canonical
order, as arranged by the pipeline) to per-pixel 3D points in one shared world frame plus a confidence.

Conventions, identical for every backend:

- World frame: the camera frame of image 0 (VGGT's convention), OpenCV axes (+x right, +y down,
  +z forward), arbitrary scale. Downstream code only uses relative distances, so the scale is irrelevant.
- `points[s, v, u]` is the 3D point imaged at row v, column u of input image s.
- Confidence: higher means more reliable; the scale is backend-specific (VGGT: 1 + exp(x) >= 1).

New backends (e.g. MASt3R or VGGT successors) implement `identity` and `predict` and are added to
`padgeo.geometry.build_geometry_backend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from padgeo.utils.checks import check_tensor


@dataclass(frozen=True)
class PixelGeometry:
    """Per-pixel geometry of S images (CPU tensors).

    Attributes:
        points: [S, H, W, 3] float32 3D points in the world frame (camera frame of image 0).
        confidence: [S, H, W] float32 per-pixel confidence.
    """

    points: torch.Tensor
    confidence: torch.Tensor

    def __post_init__(self) -> None:
        check_tensor("points", self.points, (None, None, None, 3), torch.float32, finite=True)
        s, h, w, _ = self.points.shape
        check_tensor("confidence", self.confidence, (s, h, w), torch.float32, finite=True)


class GeometryBackend(ABC):
    """Predicts per-pixel geometry for a set of images in one forward pass."""

    @property
    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Everything the output depends on besides the images (used in cache keys)."""

    @abstractmethod
    def predict(self, images: torch.Tensor) -> PixelGeometry:
        """Predict geometry for one image set.

        Args:
            images: [S, 3, H, W] float32 in [0, 1] (model input, see `padgeo.data.images`); image 0
                defines the world frame.

        Returns:
            Per-pixel points and confidence, on CPU.
        """
