"""VGGT backend: frozen VGGT-1B, one forward pass over all S images of a query.

The forward pass mirrors the "detailed usage" of the VGGT README at the pinned commit. The aggregator runs
under autocast when precision is "bf16"/"fp16"; the heads always run in fp32, as `VGGT.forward` does.
Only the heads needed for the chosen point source are kept:

- "depth" (default): camera head + depth head, then depth unprojected with the predicted cameras through
  VGGT's own utility (`padgeo.geometry.camera.unproject_depth`). The VGGT README states this is usually
  more accurate than the point-map head. Confidence is the depth head's.
- "pointmap": the point-map head's world points and confidence.

Both are expressed in the camera frame of image 0, VGGT's world frame.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from padgeo.config import PointSource, VGGTGeometryConfig
from padgeo.geometry.base import GeometryBackend, PixelGeometry
from padgeo.geometry.camera import unproject_depth
from padgeo.utils.checks import check_tensor
from padgeo.utils.precision import autocast
from padgeo.utils.provenance import distribution_version

logger = logging.getLogger(__name__)


def keep_needed_heads(model: VGGT, point_source: PointSource) -> VGGT:
    """Drop the heads whose output the point source does not use (the track head is never used).

    "depth" keeps the camera and depth heads; "pointmap" keeps the point head. Dropped heads are
    neither computed nor kept in memory.
    """
    model.track_head = None
    if point_source == "depth":
        model.point_head = None
    else:
        model.camera_head = None
        model.depth_head = None
    return model


class VGGTBackend(GeometryBackend):
    """Loads the checkpoint lazily on the first `predict`, so runs served from cache never load it."""

    def __init__(self, config: VGGTGeometryConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._model: VGGT | None = None

    @property
    def identity(self) -> dict[str, Any]:
        """Model, checkpoint revision, code version, point source and precision (part of cache keys)."""
        return {
            "backend": "vggt",
            "model": self.config.model,
            "revision": self.config.revision,
            "vggt_package": distribution_version("vggt"),  # includes the git commit
            "point_source": self.config.point_source,
            "precision": self.config.precision,
        }

    def _load_model(self) -> VGGT:
        if self._model is None:
            logger.info("Loading %s@%s on %s", self.config.model, self.config.revision, self.device)
            # strict=True: every checkpoint tensor must match the architecture (the mixin defaults to False).
            model = VGGT.from_pretrained(self.config.model, revision=self.config.revision, strict=True)
            self._model = keep_needed_heads(model, self.config.point_source).eval().to(self.device)
        return self._model

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> PixelGeometry:
        """Points and confidence for `images` [S, 3, H, W] in [0, 1]; image 0 defines the world frame."""
        check_tensor("images", images, (None, 3, None, None), torch.float32, value_range=(0.0, 1.0))
        model = self._load_model()
        batch = images.to(self.device)[None]  # [1, S, 3, H, W]
        with autocast(self.config.precision, self.device):
            tokens, patch_start = model.aggregator(batch)
        with torch.autocast(device_type=self.device.type, enabled=False):
            if self.config.point_source == "depth":
                if model.camera_head is None or model.depth_head is None:
                    raise RuntimeError("depth point source needs the camera and depth heads")
                pose_encoding = model.camera_head(tokens)[-1]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_encoding, batch.shape[-2:])
                depth, confidence = model.depth_head(tokens, images=batch, patch_start_idx=patch_start)
                points = unproject_depth(
                    depth[0, ..., 0].float(), extrinsics[0].float(), intrinsics[0].float()
                )
            else:
                if model.point_head is None:
                    raise RuntimeError("pointmap point source needs the point head")
                world_points, confidence = model.point_head(tokens, images=batch, patch_start_idx=patch_start)
                points = world_points[0].float().cpu()
        return PixelGeometry(points=points, confidence=confidence[0].float().cpu())
