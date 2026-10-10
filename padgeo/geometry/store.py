"""Patch-level geometry of one query's image set, computed once and cached.

The geometry of an image set depends on every image in it and on their order (image 0 defines VGGT's
world frame). So the cache key holds the ordered list of image content hashes, the backend identity
(model, revision, point source, precision) and the foreground settings used to pool pixels into patches.
That covers class, seed, N and sampling mode: two runs share an entry only if they ran the backend on the
very same ordered images. Only patch-level xyz and confidence are stored, not full-resolution maps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from padgeo.config import ForegroundConfig
from padgeo.data.base import ImageRef
from padgeo.data.images import MODEL_INPUT_WIDTH, PATCH_SIZE, load_model_input
from padgeo.features.foreground import pixel_foreground
from padgeo.geometry.base import GeometryBackend
from padgeo.geometry.patches import PatchGeometry, pool_to_patch_grid
from padgeo.utils.cache import TensorCache


class GeometryStore:
    """Cache-backed access to patch geometry; the backend runs only on a miss.

    Args:
        backend: Geometry backend (its model is loaded lazily, i.e. never if every set is cached).
        cache: Tensor cache (namespace "geometry").
        foreground: Foreground settings; pixels outside the mask do not contribute to a patch's point.
    """

    def __init__(self, backend: GeometryBackend, cache: TensorCache, foreground: ForegroundConfig) -> None:
        self.backend = backend
        self.cache = cache
        self.foreground = foreground

    def _identity(self, images: Sequence[ImageRef]) -> dict[str, Any]:
        return {
            "geometry": self.backend.identity,
            "foreground": self.foreground.model_dump(mode="json"),
            "images": [ref.content_hash for ref in images],
            "input_width": MODEL_INPUT_WIDTH,
            "patch_size": PATCH_SIZE,
        }

    def get(self, images: Sequence[ImageRef]) -> tuple[PatchGeometry, bool]:
        """Patch geometry for `images` in the given order (image 0 defines the world frame).

        Returns:
            The patch geometry (xyz [S, h, w, 3], confidence [S, h, w]) and whether it came from cache.
        """
        identity = self._identity(images)
        entry = self.cache.load(identity)
        if entry is not None:
            return PatchGeometry(xyz=entry["xyz"], confidence=entry["confidence"]), True
        model_inputs = [load_model_input(ref.path) for ref in images]
        if len({tuple(item.pixels.shape) for item in model_inputs}) != 1:
            raise ValueError("all images of a geometry pass must have the same model-input size")
        pixels = torch.stack([item.pixels for item in model_inputs])
        foreground = torch.stack([pixel_foreground(item.pixels, self.foreground) for item in model_inputs])
        patch_geometry = pool_to_patch_grid(self.backend.predict(pixels), foreground, PATCH_SIZE)
        self.cache.save(identity, {"xyz": patch_geometry.xyz, "confidence": patch_geometry.confidence})
        return patch_geometry, False
