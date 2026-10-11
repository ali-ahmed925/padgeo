"""Frozen DINOv2 patch features, cached per image.

The extractor runs on the same [3, H, W] model input that VGGT receives (VGGT's own preprocessing,
`padgeo.data.images`), so the DINOv2 patch grid and the geometry patch grid are the same grid by
construction. Features are the last layer's normalised patch tokens (`x_norm_patchtokens`, as in
AnomalyDINO), reshaped row-major to [h, w, D] (token k = i * w + j is patch row i, column j), and
L2-normalised.

Stored precision: features are cached as float16 and re-normalised in float32 when read. The values
used for scoring are the cached ones in both the fresh and the cached path, so a result never depends on
whether a feature was computed in this run or read from disk.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from padgeo.config import FeatureConfig
from padgeo.utils.cache import TensorCache
from padgeo.utils.checks import check_tensor
from padgeo.utils.precision import autocast
from padgeo.utils.provenance import distribution_version

logger = logging.getLogger(__name__)

# Pinned DINOv2 code (torch.hub fetches this exact commit; weights come from the URLs it hard-codes).
DINOV2_HUB_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_STORAGE_DTYPE = torch.float16


class DINOv2Extractor:
    """Computes L2-normalised DINOv2 patch features; the model is loaded on first use.

    Args:
        config: Backbone, precision, device and batch size.
    """

    patch_size = 14

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._model: torch.nn.Module | None = None
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    @property
    def identity(self) -> dict[str, Any]:
        """Everything a cached feature depends on besides the image (part of cache keys)."""
        return {
            "hub_repo": DINOV2_HUB_REPO,
            "backbone": self.config.backbone,
            "precision": self.config.precision,
            "layer": "x_norm_patchtokens",
            "storage": str(_STORAGE_DTYPE),
            # The input tensor comes from VGGT's preprocessing, so its code version is part of the identity.
            "preprocessing": f"vggt {distribution_version('vggt')}",
        }

    def _load_model(self) -> torch.nn.Module:
        if self._model is None:
            logger.info("Loading %s from %s on %s", self.config.backbone, DINOV2_HUB_REPO, self.device)
            model = torch.hub.load(
                DINOV2_HUB_REPO, self.config.backbone, trust_repo=True, skip_validation=True, verbose=False
            )
            self._model = model.eval().to(self.device)
        return self._model

    @torch.inference_mode()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Patch features of one batch, rounded to the storage precision.

        Args:
            images: [B, 3, H, W] float32 model inputs in [0, 1]; H and W multiples of 14.

        Returns:
            [B, H/14, W/14, D] float16 L2-normalised patch features (CPU).
        """
        check_tensor("images", images, (None, 3, None, None), torch.float32, value_range=(0.0, 1.0))
        batch, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"input {height}x{width} is not a multiple of {self.patch_size}")
        model = self._load_model()
        normalised = ((images - self._mean) / self._std).to(self.device)
        with autocast(self.config.precision, self.device):
            tokens = model.forward_features(normalised)["x_norm_patchtokens"]
        h, w = height // self.patch_size, width // self.patch_size
        check_tensor("patch tokens", tokens, (batch, h * w, None), tokens.dtype)
        features = F.normalize(tokens.float(), dim=-1).reshape(batch, h, w, -1)
        return features.to(_STORAGE_DTYPE).cpu()


def unpack_features(stored: torch.Tensor) -> torch.Tensor:
    """Stored float16 features [h, w, D] to the float32 unit vectors used for scoring."""
    return F.normalize(stored.to(torch.float32), dim=-1)


class FeatureStore:
    """Per-image feature cache in front of a `DINOv2Extractor`.

    Args:
        extractor: Computes features on a miss.
        cache: Tensor cache (namespace "features").
    """

    def __init__(self, extractor: DINOv2Extractor, cache: TensorCache) -> None:
        self.extractor = extractor
        self.cache = cache

    def _identity(self, content_hash: str, input_hw: tuple[int, int]) -> dict[str, Any]:
        return {**self.extractor.identity, "image": content_hash, "input_hw": list(input_hw)}

    def lookup(self, content_hash: str, input_hw: tuple[int, int]) -> torch.Tensor | None:
        """Stored features [h, w, D] float16 of one image, or None if not cached."""
        entry = self.cache.load(self._identity(content_hash, input_hw))
        return None if entry is None else entry["features"]

    def compute_and_store(self, content_hashes: list[str], images: torch.Tensor) -> list[torch.Tensor]:
        """Compute features for a batch of images, store each, and return the stored tensors."""
        stored = self.extractor.extract(images)
        input_hw = (images.shape[2], images.shape[3])
        for content_hash, features in zip(content_hashes, stored, strict=True):
            self.cache.save(self._identity(content_hash, input_hw), {"features": features})
        return list(stored)
