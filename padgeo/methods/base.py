"""The scoring interface shared by both methods, and every step they share.

A method receives one query and its references as `PatchSet`s, which hold tensors only (no paths, names
or labels), and returns a `MethodOutput`. Everything except the nearest-neighbour rule is implemented once
here:

1. select the query's foreground patches and stack the references' foreground patches into a memory,
2. compute the query-by-memory cosine distance matrix,
3. (method-specific) reduce each row to a patch score and an unseen flag,
4. scatter patch scores back to the grid (background patches score 0),
5. aggregate to an image score and a pixel map (`padgeo.methods.aggregation`).

So switching method changes step 3 and nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from padgeo.config import AggregationConfig
from padgeo.methods.aggregation import aggregate_image_score, make_pixel_map
from padgeo.utils.checks import check_tensor

# Cosine distance 1 - <a, b> of unit vectors lies in [0, 2].
MAX_COSINE_DISTANCE = 2.0


@dataclass(frozen=True)
class PatchSet:
    """One image on the patch grid.

    The grid has h rows and w columns in row-major order; patch (i, j) covers input pixels rows
    14i..14i+13 and columns 14j..14j+13.

    Attributes:
        features: [h, w, D] float32 L2-normalised DINOv2 patch features.
        foreground: [h, w] bool foreground flags (identical for both methods).
        xyz: [h, w, 3] float32 patch 3D points in the world frame shared by all images of the same query
            (anchored_nn only, else None). Finite on foreground patches; NaN where a patch has no
            foreground pixel.
        confidence: [h, w] float32 geometry confidence on the same patches as `xyz`, or None.
    """

    features: torch.Tensor
    foreground: torch.Tensor
    xyz: torch.Tensor | None = None
    confidence: torch.Tensor | None = None

    def __post_init__(self) -> None:
        check_tensor("features", self.features, (None, None, None), torch.float32, finite=True)
        h, w, _ = self.features.shape
        check_tensor("foreground", self.foreground, (h, w), torch.bool)
        if (self.xyz is None) != (self.confidence is None):
            raise ValueError("xyz and confidence must be given together")
        if self.xyz is not None and self.confidence is not None:
            check_tensor("xyz", self.xyz, (h, w, 3), torch.float32)
            check_tensor("confidence", self.confidence, (h, w), torch.float32)
            if not bool(torch.isfinite(self.xyz[self.foreground]).all()):
                raise ValueError("xyz must be finite on foreground patches")
            if not bool(torch.isfinite(self.confidence[self.foreground]).all()):
                raise ValueError("confidence must be finite on foreground patches")


@dataclass(frozen=True)
class Patches:
    """Foreground patches flattened to rows (query patches, or the reference memory).

    Attributes:
        features: [P, D] float32.
        xyz: [P, 3] float32 or None.
        confidence: [P] float32 or None.
    """

    features: torch.Tensor
    xyz: torch.Tensor | None
    confidence: torch.Tensor | None


@dataclass(frozen=True)
class NearestNeighbourScores:
    """Per-row result of the method-specific rule.

    Attributes:
        scores: [Q] float32 patch anomaly scores.
        unseen_no_neighbour: [Q] bool; query patch confident but no reference patch near it in 3D.
        unseen_low_confidence: [Q] bool; query patch geometry confidence below the threshold.
    """

    scores: torch.Tensor
    unseen_no_neighbour: torch.Tensor
    unseen_low_confidence: torch.Tensor


@dataclass(frozen=True)
class MethodOutput:
    """Scores of one query (CPU tensors).

    Attributes:
        patch_scores: [h, w] float32; background patches are 0.
        unseen_no_neighbour: [h, w] bool (all False for global_nn).
        unseen_low_confidence: [h, w] bool (all False for global_nn).
        image_score: Image-level anomaly score.
        pixel_map: [H0, W0] float32 anomaly map at the original image size.
    """

    patch_scores: torch.Tensor
    unseen_no_neighbour: torch.Tensor
    unseen_low_confidence: torch.Tensor
    image_score: float
    pixel_map: np.ndarray

    @property
    def unseen(self) -> torch.Tensor:
        """[h, w] bool union of both unseen reasons."""
        return self.unseen_no_neighbour | self.unseen_low_confidence


def foreground_patches(patch_set: PatchSet, device: torch.device) -> Patches:
    """Flatten the foreground patches of one image into rows (row-major grid order)."""
    mask = patch_set.foreground
    return Patches(
        features=patch_set.features[mask].to(device),
        xyz=None if patch_set.xyz is None else patch_set.xyz[mask].to(device),
        confidence=None if patch_set.confidence is None else patch_set.confidence[mask].to(device),
    )


def build_memory(references: Sequence[PatchSet], device: torch.device) -> Patches:
    """Stack the foreground patches of all references, in reference order.

    Raises:
        ValueError: If there are no references, or they have no foreground patch at all.
    """
    if not references:
        raise ValueError("at least one reference is required")
    parts = [foreground_patches(reference, device) for reference in references]
    features = torch.cat([part.features for part in parts])
    if features.shape[0] == 0:
        raise ValueError("the references contain no foreground patch")
    has_geometry = [part.xyz is not None for part in parts]
    if any(has_geometry) and not all(has_geometry):
        raise ValueError("either all references carry geometry or none does")
    if not all(has_geometry):
        return Patches(features=features, xyz=None, confidence=None)
    return Patches(
        features=features,
        xyz=torch.cat([part.xyz for part in parts if part.xyz is not None]),
        confidence=torch.cat([part.confidence for part in parts if part.confidence is not None]),
    )


def cosine_distances(query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    """[Q, M] cosine distances 1 - <q, m> between L2-normalised rows, clamped to [0, 2]."""
    return (1.0 - query @ memory.T).clamp_(0.0, MAX_COSINE_DISTANCE)


def global_nearest_neighbour(distances: torch.Tensor) -> torch.Tensor:
    """[Q] distance from each query patch to its nearest neighbour among all memory patches.

    This is the global_nn score, and also anchored_nn's `fallback_global` value for unseen patches.
    """
    return distances.min(dim=1).values


class Method(ABC):
    """Score a query against references; subclasses implement only the nearest-neighbour rule.

    Args:
        aggregation: Image-score and pixel-map settings (shared by both methods).
        device: Torch device for distance computations.
    """

    uses_geometry: bool = False

    def __init__(self, aggregation: AggregationConfig, device: str) -> None:
        self.aggregation = aggregation
        self.device = torch.device(device)

    @abstractmethod
    def _nearest_neighbour_scores(
        self, distances: torch.Tensor, query: Patches, memory: Patches
    ) -> NearestNeighbourScores:
        """Reduce the [Q, M] distance matrix to per-query-patch scores and unseen flags."""

    def score(
        self, query: PatchSet, references: Sequence[PatchSet], image_size: tuple[int, int]
    ) -> MethodOutput:
        """Score one query.

        Args:
            query: The query's patch set.
            references: The reference patch sets (order does not affect the result).
            image_size: (H0, W0) of the original query image; the pixel map has this size.

        Returns:
            Patch scores, unseen masks, image score and pixel map.

        Raises:
            ValueError: If geometry is missing for a geometry-based method, or shapes are inconsistent.
        """
        if self.uses_geometry and (query.xyz is None or any(ref.xyz is None for ref in references)):
            raise ValueError(f"{type(self).__name__} needs xyz/confidence on the query and all references")
        h, w, dim = query.features.shape
        if any(ref.features.shape[-1] != dim for ref in references):
            raise ValueError("query and references must have the same feature dimension")

        query_patches = foreground_patches(query, self.device)
        memory = build_memory(references, self.device)
        distances = cosine_distances(query_patches.features, memory.features)
        nearest = self._nearest_neighbour_scores(distances, query_patches, memory)
        num_query = query_patches.features.shape[0]
        check_tensor("scores", nearest.scores, (num_query,), torch.float32, finite=True)

        patch_scores = torch.zeros(h, w, dtype=torch.float32)
        patch_scores[query.foreground] = nearest.scores.cpu()
        no_neighbour = torch.zeros(h, w, dtype=torch.bool)
        no_neighbour[query.foreground] = nearest.unseen_no_neighbour.cpu()
        low_confidence = torch.zeros(h, w, dtype=torch.bool)
        low_confidence[query.foreground] = nearest.unseen_low_confidence.cpu()

        return MethodOutput(
            patch_scores=patch_scores,
            unseen_no_neighbour=no_neighbour,
            unseen_low_confidence=low_confidence,
            image_score=aggregate_image_score(patch_scores, self.aggregation.image_score),
            pixel_map=make_pixel_map(patch_scores, image_size, self.aggregation.pixel_sigma),
        )
