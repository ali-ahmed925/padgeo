"""Ours: nearest neighbour among foreground reference patches that lie close to the query patch in 3D.

All 3D points of one query (query and references) come from one geometry pass and share one world frame
whose scale and orientation are arbitrary. The radius is therefore relative to the diagonal of the
reference point cloud's bounding box, taken in the cloud's principal axes and robust to a few stray
points. The neighbourhood test is then invariant to rotating, translating or uniformly scaling all points.
"""

from __future__ import annotations

import math

import torch

from padgeo.config import AggregationConfig, AnchoredNNConfig, KNNNeighbourhood, Neighbourhood
from padgeo.methods.base import (
    MAX_COSINE_DISTANCE,
    Method,
    NearestNeighbourScores,
    Patches,
    global_nearest_neighbour,
)

# Fraction trimmed from each end of every principal axis when measuring the reference cloud's extent. A
# few stray points (for example a silhouette patch whose depth bleeds towards the background) would
# otherwise inflate the box, and with it the radius, without any visible sign. Fixed a priori (DECISIONS.md).
EXTENT_TRIM = 0.01


def bounding_box_diagonal(points: torch.Tensor) -> float:
    """Diagonal of the bounding box of [P, 3] points in their principal axes, trimmed by `EXTENT_TRIM`.

    Principal axes: an axis-aligned box would change with the orientation of the world frame (rotating a
    square by 45 degrees changes its axis-aligned diagonal from sqrt(2) to 2), and VGGT's world frame is the
    query camera's, whose orientation relative to the object is arbitrary. Trimming: each axis' extent is
    its `EXTENT_TRIM` to `1 - EXTENT_TRIM` quantile range, so up to that fraction of outliers on either side
    cannot stretch the box. Returns 0 for fewer than two points.
    """
    if points.shape[0] < 2:
        return 0.0
    centred = points.to(torch.float64) - points.to(torch.float64).mean(dim=0)
    axes = torch.linalg.svd(centred, full_matrices=False).Vh  # rows: principal directions
    projected = centred @ axes.T
    low = torch.quantile(projected, EXTENT_TRIM, dim=0)
    high = torch.quantile(projected, 1.0 - EXTENT_TRIM, dim=0)
    return float((high - low).norm())


def neighbour_mask(
    query_xyz: torch.Tensor, memory_xyz: torch.Tensor, eligible: torch.Tensor, neighbourhood: Neighbourhood
) -> tuple[torch.Tensor, float | None]:
    """Which memory patches count as "the same place on the surface" for each query patch.

    Args:
        query_xyz: [Q, 3] query patch points (world frame of this query's geometry pass).
        memory_xyz: [M, 3] reference patch points (same frame).
        eligible: [M] bool; reference patches allowed to be neighbours (confident ones).
        neighbourhood: Radius mode: eligible patches within `radius` times the bounding-box diagonal of
            the eligible reference points (Euclidean distance; with fewer than two eligible points the
            diagonal, and so the radius, is 0). kNN mode: the `k` eligible patches nearest in 3D (fewer if
            fewer are eligible; ties broken arbitrarily, which has probability zero for continuous
            coordinates).

    Returns:
        [Q, M] bool neighbour mask, and the absolute radius used (world units; inf for `radius: .inf`;
        None in kNN mode).
    """
    num_query, num_memory = query_xyz.shape[0], memory_xyz.shape[0]
    if isinstance(neighbourhood, KNNNeighbourhood):
        mask = torch.zeros(num_query, num_memory, dtype=torch.bool, device=query_xyz.device)
        k = min(neighbourhood.k, int(eligible.sum()))
        if k == 0 or num_query == 0:
            return mask, None
        distances = torch.cdist(query_xyz, memory_xyz, compute_mode="donot_use_mm_for_euclid_dist")
        distances = distances.masked_fill(~eligible[None, :], math.inf)
        nearest = distances.topk(k, dim=1, largest=False).indices
        return mask.scatter_(1, nearest, True), None
    if math.isinf(neighbourhood.radius):
        return eligible[None, :].expand(num_query, num_memory).clone(), math.inf
    limit = neighbourhood.radius * bounding_box_diagonal(memory_xyz[eligible])
    distances = torch.cdist(query_xyz, memory_xyz, compute_mode="donot_use_mm_for_euclid_dist")
    return (distances <= limit) & eligible[None, :], limit


def median_patch_spacing(points: torch.Tensor) -> float | None:
    """Median distance from each of [P, 3] points to its nearest other point (None if P < 2).

    Applied to the foreground patches of one image, this is the footprint of one patch on the surface,
    which turns a radius in world units into a scale-free number of patches.
    """
    if points.shape[0] < 2:
        return None
    distances = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
    distances.fill_diagonal_(math.inf)
    return float(distances.min(dim=1).values.median())


class AnchoredNN(Method):
    """score(q) = min of d(q, m) over reference patches m that are near q in 3D and confident.

    A query patch is *unseen* if its own confidence is below `min_confidence` (`unseen_low_confidence`),
    or if it is confident but no eligible (confident) reference patch lies within its neighbourhood
    (`unseen_no_neighbour`; `eligible_reference_fraction` tells how many reference patches the confidence
    filter removed). Unseen patches score according to `unseen`: the global_nn score, 0, or the maximum
    cosine distance.
    """

    uses_geometry = True

    def __init__(self, config: AnchoredNNConfig, aggregation: AggregationConfig) -> None:
        super().__init__(aggregation, config.device)
        self.neighbourhood = config.neighbourhood
        self.unseen_mode = config.unseen
        self.min_confidence = config.min_confidence

    def _unseen_scores(self, distances: torch.Tensor) -> torch.Tensor:
        if self.unseen_mode == "fallback_global":
            return global_nearest_neighbour(distances)
        fill = 0.0 if self.unseen_mode == "zero" else MAX_COSINE_DISTANCE
        return torch.full((distances.shape[0],), fill, dtype=distances.dtype, device=distances.device)

    def _nearest_neighbour_scores(
        self, distances: torch.Tensor, query: Patches, memory: Patches
    ) -> NearestNeighbourScores:
        if query.xyz is None or query.confidence is None or memory.xyz is None or memory.confidence is None:
            raise ValueError("anchored_nn needs xyz and confidence for the query and the memory")
        eligible = memory.confidence >= self.min_confidence
        confident_query = query.confidence >= self.min_confidence
        neighbours, radius = neighbour_mask(query.xyz, memory.xyz, eligible, self.neighbourhood)
        anchored = distances.masked_fill(~neighbours, math.inf).min(dim=1).values
        unseen_low_confidence = ~confident_query
        unseen_no_neighbour = confident_query & ~neighbours.any(dim=1)
        unseen = unseen_low_confidence | unseen_no_neighbour
        spacing = median_patch_spacing(query.xyz[confident_query])
        radius_in_patches = None
        if radius is not None and math.isfinite(radius) and spacing:
            radius_in_patches = radius / spacing
        return NearestNeighbourScores(
            scores=torch.where(unseen, self._unseen_scores(distances), anchored),
            unseen_no_neighbour=unseen_no_neighbour,
            unseen_low_confidence=unseen_low_confidence,
            radius_in_query_patches=radius_in_patches,
            eligible_reference_fraction=float(eligible.to(torch.float32).mean()),
        )
