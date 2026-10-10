"""Ours: nearest neighbour among foreground reference patches that lie close to the query patch in 3D.

All 3D points of one query (query and references) come from one geometry pass and share one world frame
whose scale and orientation are arbitrary. The radius is therefore relative to the diagonal of the
reference point cloud's bounding box, taken in the cloud's principal axes. The neighbourhood test is then
invariant to rotating, translating or uniformly scaling all points.
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


def bounding_box_diagonal(points: torch.Tensor) -> float:
    """Diagonal of the bounding box of [P, 3] points in their principal axes (0 if P == 0).

    An axis-aligned box would change with the orientation of the world frame (rotating a square by 45
    degrees changes its axis-aligned diagonal from sqrt(2) to 2), and VGGT's world frame is the query
    camera's, whose orientation relative to the object is arbitrary. In the principal axes of the points,
    the box, and hence the radius, is invariant to rotating, translating or uniformly scaling the points.
    """
    if points.shape[0] == 0:
        return 0.0
    centred = points.to(torch.float64) - points.to(torch.float64).mean(dim=0)
    axes = torch.linalg.svd(centred, full_matrices=False).Vh  # rows: principal directions
    projected = centred @ axes.T
    return float((projected.max(dim=0).values - projected.min(dim=0).values).norm())


def neighbour_mask(
    query_xyz: torch.Tensor, memory_xyz: torch.Tensor, eligible: torch.Tensor, neighbourhood: Neighbourhood
) -> torch.Tensor:
    """Which memory patches count as "the same place on the surface" for each query patch.

    Args:
        query_xyz: [Q, 3] query patch points (world frame of this query's geometry pass).
        memory_xyz: [M, 3] reference patch points (same frame).
        eligible: [M] bool; reference patches allowed to be neighbours (confident ones).
        neighbourhood: Radius mode: eligible patches within `radius` times the bounding-box diagonal of
            the eligible reference points, Euclidean distance. kNN mode: the `k` eligible patches nearest in
            3D (fewer if fewer are eligible; ties broken arbitrarily, which has probability zero for
            continuous coordinates).

    Returns:
        [Q, M] bool neighbour mask.
    """
    num_query, num_memory = query_xyz.shape[0], memory_xyz.shape[0]
    if isinstance(neighbourhood, KNNNeighbourhood):
        mask = torch.zeros(num_query, num_memory, dtype=torch.bool, device=query_xyz.device)
        k = min(neighbourhood.k, int(eligible.sum()))
        if k == 0 or num_query == 0:
            return mask
        distances = torch.cdist(query_xyz, memory_xyz, compute_mode="donot_use_mm_for_euclid_dist")
        distances = distances.masked_fill(~eligible[None, :], math.inf)
        nearest = distances.topk(k, dim=1, largest=False).indices
        return mask.scatter_(1, nearest, True)
    if math.isinf(neighbourhood.radius):
        return eligible[None, :].expand(num_query, num_memory).clone()
    limit = neighbourhood.radius * bounding_box_diagonal(memory_xyz[eligible])
    distances = torch.cdist(query_xyz, memory_xyz, compute_mode="donot_use_mm_for_euclid_dist")
    return (distances <= limit) & eligible[None, :]


class AnchoredNN(Method):
    """score(q) = min of d(q, m) over reference patches m that are near q in 3D and confident.

    A query patch is *unseen* if its own confidence is below `min_confidence` (`unseen_low_confidence`)
    or if no eligible reference patch is near it (`unseen_no_neighbour`). Unseen patches score according
    to `unseen`: the global_nn score, 0, or the maximum cosine distance.
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
        neighbours = neighbour_mask(query.xyz, memory.xyz, eligible, self.neighbourhood)
        anchored = distances.masked_fill(~neighbours, math.inf).min(dim=1).values
        unseen_low_confidence = ~confident_query
        unseen_no_neighbour = confident_query & ~neighbours.any(dim=1)
        unseen = unseen_low_confidence | unseen_no_neighbour
        return NearestNeighbourScores(
            scores=torch.where(unseen, self._unseen_scores(distances), anchored),
            unseen_no_neighbour=unseen_no_neighbour,
            unseen_low_confidence=unseen_low_confidence,
        )
