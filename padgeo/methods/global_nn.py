"""Baseline (AnomalyDINO-style): nearest neighbour among all foreground reference patches."""

from __future__ import annotations

import torch

from padgeo.config import AggregationConfig, GlobalNNConfig
from padgeo.methods.base import Method, NearestNeighbourScores, Patches, global_nearest_neighbour


class GlobalNN(Method):
    """score(q) = min over all foreground reference patches m of the cosine distance d(q, m).

    Ignores geometry entirely; no patch is ever unseen.
    """

    uses_geometry = False

    def __init__(self, config: GlobalNNConfig, aggregation: AggregationConfig) -> None:
        super().__init__(aggregation, config.device)

    def _nearest_neighbour_scores(
        self,
        distances: torch.Tensor,
        query: Patches,  # noqa: ARG002 - part of the shared interface; global_nn needs only distances
        memory: Patches,  # noqa: ARG002
    ) -> NearestNeighbourScores:
        never_unseen = torch.zeros(distances.shape[0], dtype=torch.bool, device=distances.device)
        return NearestNeighbourScores(
            scores=global_nearest_neighbour(distances),
            unseen_no_neighbour=never_unseen,
            unseen_low_confidence=never_unseen,
        )
