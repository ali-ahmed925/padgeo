"""Milestone 4 check on a synthetic scene with exact geometry from the fake backend.

The object is the fake backend's unit sphere; its "appearance" is a smooth function of the 3D surface point
(random Fourier features), so the same surface point looks the same in every view. The defect is a block of
query patches that shows the appearance of a *different* place on the object, a place the references do
see. That is exactly the failure mode the hypothesis is about: `global_nn` finds the look-alike and misses
the defect; `anchored_nn` compares the block with the same surface location and catches it.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from padgeo.config import (
    AggregationConfig,
    AnchoredNNConfig,
    FakeGeometryConfig,
    GlobalNNConfig,
    MaxImageScore,
    RadiusNeighbourhood,
)
from padgeo.geometry.fake import FakeGeometryBackend, look_at_camera
from padgeo.geometry.patches import pool_to_patch_grid
from padgeo.methods.anchored_nn import AnchoredNN
from padgeo.methods.base import PatchSet
from padgeo.methods.global_nn import GlobalNN

SIZE = 518  # real model-input size: 37 x 37 patches
AGG = AggregationConfig(image_score=MaxImageScore(mode="max"), pixel_sigma=0.0)
FAKE = FakeGeometryConfig(backend="fake", point_source="depth")
BLOCK = (slice(17, 20), slice(17, 20))  # 3 x 3 query patches at the centre of the grid


def texture(xyz: torch.Tensor, dim: int = 256, bandwidth: float = 5.0) -> torch.Tensor:
    """Appearance as a smooth function of position: unit-norm random Fourier features of the 3D point."""
    gen = torch.Generator().manual_seed(1234)
    omega = bandwidth * torch.randn(dim, 3, generator=gen)
    phase = 2 * math.pi * torch.rand(dim, generator=gen)
    return F.normalize(torch.cos(xyz @ omega.T + phase), dim=-1)


@pytest.fixture(scope="module")
def scene():
    """Patch geometry of [query, 4 references] seen from nearby viewpoints (frame = query camera)."""
    viewpoints = [(0.0, 0.1), (-0.25, 0.1), (0.25, 0.1), (0.0, 0.35), (0.0, -0.15)]
    cameras = [look_at_camera(az, el, SIZE, SIZE) for az, el in viewpoints]
    geometry = FakeGeometryBackend(FAKE).render(cameras, SIZE, SIZE)
    everything = torch.ones(len(cameras), SIZE, SIZE, dtype=torch.bool)
    return pool_to_patch_grid(geometry, everything, patch_size=14)


def patch_sets(patches, query_features: torch.Tensor) -> tuple[PatchSet, list[PatchSet]]:
    foreground = torch.ones(37, 37, dtype=torch.bool)
    query = PatchSet(query_features, foreground, patches.xyz[0], patches.confidence[0])
    references = [
        PatchSet(texture(patches.xyz[i]), foreground, patches.xyz[i], patches.confidence[i])
        for i in range(1, patches.xyz.shape[0])
    ]
    return query, references


def defective_query_features(patches) -> torch.Tensor:
    """The query's own appearance, except the block, which shows a far-away reference patch's appearance."""
    features = texture(patches.xyz[0])
    block_points = patches.xyz[0][BLOCK].reshape(-1, 3)
    candidates = patches.xyz[1].reshape(-1, 3)
    farthest = torch.cdist(candidates, block_points).min(dim=1).values.argmax()
    assert float(torch.cdist(candidates[farthest][None], block_points).min()) > 0.4
    features[BLOCK] = texture(candidates[farthest])
    return features


def methods(radius: float):
    anchored = AnchoredNN(
        AnchoredNNConfig(
            name="anchored_nn",
            geometry=FAKE,
            neighbourhood=RadiusNeighbourhood(mode="radius", radius=radius),
            min_confidence=0.0,
        ),
        AGG,
    )
    return GlobalNN(GlobalNNConfig(name="global_nn"), AGG), anchored


def test_lookalike_defect_missed_by_global_caught_by_anchored(scene):
    global_nn, anchored_nn = methods(radius=0.05)
    normal = patch_sets(scene, texture(scene.xyz[0]))
    defective = patch_sets(scene, defective_query_features(scene))

    g_normal, a_normal = global_nn.score(*normal, (SIZE, SIZE)), anchored_nn.score(*normal, (SIZE, SIZE))
    g_defect, a_defect = (
        global_nn.score(*defective, (SIZE, SIZE)),
        anchored_nn.score(*defective, (SIZE, SIZE)),
    )

    # The references cover the query's view: almost nothing is unseen.
    assert a_normal.unseen.float().mean() < 0.05
    # Normal query: both methods find a close match everywhere.
    assert g_normal.image_score < 0.1
    assert a_normal.image_score < 0.1
    # Defect: the look-alike exists in the memory, so global_nn scores the block as normal ...
    assert float(g_defect.patch_scores[BLOCK].max()) < 1e-3
    assert g_defect.image_score - g_normal.image_score < 0.05
    # ... while anchored_nn compares it with the same surface location and flags it.
    assert float(a_defect.patch_scores[BLOCK].min()) > 0.5
    assert a_defect.image_score - a_normal.image_score > 0.4


def test_infinite_radius_matches_global_on_the_synthetic_scene(scene):
    global_nn, anchored_nn = methods(radius=math.inf)
    query, references = patch_sets(scene, defective_query_features(scene))
    assert torch.equal(
        anchored_nn.score(query, references, (SIZE, SIZE)).patch_scores,
        global_nn.score(query, references, (SIZE, SIZE)).patch_scores,
    )
