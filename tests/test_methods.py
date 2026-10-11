"""Scoring methods: hand-computed toy cases, unseen modes, confidence filtering, exactness and invariances."""

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
    KNNNeighbourhood,
    MaxImageScore,
    RadiusNeighbourhood,
)
from padgeo.methods import build_method
from padgeo.methods.anchored_nn import AnchoredNN, bounding_box_diagonal, neighbour_mask
from padgeo.methods.base import MAX_COSINE_DISTANCE, PatchSet, cosine_distances
from padgeo.methods.global_nn import GlobalNN

AGG = AggregationConfig(image_score=MaxImageScore(mode="max"), pixel_sigma=0.0)
FAKE = FakeGeometryConfig(backend="fake")


def anchored(
    radius: float = 0.05, unseen: str = "fallback_global", min_confidence: float = 0.0, k: int | None = None
):
    neighbourhood = (
        KNNNeighbourhood(mode="knn", k=k) if k else RadiusNeighbourhood(mode="radius", radius=radius)
    )
    config = AnchoredNNConfig(
        name="anchored_nn",
        geometry=FAKE,
        neighbourhood=neighbourhood,
        unseen=unseen,
        min_confidence=min_confidence,
    )
    return AnchoredNN(config, AGG)


def unit(*vectors: list[float]) -> torch.Tensor:
    return F.normalize(torch.tensor(vectors, dtype=torch.float32), dim=-1)


def patch_set(
    features: torch.Tensor,
    xyz: torch.Tensor | None = None,
    conf: torch.Tensor | None = None,
    foreground: torch.Tensor | None = None,
) -> PatchSet:
    """A 1 x P grid from P feature rows."""
    num = features.shape[0]
    return PatchSet(
        features=features.reshape(1, num, -1),
        foreground=torch.ones(1, num, dtype=torch.bool) if foreground is None else foreground.reshape(1, num),
        xyz=None if xyz is None else xyz.reshape(1, num, 3).float(),
        confidence=None if conf is None else conf.reshape(1, num).float(),
    )


# --- shared pieces ---------------------------------------------------------------------------------


def test_cosine_distances_hand_computed():
    query = unit([1, 0], [0, 1])
    memory = unit([1, 0], [1, 1], [-1, 0])
    expected = torch.tensor([[0.0, 1 - 1 / math.sqrt(2), 2.0], [1.0, 1 - 1 / math.sqrt(2), 1.0]])
    assert torch.allclose(cosine_distances(query, memory), expected, atol=1e-6)


def test_bounding_box_diagonal_in_principal_axes():
    # Corners of a 2 x 3 x 6 cuboid: the principal axes are the cuboid's edges, so the diagonal is 7.
    corners = torch.tensor([[x, y, z] for x in (0.0, 2.0) for y in (0.0, 3.0) for z in (0.0, 6.0)])
    assert bounding_box_diagonal(corners) == pytest.approx(7.0)
    # Rotated and shifted, the diagonal is unchanged (an axis-aligned box would grow).
    rotation = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(1)))[0]
    moved = corners @ rotation.T + torch.tensor([10.0, -4.0, 2.0])
    assert bounding_box_diagonal(moved) == pytest.approx(7.0, abs=1e-5)
    axis_aligned = float((moved.max(0).values - moved.min(0).values).norm())
    assert axis_aligned > 7.1
    assert bounding_box_diagonal(torch.zeros(0, 3)) == 0.0
    assert bounding_box_diagonal(torch.ones(1, 3)) == 0.0


# --- global_nn -------------------------------------------------------------------------------------


def test_global_nn_hand_computed():
    query = patch_set(unit([1, 0], [0, 1], [1, 1]))
    refs = [patch_set(unit([1, 0.1])), patch_set(unit([-1, 0], [0.2, 1]))]
    out = GlobalNN(GlobalNNConfig(name="global_nn"), AGG).score(query, refs, (1, 3))
    memory = unit([1, 0.1], [-1, 0], [0.2, 1])
    expected = cosine_distances(unit([1, 0], [0, 1], [1, 1]), memory).min(dim=1).values
    assert torch.allclose(out.patch_scores[0], expected)
    assert out.image_score == pytest.approx(float(expected.max()))
    assert not out.unseen.any()


def test_background_patches_score_zero_and_are_not_in_memory():
    features = unit([1, 0], [0, 1], [-1, 0])
    query = patch_set(features, foreground=torch.tensor([True, False, True]))
    # Reference: its background patch is identical to the query's third patch; it must not be matched.
    ref = patch_set(unit([1, 0], [-1, 0]), foreground=torch.tensor([True, False]))
    out = GlobalNN(GlobalNNConfig(name="global_nn"), AGG).score(query, [ref], (1, 3))
    assert out.patch_scores[0, 1] == 0.0
    assert out.patch_scores[0, 2] == pytest.approx(2.0)  # opposite of the only foreground reference patch


def test_empty_query_foreground_scores_zero():
    query = patch_set(unit([1, 0]), foreground=torch.tensor([False]))
    out = GlobalNN(GlobalNNConfig(name="global_nn"), AGG).score(query, [patch_set(unit([1, 0]))], (2, 2))
    assert out.image_score == 0.0
    assert float(out.pixel_map.max()) == 0.0


def test_references_without_foreground_raise():
    ref = patch_set(unit([1, 0]), foreground=torch.tensor([False]))
    with pytest.raises(ValueError, match="no foreground"):
        GlobalNN(GlobalNNConfig(name="global_nn"), AGG).score(patch_set(unit([1, 0])), [ref], (1, 1))


# --- anchored_nn -----------------------------------------------------------------------------------


def toy_anchored_inputs():
    """Two query patches; the reference has a look-alike of query patch 0 far away in 3D."""
    query = patch_set(
        unit([1, 0], [0, 1]), xyz=torch.tensor([[0.0, 0, 0], [1.0, 0, 0]]), conf=torch.tensor([5.0, 5.0])
    )
    ref = patch_set(
        unit([0, 1], [1, 0], [0.6, 0.8]),
        xyz=torch.tensor([[0.01, 0, 0], [1.0, 0.0, 0.0], [1.0, 0.01, 0.0]]),
        conf=torch.tensor([5.0, 5.0, 5.0]),
    )
    return query, [ref]


def test_anchored_radius_hand_computed():
    query, refs = toy_anchored_inputs()
    # Reference cloud bbox diagonal = |(0.99, 0.01, 0)|; radius 0.1 of it is about 0.099.
    out = anchored(radius=0.1).score(query, refs, (1, 2))
    # Patch 0 (at x=0) has one neighbour, ref patch 0 (feature [0,1]): distance 1. Globally it would be 0.
    assert out.patch_scores[0, 0] == pytest.approx(1.0)
    # Patch 1 (at x=1) neighbours ref patches 1 and 2; nearest feature [0.6, 0.8]: distance 1 - 0.8 = 0.2.
    assert out.patch_scores[0, 1] == pytest.approx(0.2, abs=1e-6)
    assert not out.unseen.any()
    global_out = GlobalNN(GlobalNNConfig(name="global_nn"), AGG).score(query, refs, (1, 2))
    assert global_out.patch_scores[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_anchored_knn_hand_computed():
    query, refs = toy_anchored_inputs()
    out = anchored(k=1).score(query, refs, (1, 2))
    assert out.patch_scores[0, 0] == pytest.approx(1.0)  # nearest in 3D is ref patch 0
    assert out.patch_scores[0, 1] == pytest.approx(1.0)  # nearest in 3D is ref patch 1, feature [1, 0]
    out_k2 = anchored(k=2).score(query, refs, (1, 2))
    assert out_k2.patch_scores[0, 1] == pytest.approx(0.2, abs=1e-6)


@pytest.mark.parametrize(
    ("mode", "expected"), [("fallback_global", 0.0), ("zero", 0.0), ("max", MAX_COSINE_DISTANCE)]
)
def test_unseen_modes(mode, expected):
    # Query patch 0 is far from every reference point: unseen.
    query = patch_set(
        unit([1, 0], [0, 1]), xyz=torch.tensor([[10.0, 0, 0], [0.0, 0, 0]]), conf=torch.tensor([5.0, 5.0])
    )
    ref = patch_set(
        unit([1, 0], [0, 1]), xyz=torch.tensor([[0.0, 0, 0], [0.0, 1, 0]]), conf=torch.tensor([5.0, 5.0])
    )
    out = anchored(radius=0.1, unseen=mode).score(query, [ref], (1, 2))
    assert out.unseen_no_neighbour.tolist() == [[True, False]]
    assert not out.unseen_low_confidence.any()
    assert out.patch_scores[0, 0] == pytest.approx(expected, abs=1e-6)
    # Seen patch 1 neighbours only ref patch 0 (feature [1, 0]): distance 1.
    assert out.patch_scores[0, 1] == pytest.approx(1.0)


def test_fallback_global_differs_from_zero_when_global_score_is_positive():
    query = patch_set(unit([1, 1]), xyz=torch.tensor([[10.0, 0, 0]]), conf=torch.tensor([5.0]))
    ref = patch_set(
        unit([1, 0], [0, 1]), xyz=torch.tensor([[0.0, 0, 0], [0.0, 1, 0]]), conf=torch.tensor([5.0, 5.0])
    )
    fallback = anchored(radius=0.1, unseen="fallback_global").score(query, [ref], (1, 1))
    zero = anchored(radius=0.1, unseen="zero").score(query, [ref], (1, 1))
    assert fallback.patch_scores[0, 0] == pytest.approx(1 - 1 / math.sqrt(2), abs=1e-6)
    assert zero.patch_scores[0, 0] == 0.0


def test_confidence_filter_query_and_reference():
    query = patch_set(unit([1, 0], [0, 1]), xyz=torch.zeros(2, 3), conf=torch.tensor([1.0, 9.0]))
    # Ref patch 1 is a perfect match for query patch 1 and sits next to it, but has low confidence.
    ref = patch_set(
        unit([1, 0], [0, 1], [-1, 0]),
        xyz=torch.tensor([[0.0, 0, 0], [0.0, 0, 0.001], [1.0, 0, 0]]),
        conf=torch.tensor([9.0, 1.0, 9.0]),
    )
    # Eligible reference cloud: ref patches 0 and 2, diagonal 1, so the neighbourhood limit is 0.5.
    out = anchored(radius=0.5, min_confidence=5.0, unseen="max").score(query, [ref], (1, 2))
    # Query patch 0 has low confidence: unseen (max), regardless of neighbours.
    assert out.unseen_low_confidence.tolist() == [[True, False]]
    assert out.patch_scores[0, 0] == pytest.approx(MAX_COSINE_DISTANCE)
    # Query patch 1 ([0, 1]) may only match confident ref patch 0 ([1, 0]): distance 1, not 0.
    assert out.patch_scores[0, 1] == pytest.approx(1.0)
    unfiltered = anchored(radius=0.5, min_confidence=0.0, unseen="max").score(query, [ref], (1, 2))
    assert unfiltered.patch_scores[0, 1] == pytest.approx(0.0, abs=1e-6)


def random_scene(seed: int, num_refs: int = 3, grid: int = 6, dim: int = 8):
    gen = torch.Generator().manual_seed(seed)

    def one() -> PatchSet:
        return PatchSet(
            features=F.normalize(torch.randn(grid, grid, dim, generator=gen), dim=-1),
            foreground=torch.rand(grid, grid, generator=gen) > 0.3,
            xyz=torch.randn(grid, grid, 3, generator=gen),
            confidence=1.0 + 9.0 * torch.rand(grid, grid, generator=gen),
        )

    return one(), [one() for _ in range(num_refs)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_infinite_radius_reproduces_global_exactly(seed):
    query, refs = random_scene(seed)
    global_out = GlobalNN(GlobalNNConfig(name="global_nn"), AGG).score(query, refs, (12, 12))
    anchored_out = anchored(radius=math.inf, min_confidence=0.0).score(query, refs, (12, 12))
    assert torch.equal(anchored_out.patch_scores, global_out.patch_scores)
    assert anchored_out.image_score == global_out.image_score
    assert not anchored_out.unseen.any()


def test_infinite_radius_with_confidence_filter_differs_only_through_confidence():
    query, refs = random_scene(3)
    confident = [PatchSet(r.features, r.foreground, r.xyz, torch.full_like(r.confidence, 9.0)) for r in refs]
    query = PatchSet(query.features, query.foreground, query.xyz, torch.full_like(query.confidence, 9.0))
    global_out = GlobalNN(GlobalNNConfig(name="global_nn"), AGG).score(query, confident, (6, 6))
    anchored_out = anchored(radius=math.inf, min_confidence=5.0).score(query, confident, (6, 6))
    assert torch.equal(anchored_out.patch_scores, global_out.patch_scores)


@pytest.mark.parametrize("method_kwargs", [{"radius": 0.2}, {"k": 4}])
def test_anchored_invariant_to_similarity_transform(method_kwargs):
    query, refs = random_scene(4)
    rotation = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(0)))[0]
    scale, shift = 3.7, torch.tensor([5.0, -2.0, 1.0])

    def move(p: PatchSet) -> PatchSet:
        assert p.xyz is not None
        return PatchSet(p.features, p.foreground, scale * p.xyz @ rotation.T + shift, p.confidence)

    method = anchored(**method_kwargs)
    before = method.score(query, refs, (6, 6))
    after = method.score(move(query), [move(r) for r in refs], (6, 6))
    assert torch.allclose(before.patch_scores, after.patch_scores, atol=1e-6)
    assert torch.equal(before.unseen, after.unseen)


@pytest.mark.parametrize("method", ["global", "radius", "knn"])
def test_reference_order_does_not_change_scores(method):
    query, refs = random_scene(5, num_refs=4)
    scorer = {
        "global": GlobalNN(GlobalNNConfig(name="global_nn"), AGG),
        "radius": anchored(radius=0.3),
        "knn": anchored(k=5),
    }[method]
    first = scorer.score(query, refs, (6, 6))
    second = scorer.score(query, refs[::-1], (6, 6))
    # A minimum over a set is order-invariant; the tolerance only allows BLAS to sum dot products in a
    # different order for permuted columns. (The pipeline canonicalises reference order, and the
    # integration test checks bitwise equality there.)
    assert torch.allclose(first.patch_scores, second.patch_scores, atol=1e-6)
    assert torch.equal(first.unseen, second.unseen)


def test_neighbour_mask_respects_eligibility_and_k():
    query = torch.tensor([[0.0, 0.0, 0.0]])
    memory = torch.tensor([[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])
    eligible = torch.tensor([False, True, True])
    knn, knn_radius = neighbour_mask(query, memory, eligible, KNNNeighbourhood(mode="knn", k=1))
    assert knn.tolist() == [[False, True, False]]
    assert knn_radius is None  # kNN has no radius
    knn_large, _ = neighbour_mask(query, memory, eligible, KNNNeighbourhood(mode="knn", k=10))
    assert knn_large.tolist() == [[False, True, True]]
    radius_mask, radius = neighbour_mask(
        query, memory, eligible, RadiusNeighbourhood(mode="radius", radius=math.inf)
    )
    assert radius_mask.tolist() == [[False, True, True]]
    assert radius == math.inf


def test_anchored_requires_geometry():
    query, refs = random_scene(6)
    no_geometry = PatchSet(query.features, query.foreground)
    with pytest.raises(ValueError, match="needs xyz"):
        anchored().score(no_geometry, refs, (6, 6))


def test_build_method_dispatches_on_config():
    assert isinstance(build_method(GlobalNNConfig(name="global_nn"), AGG), GlobalNN)
    anchored_config = AnchoredNNConfig(
        name="anchored_nn",
        geometry=FAKE,
        neighbourhood=RadiusNeighbourhood(mode="radius", radius=0.05),
        min_confidence=1.5,
    )
    assert isinstance(build_method(anchored_config, AGG), AnchoredNN)


def test_bounding_box_diagonal_ignores_a_few_stray_points():
    gen = torch.Generator().manual_seed(0)
    cloud = torch.rand(300, 3, generator=gen) * torch.tensor([1.0, 0.6, 0.4])
    with_outlier = torch.cat([cloud, torch.tensor([[4.0, 0.3, 0.2]])])
    clean, stretched = bounding_box_diagonal(cloud), bounding_box_diagonal(with_outlier)
    assert stretched == pytest.approx(clean, rel=0.05)  # an untrimmed box would be about 2.6 times larger


def test_radius_scales_with_the_diagonal_of_eligible_reference_points_only():
    # Eligible references span x in [0, 1]; an ineligible far point at x = 50 must not enlarge the radius.
    query = patch_set(unit([1, 0]), xyz=torch.tensor([[0.5, 0, 0]]), conf=torch.tensor([9.0]))
    ref = patch_set(
        unit([0, 1], [0, 1], [1, 0]),
        xyz=torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [50.0, 0, 0]]),
        conf=torch.tensor([9.0, 9.0, 1.0]),
    )
    out = anchored(radius=0.4, min_confidence=5.0, unseen="max").score(query, [ref], (1, 1))
    # radius 0.4 x diagonal 1 = 0.4 < 0.5: no eligible neighbour (with the far point counted, 0.4 x 50 = 20).
    assert out.unseen_no_neighbour.tolist() == [[True]]
    assert out.eligible_reference_fraction == pytest.approx(2 / 3)


def test_anchored_reports_radius_in_query_patches():
    # Query patches 0.1 apart: the patch spacing is 0.1. References are the corners of a 2 x 1 rectangle
    # (distinct principal axes, so the box is unambiguous): diagonal sqrt(5).
    query_xyz = torch.tensor([[0.1 * i, 0.0, 0.0] for i in range(5)])
    query = patch_set(F.normalize(torch.randn(5, 4), dim=-1), xyz=query_xyz, conf=torch.full((5,), 9.0))
    corners = torch.tensor([[x, y, 0.0] for x in (0.0, 2.0) for y in (0.0, 1.0)])
    ref = patch_set(F.normalize(torch.randn(4, 4), dim=-1), xyz=corners, conf=torch.full((4,), 9.0))
    out = anchored(radius=0.1).score(query, [ref], (1, 5))
    assert out.radius_in_query_patches == pytest.approx(0.1 * math.sqrt(5) / 0.1, rel=1e-3)
    assert out.eligible_reference_fraction == 1.0
    global_out = GlobalNN(GlobalNNConfig(name="global_nn"), AGG).score(query, [ref], (1, 5))
    assert global_out.radius_in_query_patches is None
    assert global_out.eligible_reference_fraction is None
