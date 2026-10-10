"""Image score and pixel map (shared by both methods)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from padgeo.config import MaxImageScore, TopPercentImageScore
from padgeo.methods.aggregation import aggregate_image_score, make_pixel_map


def test_top_percent_uses_full_grid_count():
    # 37 x 37 = 1369 patches: k = floor(13.69) = 13 highest, as AnomalyDINO's mean_top1p.
    scores = torch.zeros(37, 37)
    scores.view(-1)[:20] = torch.arange(20, dtype=torch.float32)
    expected = float(torch.arange(7, 20, dtype=torch.float32).mean())  # the 13 largest: 7..19
    got = aggregate_image_score(scores, TopPercentImageScore(mode="top_percent", percent=1.0))
    assert got == pytest.approx(expected)


def test_top_percent_minimum_one_patch_and_max():
    scores = torch.tensor([[0.1, 0.7], [0.3, 0.2]])
    assert aggregate_image_score(
        scores, TopPercentImageScore(mode="top_percent", percent=1.0)
    ) == pytest.approx(0.7)
    assert aggregate_image_score(
        scores, TopPercentImageScore(mode="top_percent", percent=50.0)
    ) == pytest.approx(0.5)
    assert aggregate_image_score(scores, MaxImageScore(mode="max")) == pytest.approx(0.7)


def test_top_percent_includes_background_zeros_only_when_foreground_is_small():
    scores = torch.zeros(10, 10)
    scores[0, 0] = 1.0  # one foreground patch; k = max(1, floor(100 * 5 / 100)) = 5
    got = aggregate_image_score(scores, TopPercentImageScore(mode="top_percent", percent=5.0))
    assert got == pytest.approx(0.2)


def test_pixel_map_places_patch_scores_on_the_right_pixels():
    patch_scores = torch.zeros(37, 37)
    patch_scores[5, 30] = 1.0  # patch row 5, column 30
    pixel_map = make_pixel_map(patch_scores, (400, 400), sigma=0.0)
    assert pixel_map.shape == (400, 400)
    assert pixel_map.dtype == np.float32
    row, col = np.unravel_index(np.argmax(pixel_map), pixel_map.shape)
    # Centre of patch (5, 30) in original pixels: ((5 + 0.5) * 400/37, (30 + 0.5) * 400/37) = (59.5, 329.7).
    assert abs(row - 59.5) <= 1.0
    assert abs(col - 329.7) <= 1.0


def test_pixel_map_non_square_and_smoothing():
    patch_scores = torch.zeros(4, 6)
    patch_scores[2, 3] = 1.0
    sharp = make_pixel_map(patch_scores, (40, 60), sigma=0.0)
    smooth = make_pixel_map(patch_scores, (40, 60), sigma=4.0)
    assert sharp.shape == smooth.shape == (40, 60)
    assert smooth.max() < sharp.max()  # smoothing spreads the peak
    assert smooth.sum() == pytest.approx(sharp.sum(), rel=1e-3)  # reflect-mode Gaussian preserves mass here


def test_pixel_map_matches_bilinear_interpolation_convention():
    # align_corners=False: a linear ramp across patch centres stays linear between them.
    patch_scores = torch.arange(4, dtype=torch.float32).repeat(4, 1)  # columns 0,1,2,3
    pixel_map = make_pixel_map(patch_scores, (8, 8), sigma=0.0)
    # Output column x has source coordinate (x + 0.5) / 2 - 0.5, clamped at the borders.
    expected_cols = np.clip((np.arange(8) + 0.5) / 2 - 0.5, 0, 3)
    assert np.allclose(pixel_map[0], expected_cols, atol=1e-6)
