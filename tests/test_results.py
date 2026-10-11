"""Per-class coverage summary computed from query records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from padgeo.data.base import ImageRef
from padgeo.eval.results import QueryRecord, coverage_summary
from padgeo.methods.base import MethodOutput


def record(
    foreground: int, no_neighbour: int, low_confidence: int, radius: float | None, eligible: float | None
):
    grid = torch.zeros(4, 4, dtype=torch.bool)
    fg = grid.clone().view(-1)
    fg[:foreground] = True
    nn = grid.clone().view(-1)
    nn[:no_neighbour] = True
    lc = grid.clone().view(-1)
    lc[no_neighbour : no_neighbour + low_confidence] = True
    output = MethodOutput(
        patch_scores=torch.zeros(4, 4),
        unseen_no_neighbour=nn.view(4, 4),
        unseen_low_confidence=lc.view(4, 4),
        image_score=0.0,
        pixel_map=np.zeros((8, 8), dtype=np.float32),
        radius_in_query_patches=radius,
        eligible_reference_fraction=eligible,
    )
    ref = ImageRef(name="q", path=Path("q"), content_hash="q")
    return QueryRecord(ref, (), fg.view(4, 4), output, 0.0, 0.0, None, None)


def test_coverage_summary_pools_patches_and_counts_degenerate_queries():
    summary = coverage_summary(
        [record(10, 2, 1, 2.0, 0.5), record(6, 0, 3, None, 1.0), record(0, 0, 0, None, None)]
    )
    assert summary["foreground_patches"] == 16
    assert summary["unseen_fraction"] == pytest.approx(6 / 16)
    assert summary["unseen_no_neighbour_fraction"] == pytest.approx(2 / 16)
    assert summary["unseen_low_confidence_fraction"] == pytest.approx(4 / 16)
    assert summary["empty_foreground_queries"] == 1
    assert summary["radius_in_query_patches_median"] == 2.0  # None values are skipped
    assert summary["eligible_reference_fraction_median"] == pytest.approx(0.75)


def test_coverage_summary_for_global_nn_has_no_geometry_diagnostics():
    summary = coverage_summary([record(5, 0, 0, None, None)])
    assert summary["radius_in_query_patches_median"] is None
    assert summary["eligible_reference_fraction_median"] is None
