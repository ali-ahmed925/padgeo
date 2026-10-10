"""Cross-seed tables: comparability checks, statistics and rendering, from synthetic result dicts."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from padgeo.eval.summary import (
    RunSet,
    check_comparable,
    load_reference_numbers,
    load_run_set,
    mean_std,
    metric_values,
    render_markdown,
)

REPO = Path(__file__).resolve().parents[1]


def fake_result(method: str, seed: int, value: float) -> dict:
    metrics = {"image_auroc": value, "pixel_auroc": value, "aupro": value}
    return {
        "experiment": f"exp_{method}",
        "method": method,
        "seed": seed,
        "classes": ["03Mallard"],
        "metrics": {"per_class": {"03Mallard": dict(metrics)}, "mean": dict(metrics)},
        "queries": {
            "03Mallard": {"total": 10, "defective": 8, "defective_with_empty_mask": 0, "per_type": {}}
        },
        "coverage": {
            "03Mallard": {
                "foreground_patches": 100,
                "unseen_fraction": 0.1,
                "unseen_no_neighbour_fraction": 0.05,
                "unseen_low_confidence_fraction": 0.05,
                "empty_foreground_queries": 0,
            }
        },
        "config": {
            "data": {"dataset": "mad_sim", "num_references": 4, "sampling": "per_query", "root": "r"},
            "features": {"backbone": "dinov2_vitl14"},
            "foreground": {"pixel_threshold": 0.98},
            "aggregation": {"pixel_sigma": 4.0},
        },
        "provenance": {
            "git": {"commit": "a" * 40, "dirty": seed == 1},
            "hardware": {"gpus": [{"name": "GPU"}]},
        },
    }


def run_set(method: str, values: list[float]) -> RunSet:
    return RunSet(
        name=f"exp_{method}",
        method=method,
        results=tuple(fake_result(method, seed, v) for seed, v in enumerate(values)),
    )


def test_mean_std_uses_sample_standard_deviation():
    mean, std = mean_std([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)  # ddof = 1
    assert math.isnan(mean_std([5.0])[1])


def test_metric_values_per_class_and_mean():
    rs = run_set("global_nn", [0.5, 0.7])
    assert metric_values(rs, "aupro", "03Mallard") == [0.5, 0.7]
    assert metric_values(rs, "aupro", None) == [0.5, 0.7]


@pytest.mark.parametrize("section", ["data", "features", "foreground", "aggregation"])
def test_runs_differing_outside_the_method_are_rejected(section):
    base = run_set("global_nn", [0.5, 0.6])
    other = run_set("anchored_nn", [0.6, 0.7])
    altered = copy.deepcopy(other.results[0])
    altered["config"][section] = {"changed": True}
    other = RunSet(other.name, other.method, (altered, *other.results[1:]))
    with pytest.raises(ValueError, match=f"differs from exp_global_nn in '{section}'"):
        check_comparable([base, other])


def test_runs_with_different_seeds_or_queries_are_rejected():
    base = run_set("global_nn", [0.5, 0.6])
    with pytest.raises(ValueError, match="seeds"):
        check_comparable([base, run_set("anchored_nn", [0.5])])
    other = run_set("anchored_nn", [0.6, 0.7])
    altered = copy.deepcopy(other.results[1])
    altered["queries"]["03Mallard"]["total"] = 9
    with pytest.raises(ValueError, match="different queries"):
        check_comparable([base, RunSet(other.name, other.method, (other.results[0], altered))])


def test_render_markdown_contents():
    reference = load_reference_numbers(REPO / "configs" / "reference_numbers.yaml")
    markdown = render_markdown(
        [run_set("global_nn", [0.5, 0.6]), run_set("anchored_nn", [0.7, 0.8])], reference, "T"
    )
    assert "## T" in markdown
    for expected in ("55.0 ± 7.1", "75.0 ± 7.1", "PADFormer", "81.3", "UniVAD"):
        assert expected in markdown
    assert "not a like-for-like comparison" in markdown
    assert "Coverage of anchored_nn" in markdown
    assert "(uncommitted changes)" in markdown  # seed 1 of each run is marked dirty


def test_reference_numbers_file_matches_the_paper_table():
    reference = load_reference_numbers(REPO / "configs" / "reference_numbers.yaml")
    padformer = {m.name: m for m in reference.by_num_references[4]}["PADFormer"]
    assert (padformer.image_auroc, padformer.pixel_auroc, padformer.aupro) == (81.3, 92.9, 86.7)


def test_load_run_set_reads_seeds_in_order(tmp_path):
    for seed in (1, 0):
        path = tmp_path / f"seed_{seed}" / "results.json"
        path.parent.mkdir()
        path.write_text(json.dumps(fake_result("global_nn", seed, 0.5)))
    rs = load_run_set(tmp_path)
    assert rs.seeds == [0, 1]
    with pytest.raises(FileNotFoundError):
        load_run_set(tmp_path / "missing")
