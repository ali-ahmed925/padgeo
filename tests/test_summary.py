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
    anchored = method == "anchored_nn"
    return {
        "experiment": f"exp_{method}",
        "method": method,
        "seed": seed,
        "classes": ["03Mallard"],
        "metrics": {"per_class": {"03Mallard": dict(metrics)}, "mean": dict(metrics)},
        "image_auroc_by_type": {"03Mallard": {"Burrs": value, "Missing": value / 2}},
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
                "radius_in_query_patches_median": 2.5 + seed if anchored else None,
                "eligible_reference_fraction_median": 0.9 if anchored else None,
            }
        },
        "config": {
            "data": {"dataset": "mad_sim", "num_references": 4, "sampling": "per_query", "root": "r"},
            "features": {"backbone": "dinov2_vitl14"},
            "foreground": {"pixel_threshold": 0.98},
            "aggregation": {"pixel_sigma": 4.0},
            "method": {"name": method, "device": "cuda"},
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
    for expected in (
        "55.0 ± 7.1",  # global_nn: 50 and 60
        "75.0 ± 7.1",  # anchored_nn: 70 and 80
        "| exp_anchored_nn | **Mean** | +20.0 ± 0.0 | +20.0 ± 0.0 | +20.0 ± 0.0 |",  # paired per seed
        "Image AUROC · exp_global_nn | Image AUROC · exp_anchored_nn",  # columns labelled by run
        '- `exp_anchored_nn`: anchored_nn, {"device": "cuda"}',  # method settings listed
        "| exp_anchored_nn | 03Mallard | 10.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 | 3.0 ± 0.7 | 90.0 ± 0.0 | 0 |",
        "| 03Mallard | Missing | 27.5 ± 3.5",
        "PADFormer",
        "81.3",
        "UniVAD",
    ):
        assert expected in markdown, expected
    assert "not a like-for-like comparison" in markdown
    assert "Coverage of anchored_nn" in markdown
    assert "(uncommitted changes)" in markdown  # seed 1 of each run is marked dirty


def test_reference_numbers_file_matches_the_paper_table():
    reference = load_reference_numbers(REPO / "configs" / "reference_numbers.yaml")
    padformer = {m.name: m for m in reference.by_num_references[4]}["PADFormer"]
    assert (padformer.image_auroc, padformer.pixel_auroc, padformer.aupro) == (81.3, 92.9, 86.7)


def test_load_run_set_reads_exactly_the_configured_seeds(tmp_path):
    from padgeo.config import Config

    config = Config.model_validate(
        {
            "experiment": {"name": "exp", "output_dir": str(tmp_path), "seeds": [1, 0]},
            "data": {"root": "r", "classes": ["03Mallard"]},
            "method": {"name": "global_nn"},
        }
    )
    resolved = json.loads(config.model_dump_json())
    for seed in (0, 1, 7):  # seed 7 is a leftover of an older run with other seeds: ignored
        path = tmp_path / "exp" / f"seed_{seed}" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({**fake_result("global_nn", seed, 0.5), "config": resolved}))
    rs = load_run_set(config)
    assert rs.seeds == [1, 0]  # the configured seeds, in config order
    stale = tmp_path / "exp" / "seed_0" / "results.json"
    stale.write_text(json.dumps({**fake_result("global_nn", 0, 0.5), "config": {**resolved, "cache": {}}}))
    with pytest.raises(ValueError, match="different config"):
        load_run_set(config)
    (tmp_path / "exp" / "seed_1" / "results.json").unlink()
    with pytest.raises(FileNotFoundError, match="seed 1"):
        load_run_set(config)
