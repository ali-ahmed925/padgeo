"""End to end on the fixture dataset: fake geometry backend, real DINOv2 ViT-S (CPU).

Covers data, features, geometry, both methods, all three metrics, result files, the table, the figure and
the profiling run, through the actual entry-point scripts. Also covers the invariances and leak guards the
project requires:

- the same references for both methods;
- degenerate samples counted, never dropped;
- the same numbers from a fresh cache;
- cached and fresh values give identical results, with no model loaded on a warm cache;
- batch size;
- reference order;
- scores blind to folder names and ground truth.
"""

from __future__ import annotations

import csv
import json
import math
import runpy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from padgeo.config import load_config
from padgeo.data.sampling import ReferenceSampler
from padgeo.features.dinov2 import DINOv2Extractor
from padgeo.geometry.fake import FakeGeometryBackend
from padgeo.pipeline import ExperimentRunner
from tests.helpers import (
    ANCHORED_METHOD,
    CLASSES,
    GLOBAL_METHOD,
    experiment_config,
    transforms_json_forbidden,
    write_config,
)

REPO = Path(__file__).resolve().parents[1]
SEEDS = (0, 1)


def run_script(name: str, *args: str) -> Any:
    """Call an entry-point script's `main` exactly as `python scripts/<name>.py <args>` would."""
    return runpy.run_path(str(REPO / "scripts" / f"{name}.py"))["main"](list(args))


def write_pair(root: Path, work: Path, *, batch_size: int = 4) -> tuple[Path, Path]:
    configs = work / "configs"
    return (
        write_config(
            configs / "global.yaml",
            experiment_config("fx_global", root, work, GLOBAL_METHOD, batch_size=batch_size),
        ),
        write_config(
            configs / "anchored.yaml",
            experiment_config("fx_anchored", root, work, ANCHORED_METHOD, batch_size=batch_size),
        ),
    )


def results(work: Path, name: str, seed: int) -> dict[str, Any]:
    return json.loads((work / "outputs" / name / f"seed_{seed}" / "results.json").read_text())


def per_query(work: Path, name: str, seed: int) -> list[dict[str, str]]:
    with (work / "outputs" / name / f"seed_{seed}" / "per_query.csv").open() as handle:
        return list(csv.DictReader(handle))


def scores(work: Path, name: str, seed: int, class_name: str) -> dict[str, np.ndarray]:
    with np.load(work / "outputs" / name / f"seed_{seed}" / "scores" / f"{class_name}.npz") as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="session")
def main_run(tmp_path_factory: pytest.TempPathFactory, dataset_root: Path) -> Path:
    """The full kill-test path on the fixture, with transforms.json reads forbidden throughout."""
    work = tmp_path_factory.mktemp("main")
    global_cfg, anchored_cfg = write_pair(dataset_root, work)
    profile_cfg = work / "configs" / "profile.yaml"
    profile_cfg.write_text(
        f"name: fx_profile\noutput_dir: {work / 'outputs'}\nexperiment_config: {anchored_cfg}\n"
        "num_references: [2, 4]\nprecisions: [fp32, bf16]\nqueries_per_setting: 2\n"
    )
    with transforms_json_forbidden():
        run_script("profile_geometry", "--config", str(profile_cfg))
        run_script("run", "--config", str(global_cfg))
        run_script("run", "--config", str(anchored_cfg))
        run_script(
            "make_table",
            "--configs",
            str(global_cfg),
            str(anchored_cfg),
            "--reference",
            str(REPO / "configs" / "reference_numbers.yaml"),
            "--out",
            str(work / "RESULTS.md"),
        )
        run_script(
            "visualise",
            "--configs",
            str(global_cfg),
            str(anchored_cfg),
            "--seed",
            "0",
            "--class-name",
            CLASSES[0],
            "--query",
            "test/Missing/0.png",
            "test/good/0.png",
            "--out",
            str(work / "figures"),
        )
    return work


def test_result_files_are_complete(main_run):
    for name, method in (("fx_global", "global_nn"), ("fx_anchored", "anchored_nn")):
        for seed in SEEDS:
            run_dir = main_run / "outputs" / name / f"seed_{seed}"
            for file in ("results.json", "per_class.csv", "per_query.csv", "config.yaml", "run.log"):
                assert (run_dir / file).is_file(), file
            summary = results(main_run, name, seed)
            assert summary["method"] == method
            assert summary["seed"] == seed
            assert set(summary["metrics"]["per_class"]) == set(CLASSES)
            for value in summary["metrics"]["mean"].values():
                assert 0.0 <= value <= 1.0
            assert summary["provenance"]["git"]["commit"]
            assert summary["provenance"]["packages"]["torch"]
            assert summary["config"]["method"]["name"] == method
            assert load_config(run_dir / "config.yaml").method.name == method  # resolved config reloads
            for class_name in CLASSES:
                assert summary["queries"][class_name]["total"] == 8
                assert summary["timing"]["per_class"][class_name]["seconds_per_query_end_to_end"] > 0
                arrays = scores(main_run, name, seed, class_name)
                assert arrays["patch_scores"].shape == (8, 37, 37)
                assert arrays["foreground"].dtype == bool
            rows = per_query(main_run, name, seed)
            assert len(rows) == 8 * len(CLASSES)
            assert {row["defect_type"] for row in rows} == {"good", "Missing", "Stains"}


def test_table_figure_and_profile(main_run):
    table = (main_run / "RESULTS.md").read_text()
    for expected in ("global_nn", "anchored_nn", "**Mean**", "PADFormer", "UniVAD"):  # N = 2 numbers exist
        assert expected in table
    assert "Coverage of anchored_nn" in table
    for query in ("test_Missing_0.png", "test_good_0.png"):
        assert (main_run / "figures" / f"{CLASSES[0]}_{query}_seed0.png").stat().st_size > 10_000
    report = json.loads((main_run / "outputs" / "fx_profile" / "report.json").read_text())
    assert [(row["num_references"], row["precision"]) for row in report["timing"]] == [
        (2, "fp32"),
        (4, "fp32"),
        (2, "bf16"),
        (4, "bf16"),
    ]
    assert len(report["precision_differences"]) == 2
    assert all(row["point_shift_median_diag"] > 0 for row in report["precision_differences"])


def test_both_methods_saw_identical_references(main_run):
    for seed in SEEDS:
        global_rows = {
            (r["class"], r["query"]): r["references"] for r in per_query(main_run, "fx_global", seed)
        }
        anchored_rows = {
            (r["class"], r["query"]): r["references"] for r in per_query(main_run, "fx_anchored", seed)
        }
        assert global_rows == anchored_rows
    # Different seeds draw different references.
    by_seed = [[r["references"] for r in per_query(main_run, "fx_global", seed)] for seed in SEEDS]
    assert by_seed[0] != by_seed[1]


def test_degenerate_query_is_counted_not_dropped(main_run):
    summary = results(main_run, "fx_anchored", 0)
    for class_name in CLASSES:
        assert summary["coverage"][class_name]["empty_foreground_queries"] == 1
    blank = [r for r in per_query(main_run, "fx_anchored", 0) if r["query"] == "test/good/3.png"]
    assert len(blank) == len(CLASSES)
    assert all(float(r["image_score"]) == 0.0 for r in blank)


def test_anchored_reports_coverage_and_global_has_none(main_run):
    anchored = results(main_run, "fx_anchored", 0)["coverage"]
    global_ = results(main_run, "fx_global", 0)["coverage"]
    for class_name in CLASSES:
        assert global_[class_name]["unseen_fraction"] == 0.0
        assert 0.0 <= anchored[class_name]["unseen_fraction"] <= 1.0
        assert anchored[class_name]["foreground_patches"] == global_[class_name]["foreground_patches"]


def assert_same_results(work_a: Path, work_b: Path, names: tuple[str, ...]) -> None:
    for name in names:
        for seed in SEEDS:
            a, b = results(work_a, name, seed), results(work_b, name, seed)
            assert a["metrics"] == b["metrics"]
            assert a["coverage"] == b["coverage"]
            for class_name in CLASSES:
                arrays_a, arrays_b = (
                    scores(work_a, name, seed, class_name),
                    scores(work_b, name, seed, class_name),
                )
                for key in arrays_a:
                    assert np.array_equal(arrays_a[key], arrays_b[key]), (name, seed, class_name, key)


def test_rerun_from_a_fresh_cache_reproduces_every_number(main_run, dataset_root, tmp_path):
    global_cfg, anchored_cfg = write_pair(dataset_root, tmp_path)
    run_script("run", "--config", str(global_cfg))
    run_script("run", "--config", str(anchored_cfg))
    assert_same_results(main_run, tmp_path, ("fx_global", "fx_anchored"))


def test_cached_run_is_identical_and_loads_no_model(main_run, dataset_root, tmp_path, monkeypatch):
    def no_model(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a model was used although every value is cached")

    monkeypatch.setattr(DINOv2Extractor, "_load_model", no_model)
    monkeypatch.setattr(FakeGeometryBackend, "predict", no_model)
    for name, method in (("fx_global", GLOBAL_METHOD), ("fx_anchored", ANCHORED_METHOD)):
        config = experiment_config(name, dataset_root, main_run, method)
        config["experiment"]["output_dir"] = str(tmp_path / "outputs")  # warm cache of main_run, new outputs
        run_script("run", "--config", str(write_config(tmp_path / f"{name}.yaml", config)))
    assert_same_results(main_run, tmp_path, ("fx_global", "fx_anchored"))
    assert results(tmp_path, "fx_anchored", 0)["cache"]["geometry_misses"] == 0


def test_batch_size_does_not_change_results(main_run, dataset_root, tmp_path):
    global_cfg, _ = write_pair(dataset_root, tmp_path, batch_size=1)
    run_script("run", "--config", str(global_cfg))
    for seed in SEEDS:
        expected = results(main_run, "fx_global", seed)["metrics"]["per_class"]
        got = results(tmp_path, "fx_global", seed)["metrics"]["per_class"]
        for class_name in CLASSES:
            for metric, value in expected[class_name].items():
                assert math.isclose(got[class_name][metric], value, abs_tol=1e-6), (seed, class_name, metric)


def test_reference_order_does_not_change_results(main_run, dataset_root, tmp_path, monkeypatch):
    original = ReferenceSampler.sample
    monkeypatch.setattr(
        ReferenceSampler, "sample", lambda self, query: tuple(reversed(original(self, query)))
    )
    _, anchored_cfg = write_pair(dataset_root, tmp_path)
    run_script("run", "--config", str(anchored_cfg))
    assert_same_results(main_run, tmp_path, ("fx_anchored",))


def test_scores_are_blind_to_folder_names_and_ground_truth(main_run, dataset_root, tmp_path):
    """Scoring a copy with renamed defect folders and no ground truth gives bit-identical scores."""
    copy = tmp_path / "renamed"
    shutil.copytree(dataset_root, copy)
    for class_name in CLASSES:
        shutil.rmtree(copy / class_name / "ground_truth")
        (copy / class_name / "test" / "Missing").rename(copy / class_name / "test" / "zz_one")
        (copy / class_name / "test" / "Stains").rename(copy / class_name / "test" / "aa_two")

    def score_all(root: Path) -> dict[str, tuple[float, np.ndarray]]:
        config = experiment_config("blind", root, main_run, ANCHORED_METHOD)  # warm cache: same image hashes
        runner = ExperimentRunner(load_config(write_config(tmp_path / f"{root.name}.yaml", config)))
        return {
            record.query.content_hash: (record.output.image_score, record.output.patch_scores.numpy())
            for class_name in CLASSES
            for record in runner.score_class(class_name, seed=0).records
        }

    expected, renamed = score_all(dataset_root), score_all(copy)
    assert renamed.keys() == expected.keys()
    for key, (image_score, patch_scores) in renamed.items():
        assert image_score == expected[key][0]
        assert np.array_equal(patch_scores, expected[key][1])


def test_per_object_sampling_reuses_one_reference_set_per_class(main_run, dataset_root, tmp_path):
    config = experiment_config("fx_per_object", dataset_root, main_run, GLOBAL_METHOD, seeds=(0,))
    config["data"]["sampling"] = "per_object"
    config["experiment"]["output_dir"] = str(tmp_path / "outputs")
    run_script("run", "--config", str(write_config(tmp_path / "per_object.yaml", config)))
    rows = per_query(tmp_path, "fx_per_object", 0)
    for class_name in CLASSES:
        assert len({r["references"] for r in rows if r["class"] == class_name}) == 1


def test_max_queries_per_class_evaluates_an_evenly_spaced_subset(main_run, dataset_root, tmp_path):
    config = experiment_config("fx_subset", dataset_root, main_run, GLOBAL_METHOD, seeds=(0,))
    config["data"]["max_queries_per_class"] = 3
    config["experiment"]["output_dir"] = str(tmp_path / "outputs")
    run_script("run", "--config", str(write_config(tmp_path / "subset.yaml", config)))
    rows = per_query(tmp_path, "fx_subset", 0)
    for class_name in CLASSES:
        names = [r["query"] for r in rows if r["class"] == class_name]
        # 8 queries sorted by name (upper case first): indices 0, 4 (3.5 rounded half to even) and 7.
        assert names == ["test/Missing/0.png", "test/good/0.png", "test/good/3.png"]
