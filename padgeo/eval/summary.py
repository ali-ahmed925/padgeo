"""Cross-seed summary tables (step 12), generated only from result files.

`scripts/make_table.py` calls these functions; `docs/RESULTS.md` is its output and is never edited by hand.
Before tabulating, `check_comparable` verifies that the runs differ only in their method section (plus
the experiment name), so a table can never silently compare runs with different data, references,
features, masks or aggregation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from padgeo.config import Config, load_yaml
from padgeo.eval.results import METRICS

# Config sections that must be identical across compared runs. `method` may differ (it is the treatment);
# `experiment.name` differs by construction; `cache.dir` and `experiment.output_dir` do not affect results.
_SHARED_SECTIONS = ("data", "features", "foreground", "aggregation")
_METRIC_LABELS = {"image_auroc": "Image AUROC", "pixel_auroc": "Pixel AUROC", "aupro": "AUPRO"}


class ReferenceMethod(BaseModel):
    """Published numbers of one method (percent)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    training: str
    image_auroc: float
    pixel_auroc: float
    aupro: float


class ReferenceNumbers(BaseModel):
    """Published reference numbers for one dataset, keyed by the number of references N."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str
    note: str
    by_num_references: dict[int, list[ReferenceMethod]] = Field(min_length=1)


def load_reference_numbers(path: Path) -> ReferenceNumbers:
    """Load and validate the reference-numbers YAML file."""
    return ReferenceNumbers.model_validate(load_yaml(path))


@dataclass(frozen=True)
class RunSet:
    """All seeds of one experiment.

    Attributes:
        name: Experiment name.
        method: Method name.
        results: Parsed `results.json` of each seed, sorted by seed.
    """

    name: str
    method: str
    results: tuple[dict[str, Any], ...]

    @property
    def seeds(self) -> list[int]:
        """Seeds present on disk."""
        return [int(result["seed"]) for result in self.results]

    @property
    def classes(self) -> list[str]:
        """Classes of the experiment, in run order."""
        return list(self.results[0]["classes"])


def load_run_set(config: Config) -> RunSet:
    """Read the results of every seed listed in an experiment config.

    Only `seed_<s>` directories for the config's seeds are read, and each must have been produced by exactly
    this config (the resolved config stored in its `results.json`). Leftovers of an earlier run with
    different settings or seeds therefore cannot enter a table.

    Raises:
        FileNotFoundError: If a configured seed has no results.
        ValueError: If a seed's results were produced with a different config.
    """
    experiment_dir = config.experiment.output_dir / config.experiment.name
    expected = json.loads(config.model_dump_json())
    results = []
    for seed in config.experiment.seeds:
        path = experiment_dir / f"seed_{seed}" / "results.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing results for seed {seed}: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["config"] != expected:
            raise ValueError(f"{path} was produced with a different config; rerun the experiment")
        results.append(result)
    return RunSet(name=config.experiment.name, method=config.method.name, results=tuple(results))


def check_comparable(run_sets: Sequence[RunSet]) -> None:
    """Raise if the runs differ in anything but the method (data, features, masks, aggregation, seeds).

    Raises:
        ValueError: Listing every difference found.
    """
    problems = []
    base = run_sets[0]
    base_config = base.results[0]["config"]
    for other in run_sets[1:]:
        other_config = other.results[0]["config"]
        for section in _SHARED_SECTIONS:
            if base_config[section] != other_config[section]:
                problems.append(f"{other.name} differs from {base.name} in '{section}'")
        if other.seeds != base.seeds:
            problems.append(f"{other.name} seeds {other.seeds} != {base.name} seeds {base.seeds}")
        for base_result, other_result in zip(base.results, other.results, strict=False):
            if base_result["queries"] != other_result["queries"]:
                problems.append(f"{other.name} seed {other_result['seed']} evaluated different queries")
    if problems:
        raise ValueError("runs are not comparable:\n" + "\n".join(problems))


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1; NaN for a single value)."""
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if array.size > 1 else float("nan")
    return float(array.mean()), std


def _format(values: Sequence[float], *, percent: bool = True, signed: bool = False) -> str:
    """Mean ± sample std of per-seed values ("n/a" if any value is missing)."""
    if any(value is None for value in values):
        return "n/a"
    scale = 100.0 if percent else 1.0
    mean, std = mean_std([scale * value for value in values])
    text = f"{mean:+.1f}" if signed else f"{mean:.1f}"
    return text if np.isnan(std) else f"{text} ± {std:.1f}"


def metric_values(run_set: RunSet, metric: str, class_name: str | None) -> list[float]:
    """One value per seed: a class's metric, or (class_name None) the mean over classes."""
    if class_name is None:
        return [float(result["metrics"]["mean"][metric]) for result in run_set.results]
    return [float(result["metrics"]["per_class"][class_name][metric]) for result in run_set.results]


def _runs_legend(run_sets: Sequence[RunSet]) -> list[str]:
    lines = ["Runs (columns are labelled by run name):", ""]
    for run_set in run_sets:
        method = {k: v for k, v in run_set.results[0]["config"]["method"].items() if k != "name"}
        lines.append(f"- `{run_set.name}`: {run_set.method}, {json.dumps(method, separators=(', ', ': '))}")
    return [*lines, ""]


def _metrics_table(run_sets: Sequence[RunSet]) -> list[str]:
    header = ["Class"] + [f"{_METRIC_LABELS[m]} · {rs.name}" for m in METRICS for rs in run_sets]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for class_name in [*run_sets[0].classes, None]:
        cells = [class_name or "**Mean**"]
        cells += [_format(metric_values(rs, m, class_name)) for m in METRICS for rs in run_sets]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _paired_difference_table(run_sets: Sequence[RunSet]) -> list[str]:
    """Each run minus the first, seed by seed.

    Both runs used the same references for every query of a seed (checked by `check_comparable`), so the
    per-seed difference is a paired comparison.
    """
    if len(run_sets) < 2:
        return []
    base = run_sets[0]
    header = ["Run", "Class"] + [f"Δ {_METRIC_LABELS[m]}" for m in METRICS]
    lines = [
        "",
        f"Paired difference to `{base.name}` (per seed, same references; mean ± std over seeds,"
        " percentage points):",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for run_set in run_sets[1:]:
        for class_name in [*base.classes, None]:
            cells = [run_set.name, class_name or "**Mean**"]
            for metric in METRICS:
                ours, theirs = (
                    metric_values(run_set, metric, class_name),
                    metric_values(base, metric, class_name),
                )
                cells.append(_format([a - b for a, b in zip(ours, theirs, strict=True)], signed=True))
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def _by_type_table(run_sets: Sequence[RunSet]) -> list[str]:
    header = ["Class", "Defect type"] + [f"Image AUROC · {rs.name}" for rs in run_sets]
    lines = [
        "",
        "Supplementary: image AUROC of the defect-free queries against each defect type alone (mean ± std"
        " over seeds, percent).",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for class_name in run_sets[0].classes:
        for defect_type in run_sets[0].results[0]["image_auroc_by_type"][class_name]:
            cells = [class_name, defect_type]
            cells += [
                _format([r["image_auroc_by_type"][class_name][defect_type] for r in rs.results])
                for rs in run_sets
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def _coverage_table(run_sets: Sequence[RunSet]) -> list[str]:
    anchored = [rs for rs in run_sets if rs.method == "anchored_nn"]
    if not anchored:
        return []
    fractions = ("unseen_fraction", "unseen_no_neighbour_fraction", "unseen_low_confidence_fraction")
    lines = [
        "",
        "Coverage of anchored_nn (mean ± std over seeds). Unseen: percent of foreground query patches, which"
        " use the configured unseen mode. Radius: the effective radius in query patches (median over"
        " queries). Eligible: percent of reference patches passing the confidence filter (median).",
        "",
        "| Run | Class | Unseen | No neighbour | Low confidence | Radius (patches) | Eligible |"
        " Empty-foreground queries |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run_set in anchored:
        for class_name in run_set.classes:
            coverage = [r["coverage"][class_name] for r in run_set.results]
            cells = [run_set.name, class_name]
            cells += [_format([c[key] for c in coverage]) for key in fractions]
            cells.append(_format([c["radius_in_query_patches_median"] for c in coverage], percent=False))
            cells.append(_format([c["eligible_reference_fraction_median"] for c in coverage]))
            cells.append(str(max(c["empty_foreground_queries"] for c in coverage)))
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def _reference_table(reference: ReferenceNumbers, num_references: int, num_classes: int) -> list[str]:
    methods = reference.by_num_references.get(num_references)
    if not methods:
        return ["", f"No published reference numbers for N = {num_references}."]
    lines = [
        "",
        f"Published numbers, N = {num_references} (single runs, percent). Source: {reference.source}",
        f"Note: {reference.note} The runs above cover {num_classes} classes, so these means are context,"
        " not a like-for-like comparison.",
        "",
        "| Method | Training | Image AUROC | Pixel AUROC | AUPRO |",
        "|---|---|---|---|---|",
    ]
    for method in methods:
        lines.append(
            f"| {method.name} | {method.training} | {method.image_auroc:.1f} | {method.pixel_auroc:.1f}"
            f" | {method.aupro:.1f} |"
        )
    return lines


def _provenance_lines(run_sets: Sequence[RunSet]) -> list[str]:
    lines = ["", "Provenance:", ""]
    for run_set in run_sets:
        for result in run_set.results:
            git = result["provenance"]["git"]
            gpus = ", ".join(gpu["name"] for gpu in result["provenance"]["hardware"]["gpus"]) or "CPU only"
            dirty = " **(uncommitted changes)**" if git["dirty"] else ""
            lines.append(
                f"- {run_set.name} seed {result['seed']}: commit `{git['commit'][:12]}`{dirty}, {gpus}"
            )
    return lines


def render_markdown(run_sets: Sequence[RunSet], reference: ReferenceNumbers, title: str) -> str:
    """The full results section: metrics table, coverage, reference numbers and provenance.

    Raises:
        ValueError: If the runs are not comparable (see `check_comparable`).
    """
    check_comparable(run_sets)
    config = run_sets[0].results[0]["config"]
    num_references = int(config["data"]["num_references"])
    seeds = run_sets[0].seeds
    lines = [
        f"## {title}",
        "",
        "Generated by `scripts/make_table.py` from result files; do not edit by hand.",
        "",
        f"Dataset `{config['data']['dataset']}`, N = {num_references} references"
        f" ({config['data']['sampling']} sampling), seeds {seeds}, backbone {config['features']['backbone']}."
        " Values are percent, mean ± sample std over seeds; the mean row averages classes within each seed"
        " first.",
        "",
        *_runs_legend(run_sets),
        *_metrics_table(run_sets),
        *_paired_difference_table(run_sets),
        *_by_type_table(run_sets),
        *_coverage_table(run_sets),
        *_reference_table(reference, num_references, len(run_sets[0].classes)),
        *_provenance_lines(run_sets),
    ]
    return "\n".join(lines) + "\n"
