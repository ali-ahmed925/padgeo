"""Step 11: per-seed result files.

Each seed of an experiment writes to `<output_dir>/<name>/seed_<seed>/` (class scores are written as soon as a
class is evaluated, so its pixel maps can be released; the seed-level files are written at the end):

- `results.json`: per-class and mean metrics, coverage (unseen fractions), degenerate-sample counts,
  timing, peak VRAM, cache statistics, the fully resolved config and provenance (git commit, library
  versions, hardware).
- `per_class.csv`: one row per class plus a "mean" row.
- `per_query.csv`: one row per query: label, defect type, image score, references, coverage and timing.
- `scores/<class>.npz`: patch scores, foreground and unseen masks per query (for visualisation and
  post-hoc analysis; pixel maps are recomputed from patch scores with the shared aggregation code).
- `config.yaml`: the resolved config, loadable with `padgeo.config.load_config`.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from padgeo.config import Config, config_to_yaml
from padgeo.data.base import ImageRef
from padgeo.eval.evaluate import ClassMetrics
from padgeo.methods.base import MethodOutput

METRICS = ("image_auroc", "pixel_auroc", "aupro")


@dataclass(frozen=True)
class QueryRecord:
    """Everything produced for one query (step 9).

    Attributes:
        query: The query image.
        references: Its references, in canonical (content-hash) order; the geometry pass sees
            [query, *references].
        foreground: [h, w] bool query foreground mask.
        output: The method's scores.
        geometry_seconds: Time to obtain the patch geometry (cache read or backend pass); 0 for global_nn.
        scoring_seconds: Time of `method.score`.
        peak_vram_mb: Peak CUDA memory during this query, or None without CUDA.
        geometry_cache_hit: Whether the geometry came from cache; None for global_nn.
    """

    query: ImageRef
    references: tuple[ImageRef, ...]
    foreground: torch.Tensor
    output: MethodOutput
    geometry_seconds: float
    scoring_seconds: float
    peak_vram_mb: float | None
    geometry_cache_hit: bool | None

    @property
    def num_foreground(self) -> int:
        """Number of foreground patches of the query."""
        return int(self.foreground.sum())


@dataclass(frozen=True)
class ClassResult:
    """All outputs of one class and seed.

    Attributes:
        class_name: Class folder name.
        metrics: Metrics and the labels they were computed from.
        records: One record per query, in query order.
        num_images_prepared: Distinct images whose features and masks were prepared.
        feature_seconds: Time to prepare them (cache reads plus feature extraction).
        feature_peak_vram_mb: Peak CUDA memory during feature preparation, or None without CUDA.
    """

    class_name: str
    metrics: ClassMetrics
    records: tuple[QueryRecord, ...]
    num_images_prepared: int
    feature_seconds: float
    feature_peak_vram_mb: float | None


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def coverage_summary(records: Sequence[QueryRecord]) -> dict[str, Any]:
    """Unseen fractions over all foreground query patches of a class, and degenerate-query counts.

    Fractions pool patches over queries (total unseen / total foreground). Queries without foreground
    patches are scored 0 everywhere; they are counted, never dropped.
    """
    foreground = sum(record.num_foreground for record in records)
    no_neighbour = sum(int(record.output.unseen_no_neighbour.sum()) for record in records)
    low_confidence = sum(int(record.output.unseen_low_confidence.sum()) for record in records)
    return {
        "foreground_patches": foreground,
        "unseen_fraction": _fraction(no_neighbour + low_confidence, foreground),
        "unseen_no_neighbour_fraction": _fraction(no_neighbour, foreground),
        "unseen_low_confidence_fraction": _fraction(low_confidence, foreground),
        "empty_foreground_queries": sum(record.num_foreground == 0 for record in records),
    }


def timing_summary(result: ClassResult) -> dict[str, Any]:
    """Seconds and peak VRAM of one class (features, per-query geometry and scoring)."""
    query_seconds = [record.geometry_seconds + record.scoring_seconds for record in result.records]
    vram = [record.peak_vram_mb for record in result.records if record.peak_vram_mb is not None]
    if result.feature_peak_vram_mb is not None:
        vram.append(result.feature_peak_vram_mb)
    num = len(result.records)
    return {
        "feature_seconds": result.feature_seconds,
        "images_prepared": result.num_images_prepared,
        "query_seconds_mean": float(np.mean(query_seconds)),
        "query_seconds_max": float(np.max(query_seconds)),
        "geometry_seconds_mean": float(np.mean([record.geometry_seconds for record in result.records])),
        "scoring_seconds_mean": float(np.mean([record.scoring_seconds for record in result.records])),
        "seconds_per_query_end_to_end": (result.feature_seconds + sum(query_seconds)) / num,
        "geometry_cache_hits": sum(record.geometry_cache_hit is True for record in result.records),
        "peak_vram_mb": max(vram) if vram else None,
    }


@dataclass(frozen=True)
class ClassSummary:
    """Everything the seed-level result files need from one class.

    It holds no pixel maps, so a class can be released from memory as soon as it is evaluated and its
    scores are written.

    Attributes:
        class_name: Class folder name.
        metrics: Image AUROC, pixel AUROC and AUPRO.
        queries: Query counts (total, defective, empty defect masks, per type).
        coverage: Unseen fractions and degenerate-query counts (`coverage_summary`).
        timing: Seconds and peak VRAM (`timing_summary`).
        query_rows: One `per_query.csv` row per query.
    """

    class_name: str
    metrics: dict[str, float]
    queries: dict[str, Any]
    coverage: dict[str, Any]
    timing: dict[str, Any]
    query_rows: list[dict[str, Any]]


def _query_rows(result: ClassResult) -> list[dict[str, Any]]:
    rows = []
    for record, label, defect_type in zip(
        result.records, result.metrics.labels, result.metrics.defect_types, strict=True
    ):
        rows.append(
            {
                "class": result.class_name,
                "query": record.query.name,
                "label": label,
                "defect_type": defect_type,
                "image_score": record.output.image_score,
                "foreground_patches": record.num_foreground,
                "unseen_no_neighbour": int(record.output.unseen_no_neighbour.sum()),
                "unseen_low_confidence": int(record.output.unseen_low_confidence.sum()),
                "references": " ".join(ref.name for ref in record.references),
                "geometry_seconds": record.geometry_seconds,
                "scoring_seconds": record.scoring_seconds,
                "peak_vram_mb": record.peak_vram_mb,
                "geometry_cache_hit": record.geometry_cache_hit,
            }
        )
    return rows


def summarise_class(result: ClassResult) -> ClassSummary:
    """Reduce a class result to the summary the seed-level files are built from."""
    return ClassSummary(
        class_name=result.class_name,
        metrics={name: float(getattr(result.metrics, name)) for name in METRICS},
        queries={
            "total": len(result.records),
            "defective": int(sum(result.metrics.labels)),
            "defective_with_empty_mask": result.metrics.empty_defect_masks,
            "per_type": result.metrics.queries_per_type,
        },
        coverage=coverage_summary(result.records),
        timing=timing_summary(result),
        query_rows=_query_rows(result),
    )


def write_class_scores(run_dir: Path, result: ClassResult) -> None:
    """Write `scores/<class>.npz`: query names, image scores, patch scores, foreground and unseen masks."""
    records = result.records
    (run_dir / "scores").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "scores" / f"{result.class_name}.npz",
        query=np.array([record.query.name for record in records]),
        image_score=np.array([record.output.image_score for record in records], dtype=np.float64),
        patch_scores=torch.stack([record.output.patch_scores for record in records]).numpy(),
        foreground=torch.stack([record.foreground for record in records]).numpy(),
        unseen_no_neighbour=torch.stack([record.output.unseen_no_neighbour for record in records]).numpy(),
        unseen_low_confidence=torch.stack(
            [record.output.unseen_low_confidence for record in records]
        ).numpy(),
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_seed_results(
    run_dir: Path,
    *,
    config: Config,
    seed: int,
    summaries: Sequence[ClassSummary],
    provenance: dict[str, Any],
    total_seconds: float,
    cache_stats: dict[str, int],
) -> None:
    """Write the seed-level result files (see the module docstring); class scores are written per class."""
    per_class = {summary.class_name: summary.metrics for summary in summaries}
    mean = {name: float(np.mean([row[name] for row in per_class.values()])) for name in METRICS}
    if any(not math.isfinite(value) for value in mean.values()):
        raise ValueError(f"non-finite mean metrics: {mean}")
    document = {
        "experiment": config.experiment.name,
        "method": config.method.name,
        "seed": seed,
        "classes": [summary.class_name for summary in summaries],
        "metrics": {"per_class": per_class, "mean": mean},
        "queries": {summary.class_name: summary.queries for summary in summaries},
        "coverage": {summary.class_name: summary.coverage for summary in summaries},
        "timing": {
            "total_seconds": total_seconds,
            "per_class": {summary.class_name: summary.timing for summary in summaries},
        },
        "cache": cache_stats,
        "config": json.loads(config.model_dump_json()),
        "provenance": provenance,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "config.yaml").write_text(config_to_yaml(config), encoding="utf-8")
    _write_csv(
        run_dir / "per_class.csv",
        [{"class": name, **row} for name, row in per_class.items()] + [{"class": "mean", **mean}],
    )
    _write_csv(run_dir / "per_query.csv", [row for summary in summaries for row in summary.query_rows])
