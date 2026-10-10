"""Geometry profiling: peak VRAM, seconds per query, and the effect of precision on the 3D points.

This is the first step of the kill test on the experiment machine. For each precision a fresh backend is
built, warmed up once (untimed), then timed on `queries_per_setting` queries per N. Point differences are
measured on foreground patches of all S images. They are reported relative to the reference cloud's
bounding-box diagonal and to the anchored radius, together with the fraction of radius-neighbour
relations that flip, which is what precision could change in anchored_nn.
"""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from padgeo.config import AnchoredNNConfig, Config, ProfileConfig, RadiusNeighbourhood
from padgeo.data.base import ImageRef
from padgeo.data.images import PATCH_SIZE, load_model_input
from padgeo.data.registry import build_dataset
from padgeo.data.sampling import ReferenceSampler, select_queries
from padgeo.features.foreground import patch_foreground, pixel_foreground
from padgeo.geometry import build_geometry_backend
from padgeo.geometry.patches import PatchGeometry, pool_to_patch_grid
from padgeo.methods.anchored_nn import bounding_box_diagonal, neighbour_mask
from padgeo.utils.resources import peak_vram_mb, reset_peak_vram, stopwatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageSet:
    """One query's geometry input: pixels, masks and the patch foreground used to compare points."""

    pixels: torch.Tensor  # [S, 3, H, W]
    pixel_foreground: torch.Tensor  # [S, H, W] bool
    patch_foreground: torch.Tensor  # [S, h, w] bool


def _image_set(images: list[ImageRef], experiment: Config) -> ImageSet:
    inputs = [load_model_input(ref.path).pixels for ref in images]
    pixel_masks = [pixel_foreground(pixels, experiment.foreground) for pixels in inputs]
    return ImageSet(
        pixels=torch.stack(inputs),
        pixel_foreground=torch.stack(pixel_masks),
        patch_foreground=torch.stack(
            [patch_foreground(mask, PATCH_SIZE, experiment.foreground) for mask in pixel_masks]
        ),
    )


def _point_differences(
    reference: PatchGeometry, other: PatchGeometry, image_set: ImageSet, method: AnchoredNNConfig
) -> dict[str, float]:
    """Differences of `other` relative to `reference` on foreground patches (image 0 is the query)."""
    mask = image_set.patch_foreground
    diagonal = bounding_box_diagonal(reference.xyz[1:][mask[1:]])
    distance = (other.xyz[mask] - reference.xyz[mask]).norm(dim=-1) / diagonal
    report = {
        "point_shift_median_diag": float(distance.median()),
        "point_shift_p95_diag": float(torch.quantile(distance, 0.95)),
        "point_shift_max_diag": float(distance.max()),
        "confidence_rel_change_median": float(
            (
                (other.confidence[mask] - reference.confidence[mask]).abs() / reference.confidence[mask]
            ).median()
        ),
    }
    if isinstance(method.neighbourhood, RadiusNeighbourhood) and math.isfinite(method.neighbourhood.radius):
        report["point_shift_p95_radius"] = report["point_shift_p95_diag"] / method.neighbourhood.radius
    eligible_ref = reference.confidence[1:][mask[1:]] >= method.min_confidence
    eligible_other = other.confidence[1:][mask[1:]] >= method.min_confidence
    neighbours_ref = neighbour_mask(
        reference.xyz[0][mask[0]], reference.xyz[1:][mask[1:]], eligible_ref, method.neighbourhood
    )
    neighbours_other = neighbour_mask(
        other.xyz[0][mask[0]], other.xyz[1:][mask[1:]], eligible_other, method.neighbourhood
    )
    union = int((neighbours_ref | neighbours_other).sum())
    report["neighbour_flip_rate"] = int((neighbours_ref ^ neighbours_other).sum()) / union if union else 0.0
    return report


def run_profile(profile: ProfileConfig, experiment: Config) -> dict[str, Any]:
    """Profile the geometry backend of an anchored_nn experiment.

    Returns:
        {"timing": [per (N, precision): seconds/query, peak VRAM], "precision_differences": [per N and
        non-reference precision: point shift and neighbour flip statistics, averaged over queries]}.
    """
    method = experiment.method
    if not isinstance(method, AnchoredNNConfig):
        raise ValueError("profiling needs an anchored_nn experiment config")
    class_name = experiment.data.classes[0]
    split = build_dataset(experiment.data).load_split(class_name)
    queries = select_queries(split.queries, profile.queries_per_setting)
    seed = experiment.experiment.seeds[0]
    image_sets = {}
    for num in profile.num_references:
        sampler = ReferenceSampler(split, num, experiment.data.sampling, seed)
        image_sets[num] = [_image_set([query, *sampler.sample(query)], experiment) for query in queries]

    timing = []
    geometry: dict[tuple[str, int], list[PatchGeometry]] = {}
    for precision in profile.precisions:
        backend = build_geometry_backend(method.geometry.model_copy(update={"precision": precision}))
        backend.predict(image_sets[profile.num_references[0]][0].pixels)  # warm-up: loading, kernel selection
        for num in profile.num_references:
            seconds, vram, outputs = [], [], []
            for image_set in image_sets[num]:
                reset_peak_vram()
                with stopwatch() as elapsed:
                    prediction = backend.predict(image_set.pixels)
                seconds.append(elapsed[0])
                vram.append(peak_vram_mb())
                outputs.append(pool_to_patch_grid(prediction, image_set.pixel_foreground, PATCH_SIZE))
            geometry[(precision, num)] = outputs
            measured = [v for v in vram if v is not None]
            timing.append(
                {
                    "num_references": num,
                    "num_images": num + 1,
                    "precision": precision,
                    "seconds_per_query_mean": float(np.mean(seconds)),
                    "seconds_per_query_max": float(np.max(seconds)),
                    "peak_vram_mb": max(measured) if measured else None,
                }
            )
            logger.info(
                "N=%d %s: %.3fs/query, peak VRAM %s MB",
                num,
                precision,
                np.mean(seconds),
                timing[-1]["peak_vram_mb"],
            )
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    differences = []
    baseline = profile.precisions[0]
    for precision in profile.precisions[1:]:
        for num in profile.num_references:
            per_query = [
                _point_differences(ref, other, image_set, method)
                for ref, other, image_set in zip(
                    geometry[(baseline, num)], geometry[(precision, num)], image_sets[num], strict=True
                )
            ]
            averaged = {key: float(np.mean([item[key] for item in per_query])) for key in per_query[0]}
            differences.append(
                {"num_references": num, "precision": precision, "baseline": baseline, **averaged}
            )
    return {
        "class": class_name,
        "queries": [query.name for query in queries],
        "geometry": method.geometry.model_dump(mode="json", exclude={"precision"}),
        "timing": timing,
        "precision_differences": differences,
    }


def render_profile(report: dict[str, Any]) -> str:
    """Markdown summary of a profile report."""
    lines = [
        f"Geometry profile on {report['class']} ({len(report['queries'])} queries per setting)",
        "",
        "| N | Images | Precision | s/query (mean) | s/query (max) | Peak VRAM (MB) |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["timing"]:
        vram = "n/a (CPU)" if row["peak_vram_mb"] is None else f"{row['peak_vram_mb']:.0f}"
        lines.append(
            f"| {row['num_references']} | {row['num_images']} | {row['precision']} |"
            f" {row['seconds_per_query_mean']:.3f} | {row['seconds_per_query_max']:.3f} | {vram} |"
        )
    if report["precision_differences"]:
        lines += [
            "",
            "Point differences vs the baseline precision (foreground patches, mean over queries; shifts are"
            " fractions of the reference bounding-box diagonal):",
            "",
            "| N | Precision vs baseline | Median shift | p95 shift | Max shift | p95 shift / r |"
            " Neighbour flips | Median conf. change |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in report["precision_differences"]:
            ratio = row.get("point_shift_p95_radius")
            lines.append(
                f"| {row['num_references']} | {row['precision']} vs {row['baseline']} |"
                f" {row['point_shift_median_diag']:.2e} | {row['point_shift_p95_diag']:.2e} |"
                f" {row['point_shift_max_diag']:.2e} | {'n/a' if ratio is None else f'{ratio:.3f}'} |"
                f" {row['neighbour_flip_rate']:.4f} | {row['confidence_rel_change_median']:.2e} |"
            )
    return "\n".join(lines) + "\n"
