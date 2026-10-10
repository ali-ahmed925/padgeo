"""Side-by-side figure of one query: image, ground truth, each method's heatmap, and the unseen mask.

Heatmaps are rebuilt from the stored patch scores with the shared `make_pixel_map`, so the figure shows
exactly the maps that were evaluated. All heatmaps share one colour scale, so the panels can be compared
by eye.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from padgeo.methods.aggregation import make_pixel_map


@dataclass(frozen=True)
class StoredQuery:
    """One query's stored outputs from a run's `scores/<class>.npz` and `per_query.csv`.

    Attributes:
        patch_scores: [h, w] float32.
        unseen: [h, w] bool (both unseen reasons).
        image_score: Image score.
        references: Space-separated reference names.
    """

    patch_scores: np.ndarray
    unseen: np.ndarray
    image_score: float
    references: str


def load_stored_query(run_dir: Path, class_name: str, query: str) -> StoredQuery:
    """Read one query's outputs from a seed result directory.

    Raises:
        KeyError: If the query is not in the run.
    """
    with np.load(run_dir / "scores" / f"{class_name}.npz") as scores:
        names = scores["query"].tolist()
        if query not in names:
            raise KeyError(f"{query} not in {run_dir} ({class_name})")
        index = names.index(query)
        unseen = scores["unseen_no_neighbour"][index] | scores["unseen_low_confidence"][index]
        patch_scores = scores["patch_scores"][index]
        image_score = float(scores["image_score"][index])
    with (run_dir / "per_query.csv").open(encoding="utf-8") as handle:
        references = next(
            row["references"]
            for row in csv.DictReader(handle)
            if row["class"] == class_name and row["query"] == query
        )
    return StoredQuery(
        patch_scores=patch_scores, unseen=unseen, image_score=image_score, references=references
    )


def upsample_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsampling of an [h, w] bool patch mask to (H0, W0)."""
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(tensor, size=size, mode="nearest")[0, 0].numpy() > 0.5


def render_query_figure(
    image: np.ndarray,
    ground_truth: np.ndarray,
    heatmaps: Sequence[tuple[str, StoredQuery]],
    sigma: float,
    title: str,
    out_path: Path,
) -> None:
    """Save the figure: image | ground truth | one heatmap per run | unseen mask of each geometry run.

    Args:
        image: [H0, W0, 3] uint8 RGB query image.
        ground_truth: [H0, W0] bool defect mask.
        heatmaps: (label, stored outputs) per run, in display order.
        sigma: Gaussian sigma used by the runs (identical across compared runs).
        title: Figure title.
        out_path: PNG file to write.
    """
    size = (image.shape[0], image.shape[1])
    maps = [make_pixel_map(torch.from_numpy(stored.patch_scores), size, sigma) for _, stored in heatmaps]
    vmax = max(float(pixel_map.max()) for pixel_map in maps) or 1.0
    unseen_panels = [(label, stored) for label, stored in heatmaps if stored.unseen.any()]
    num_panels = 2 + len(maps) + len(unseen_panels)
    figure, axes = plt.subplots(1, num_panels, figsize=(3.2 * num_panels, 3.6))
    axes[0].imshow(image)
    axes[0].set_title("query")
    axes[1].imshow(image)
    axes[1].imshow(np.ma.masked_where(~ground_truth, ground_truth), cmap="autumn", alpha=0.6)
    axes[1].set_title("ground truth")
    for axis, (label, stored), pixel_map in zip(axes[2 : 2 + len(maps)], heatmaps, maps, strict=True):
        axis.imshow(image)
        axis.imshow(pixel_map, cmap="jet", alpha=0.5, vmin=0.0, vmax=vmax)
        axis.set_title(f"{label}\nimage score {stored.image_score:.3f}")
    for axis, (label, stored) in zip(axes[2 + len(maps) :], unseen_panels, strict=True):
        unseen = upsample_mask(stored.unseen, size)
        axis.imshow(image)
        axis.imshow(np.ma.masked_where(~unseen, unseen), cmap="cool", alpha=0.6)
        axis.set_title(f"{label}: unseen")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.9))  # leave room for the suptitle
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
