"""Step 10: join scores with ground truth and compute the three metrics for one class.

This is the only module that reads labels, defect types or ground-truth masks, and it runs after every
query of the class has been scored.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from padgeo.data.base import GOOD, GroundTruth
from padgeo.eval.metrics import aupro, image_auroc, pixel_auroc


@dataclass(frozen=True)
class ClassMetrics:
    """Metrics of one class (values in [0, 1]) plus the labels they were computed from.

    Attributes:
        image_auroc: Image scores vs good/defective labels.
        pixel_auroc: Pixel maps vs ground-truth masks, all pixels of all queries pooled.
        aupro: Per-region overlap integrated up to FPR 0.3, normalised to [0, 1].
        labels: Per-query label (0 good, 1 defective; from the folder), in query order.
        defect_types: Per-query defect type ("good" for defect-free), in query order.
        empty_defect_masks: Defective queries whose ground-truth mask is empty (kept, counted).
        image_auroc_by_type: Supplementary diagnostic: image AUROC of the defect-free queries against the
            queries of each defect type alone.
    """

    image_auroc: float
    pixel_auroc: float
    aupro: float
    labels: tuple[int, ...]
    defect_types: tuple[str, ...]
    empty_defect_masks: int
    image_auroc_by_type: dict[str, float]

    @property
    def queries_per_type(self) -> dict[str, int]:
        """Number of queries per defect type."""
        return dict(sorted(Counter(self.defect_types).items()))


def load_mask(target: GroundTruth, size: tuple[int, int]) -> np.ndarray:
    """Binary ground-truth mask (any non-zero pixel is defective) of size (H0, W0).

    A defective image may have an empty mask (MAD-Sim has some, presumably defects hidden from that
    viewpoint). It keeps its folder label for image AUROC and contributes only defect-free pixels to the
    pixel metrics; `evaluate_class` counts such images.

    Raises:
        ValueError: If the mask's size differs from the image's.
    """
    if target.mask_path is None:
        return np.zeros(size, dtype=bool)
    with Image.open(target.mask_path) as image:
        mask = np.asarray(image.convert("L")) > 0
    if mask.shape != size:
        raise ValueError(f"{target.mask_path}: mask is {mask.shape}, image is {size}")
    return mask


def evaluate_class(
    query_names: Sequence[str],
    image_scores: Sequence[float],
    pixel_maps: Sequence[np.ndarray],
    ground_truth: Mapping[str, GroundTruth],
) -> ClassMetrics:
    """Image AUROC, pixel AUROC and AUPRO of one class.

    Args:
        query_names: `ImageRef.name` of each scored query.
        image_scores: Image score per query.
        pixel_maps: [H0, W0] float32 pixel map per query.
        ground_truth: Targets keyed by query name (from `AnomalyDataset.ground_truth`).

    Raises:
        KeyError: If a scored query has no ground truth.
        ValueError: If the inputs disagree in length.
    """
    if not len(query_names) == len(image_scores) == len(pixel_maps):
        raise ValueError("query_names, image_scores and pixel_maps must have the same length")
    targets = [ground_truth[name] for name in query_names]
    labels = np.array([target.label for target in targets], dtype=np.int64)
    masks = np.stack(
        [
            load_mask(target, (pixel_map.shape[0], pixel_map.shape[1]))
            for target, pixel_map in zip(targets, pixel_maps, strict=True)
        ]
    )
    maps = np.stack(pixel_maps).astype(np.float32, copy=False)
    scores = np.asarray(image_scores, dtype=np.float64)
    empty = int(sum(1 for label, mask in zip(labels, masks, strict=True) if label == 1 and not mask.any()))
    types = np.array([target.defect_type for target in targets])
    by_type = {
        defect_type: image_auroc(labels[keep], scores[keep])
        for defect_type in sorted(set(types.tolist()) - {GOOD})
        for keep in [(types == GOOD) | (types == defect_type)]
    }
    return ClassMetrics(
        image_auroc=image_auroc(labels, scores),
        pixel_auroc=pixel_auroc(masks, maps),
        aupro=aupro(masks, maps),
        labels=tuple(int(label) for label in labels),
        defect_types=tuple(target.defect_type for target in targets),
        empty_defect_masks=empty,
        image_auroc_by_type=by_type,
    )
