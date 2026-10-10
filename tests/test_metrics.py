"""Metric functions against hand-computed toy cases, and the evaluation join with ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from padgeo.data.base import GroundTruth
from padgeo.eval.evaluate import evaluate_class, load_mask
from padgeo.eval.metrics import aupro, image_auroc, pixel_auroc


def test_image_auroc_hand_computed():
    # Positive/negative pairs: 0.35 > 0.1, 0.35 < 0.4, 0.8 > both  ->  3 of 4 correctly ordered.
    assert image_auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)


def test_image_auroc_requires_both_classes():
    with pytest.raises(ValueError, match="both positive and negative"):
        image_auroc(np.array([1, 1]), np.array([0.2, 0.3]))


def test_pixel_auroc_hand_computed():
    masks = np.array([[[True, False], [False, False]]])
    maps = np.array([[[0.9, 0.1], [0.95, 0.2]]], dtype=np.float32)
    # One positive (0.9) against negatives {0.1, 0.95, 0.2}: ranked above 2 of 3.
    assert pixel_auroc(masks, maps) == pytest.approx(2 / 3)


def two_region_case(missed_score: float, ok_score: float) -> tuple[np.ndarray, np.ndarray]:
    """Image 0: region A (4 px, score 1). Image 1: region B (4 px, `missed_score`). Rest: `ok_score`."""
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 2:4, 2:4] = True
    masks[1, 6:8, 6:8] = True
    maps = np.full((2, 10, 10), ok_score, dtype=np.float32)
    maps[0][masks[0]] = 1.0
    maps[1][masks[1]] = missed_score
    return masks, maps


def test_aupro_perfect_and_inverted():
    masks, _ = two_region_case(0.0, 0.0)
    assert aupro(masks, masks.astype(np.float32)) == pytest.approx(1.0)
    assert aupro(masks, (~masks).astype(np.float32)) == pytest.approx(0.0)


def test_aupro_half_of_the_regions_detected():
    # Thresholds: 1.0 -> (FPR 0, PRO 0.5); 0.5 -> (1, 0.5); 0.0 -> (1, 1). Up to FPR 0.3, PRO = 0.5.
    masks, maps = two_region_case(missed_score=0.0, ok_score=0.5)
    assert aupro(masks, maps) == pytest.approx(0.5)


def test_aupro_tie_between_missed_region_and_background():
    # Thresholds: 1.0 -> (0, 0.5); 0.0 -> (1, 1). The reference code interpolates linearly between the two
    # points: PRO(0.3) = 0.65, area = 0.5 * (0.5 + 0.65) * 0.3 = 0.1725, normalised by 0.3 -> 0.575.
    masks, maps = two_region_case(missed_score=0.0, ok_score=0.0)
    assert aupro(masks, maps) == pytest.approx(0.575)


def test_aupro_counts_regions_not_pixels():
    # One large detected region and one small missed region weigh equally (per-region overlap).
    masks = np.zeros((1, 20, 20), dtype=bool)
    masks[0, 0:10, 0:10] = True  # 100 px, detected
    masks[0, 15:16, 15:16] = True  # 1 px, missed
    maps = np.full((1, 20, 20), 0.5, dtype=np.float32)
    maps[0, 0:10, 0:10] = 1.0
    maps[0, 15, 15] = 0.0
    assert aupro(masks, maps) == pytest.approx(0.5)


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="differ in shape"):
        pixel_auroc(np.zeros((1, 2, 2), dtype=bool), np.zeros((1, 2, 3), dtype=np.float32))


def write_mask(path: Path, mask: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return path


def test_load_mask_checks_size_and_keeps_empty_defect_masks(tmp_path):
    good = GroundTruth(label=0, defect_type="good", mask_path=None)
    assert not load_mask(good, (4, 5)).any()
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 2] = True
    target = GroundTruth(label=1, defect_type="Burrs", mask_path=write_mask(tmp_path / "m.png", mask))
    assert np.array_equal(load_mask(target, (4, 5)), mask)
    with pytest.raises(ValueError, match="mask is"):
        load_mask(target, (5, 5))
    # A defective image may have an empty mask: it is loaded (all False), not rejected.
    empty = GroundTruth(
        label=1, defect_type="Burrs", mask_path=write_mask(tmp_path / "e.png", np.zeros((4, 5)))
    )
    assert not load_mask(empty, (4, 5)).any()


def test_evaluate_class_joins_by_query_name(tmp_path):
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    ground_truth = {
        "test/good/0.png": GroundTruth(0, "good", None),
        "test/Burrs/0.png": GroundTruth(1, "Burrs", write_mask(tmp_path / "0_mask.png", mask)),
    }
    perfect = mask.astype(np.float32)
    metrics = evaluate_class(
        ["test/Burrs/0.png", "test/good/0.png"],
        [0.9, 0.1],
        [perfect, np.zeros((4, 4), np.float32)],
        ground_truth,
    )
    assert metrics.image_auroc == pytest.approx(1.0)
    assert metrics.pixel_auroc == pytest.approx(1.0)
    assert metrics.aupro == pytest.approx(1.0)
    assert metrics.labels == (1, 0)
    assert metrics.queries_per_type == {"Burrs": 1, "good": 1}
    assert metrics.empty_defect_masks == 0
    assert metrics.image_auroc_by_type == {"Burrs": 1.0}
    with pytest.raises(KeyError):
        evaluate_class(["test/Missing/9.png"], [0.5], [perfect], ground_truth)


def test_image_auroc_by_defect_type_hand_computed(tmp_path):
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask_path = write_mask(tmp_path / "m.png", mask)
    ground_truth = {
        "g0": GroundTruth(0, "good", None),
        "g1": GroundTruth(0, "good", None),
        "b0": GroundTruth(1, "Burrs", mask_path),
        "m0": GroundTruth(1, "Missing", mask_path),
    }
    names = ["g0", "g1", "b0", "m0"]
    scores = [0.2, 0.6, 0.9, 0.4]  # Burrs above both good queries; Missing above one of two
    maps = [mask.astype(np.float32) * s for s in scores]
    metrics = evaluate_class(names, scores, maps, ground_truth)
    assert metrics.image_auroc_by_type == {"Burrs": 1.0, "Missing": 0.5}
    assert metrics.image_auroc == pytest.approx(0.75)
