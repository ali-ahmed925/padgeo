"""Image AUROC, pixel AUROC and AUPRO. All return values in [0, 1].

- Image and pixel AUROC: `sklearn.metrics.roc_auc_score` (pixel AUROC pools every pixel of every query of a
  class, as in the MVTec AD and PAD evaluations).
- AUPRO: the official MVTec AD evaluation code (`compute_pro` and `trapezoid`), as distributed in SplatPose
  (Kruse et al., CVPRW 2024), the reference implementation used for MAD in the pose-agnostic AD
  literature. It is loaded unmodified from the pinned git submodule `third_party/SplatPose` (commit
  2c07b6d, file `aupro.py`), checked against its SHA-256, and called exactly as SplatPose's
  `calculate_au_pro_au_roc` does: AUPRO = trapezoid(fpr, pro, x_max=0.3) / 0.3. That is the area under
  the per-region-overlap curve up to a false-positive rate of 0.3, normalised by 0.3. Regions are
  8-connected components of the ground truth, and every distinct score is a threshold (no discretisation).
  The code is used through the submodule rather than copied because the SplatPose repository states no
  licence for that file.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from sklearn.metrics import roc_auc_score

FPR_LIMIT = 0.3
_AUPRO_FILE = Path(__file__).resolve().parents[2] / "third_party" / "SplatPose" / "aupro.py"
_AUPRO_SHA256 = "28c5424f0d731e80b0b42a1a19b12919512789b6b4dcd255520f68745de27ee9"


@functools.cache
def _mvtec_aupro_module() -> ModuleType:
    """Import SplatPose's `aupro.py` from the submodule after verifying it is the pinned file.

    Raises:
        FileNotFoundError: If the submodule is not checked out.
        RuntimeError: If the file differs from the pinned version.
    """
    if not _AUPRO_FILE.exists():
        raise FileNotFoundError(
            f"{_AUPRO_FILE} missing; run `git submodule update --init third_party/SplatPose`"
        )
    digest = hashlib.sha256(_AUPRO_FILE.read_bytes()).hexdigest()
    if digest != _AUPRO_SHA256:
        raise RuntimeError(f"{_AUPRO_FILE} has SHA-256 {digest}, expected the pinned {_AUPRO_SHA256}")
    spec = importlib.util.spec_from_file_location("splatpose_mvtec_aupro", _AUPRO_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_AUPRO_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_binary_targets(name: str, targets: np.ndarray) -> None:
    values = np.unique(targets)
    if not set(values.tolist()) <= {0, 1}:
        raise ValueError(f"{name}: targets must be binary, got values {values[:10]}")
    if values.size < 2:
        raise ValueError(f"{name}: needs both positive and negative targets, got only {values.tolist()}")


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC of image scores against labels (0 good, 1 defective).

    Raises:
        ValueError: If labels are not binary or only one class is present.
    """
    labels = np.asarray(labels)
    _check_binary_targets("image labels", labels)
    if labels.shape != np.shape(scores):
        raise ValueError(f"labels {labels.shape} and scores {np.shape(scores)} differ in shape")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """AUROC over all pixels of all images: masks [Q, H, W] bool vs maps [Q, H, W] float.

    Raises:
        ValueError: If shapes differ, or no defective (or no defect-free) pixel exists.
    """
    if masks.shape != maps.shape:
        raise ValueError(f"masks {masks.shape} and maps {maps.shape} differ in shape")
    _check_binary_targets("pixel masks", masks.astype(np.uint8))
    return float(roc_auc_score(masks.ravel(), maps.ravel()))


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = FPR_LIMIT) -> float:
    """Normalised area under the PRO curve up to `fpr_limit`, via the MVTec AD reference implementation.

    Args:
        masks: [Q, H, W] bool ground-truth masks (all-False for defect-free images).
        maps: [Q, H, W] float anomaly maps.
        fpr_limit: Integration limit of the false-positive rate (0.3 everywhere in this project).

    Raises:
        ValueError: If shapes differ or there is no defect region.
    """
    if masks.shape != maps.shape:
        raise ValueError(f"masks {masks.shape} and maps {maps.shape} differ in shape")
    _check_binary_targets("pixel masks", masks.astype(np.uint8))
    mvtec = _mvtec_aupro_module()
    fprs, pros = mvtec.compute_pro(
        anomaly_maps=[m.astype(np.float32) for m in maps],
        ground_truth_maps=[m.astype(np.uint8) for m in masks],
    )
    return float(mvtec.trapezoid(fprs, pros, x_max=fpr_limit) / fpr_limit)
