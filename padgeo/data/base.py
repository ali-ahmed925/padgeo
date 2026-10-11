"""Dataset interface shared by all loaders.

The interface separates what the scoring path may see from what only evaluation may see:

- `AnomalyDataset.load_split` returns image references only: the reference pool (defect-free training
  images) and the query images. An `ImageRef` carries a path, used solely to read pixels, and a content
  hash, used for seeding and caching. Nothing on the scoring path derives information from the path.
- `AnomalyDataset.ground_truth` returns labels, defect types and mask paths. Only the evaluation step
  (`padgeo.eval.evaluate`) calls it.

A new dataset (for example PIAD or MAD-Real) is added by implementing these two methods and registering
it in `padgeo.data.registry`; the methods and the rest of the pipeline do not change.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

GOOD = "good"


@dataclass(frozen=True)
class ImageRef:
    """One image file.

    Attributes:
        name: Path relative to the class directory, e.g. "test/Burrs/0.png". Used to label outputs and to
            join scores with ground truth at evaluation time; never used for scoring.
        path: Absolute path, used only to read pixels.
        content_hash: SHA-256 hex digest of the file bytes. Identifies the image for seeding and caching,
            so neither depends on file or folder names.
    """

    name: str
    path: Path
    content_hash: str


@dataclass(frozen=True)
class ClassSplit:
    """The images of one object class visible to the scoring path.

    Attributes:
        class_name: Dataset class folder name, e.g. "03Mallard".
        reference_pool: Defect-free training images, sorted by content hash (a canonical order that does
            not depend on file names).
        queries: Test images (defect-free and defective), sorted by name.
    """

    class_name: str
    reference_pool: tuple[ImageRef, ...]
    queries: tuple[ImageRef, ...]


@dataclass(frozen=True)
class GroundTruth:
    """Evaluation targets of one query image.

    Attributes:
        label: 0 for defect-free, 1 for defective.
        defect_type: Defect category folder name, or "good".
        mask_path: Pixel mask of the defect, or None for defect-free images (an all-zero mask).
    """

    label: int
    defect_type: str
    mask_path: Path | None


def hash_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AnomalyDataset(ABC):
    """A few-shot anomaly-detection dataset organised by object class."""

    @abstractmethod
    def load_split(self, class_name: str) -> ClassSplit:
        """Image references for one class (reference pool and queries), without any labels."""

    @abstractmethod
    def ground_truth(self, class_name: str) -> dict[str, GroundTruth]:
        """Evaluation targets for every query of one class, keyed by `ImageRef.name`."""
