"""MAD-Sim loader (PAD benchmark: Zhou et al., NeurIPS 2023 Datasets and Benchmarks).

Layout per class (what is on disk is recorded in docs/DATA_NOTES.md):

    <root>/<class>/train/good/*.png                     defect-free reference pool
    <root>/<class>/test/<type>/*.png                    queries; <type> is "good" or a defect type
    <root>/<class>/ground_truth/<type>/<stem>_mask.png  masks of defective queries
    <root>/<class>/transforms.json                      camera poses of the training images

`transforms.json` is never opened here. The method must not use poses; the only permitted use is the
optional pose-accuracy diagnostic, which is not part of this codebase yet.
"""

from __future__ import annotations

from pathlib import Path

from padgeo.data.base import GOOD, AnomalyDataset, ClassSplit, GroundTruth, ImageRef, hash_file

IMAGE_SUFFIX = ".png"
MASK_SUFFIX = "_mask.png"


def _images(directory: Path) -> list[Path]:
    """Sorted image files of one directory (non-recursive).

    Raises:
        FileNotFoundError: If the directory does not exist or holds no image.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"missing directory {directory}")
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == IMAGE_SUFFIX)
    if not files:
        raise FileNotFoundError(f"no {IMAGE_SUFFIX} images in {directory}")
    return files


class MADSim(AnomalyDataset):
    """MAD-Sim classes under one root directory.

    Args:
        root: Directory containing the class folders (e.g. "01Gorilla").
    """

    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root {root} does not exist")
        self.root = root

    def _class_dir(self, class_name: str) -> Path:
        class_dir = self.root / class_name
        if not class_dir.is_dir():
            available = sorted(p.name for p in self.root.iterdir() if p.is_dir())
            raise FileNotFoundError(f"class {class_name!r} not found under {self.root}; available: {available}")
        return class_dir

    def _query_files(self, class_dir: Path) -> list[tuple[str, Path]]:
        """(type folder name, image path) of every test image, sorted by relative path."""
        test_dir = class_dir / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(f"missing directory {test_dir}")
        type_dirs = sorted(p for p in test_dir.iterdir() if p.is_dir())
        return [(type_dir.name, path) for type_dir in type_dirs for path in _images(type_dir)]

    def _ref(self, class_dir: Path, path: Path) -> ImageRef:
        return ImageRef(name=path.relative_to(class_dir).as_posix(), path=path, content_hash=hash_file(path))

    def load_split(self, class_name: str) -> ClassSplit:
        """Reference pool (train/good, sorted by content hash) and queries (test/*, sorted by name).

        Raises:
            ValueError: If a test image is byte-identical to a training image (query leakage).
        """
        class_dir = self._class_dir(class_name)
        pool = [self._ref(class_dir, path) for path in _images(class_dir / "train" / GOOD)]
        queries = [self._ref(class_dir, path) for _, path in self._query_files(class_dir)]
        leaked = {ref.content_hash for ref in pool} & {ref.content_hash for ref in queries}
        if leaked:
            names = sorted(q.name for q in queries if q.content_hash in leaked)
            raise ValueError(f"{class_name}: test images identical to training images: {names[:5]}")
        return ClassSplit(
            class_name=class_name,
            reference_pool=tuple(sorted(pool, key=lambda ref: ref.content_hash)),
            queries=tuple(sorted(queries, key=lambda ref: ref.name)),
        )

    def ground_truth(self, class_name: str) -> dict[str, GroundTruth]:
        """Label, defect type and mask path of every query, keyed by query name.

        Raises:
            FileNotFoundError: If a defective query has no mask file.
        """
        class_dir = self._class_dir(class_name)
        targets = {}
        for defect_type, path in self._query_files(class_dir):
            name = path.relative_to(class_dir).as_posix()
            if defect_type == GOOD:
                targets[name] = GroundTruth(label=0, defect_type=GOOD, mask_path=None)
                continue
            mask_path = class_dir / "ground_truth" / defect_type / f"{path.stem}{MASK_SUFFIX}"
            if not mask_path.is_file():
                raise FileNotFoundError(f"missing mask {mask_path} for {name}")
            targets[name] = GroundTruth(label=1, defect_type=defect_type, mask_path=mask_path)
        return targets
