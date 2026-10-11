"""Test helpers: a tiny dataset in MAD-Sim's on-disk format, experiment configs for it, and a guard
that fails any read of `transforms.json`.

The fixture mirrors what docs/DATA_NOTES.md records about MAD-Sim: RGB PNGs on a pure white background,
`test/<type>/<n>.png` with `ground_truth/<type>/<n>_mask.png` masks holding {0, 255}, a Missing defect whose
mask covers pixels that now show background, and a `transforms.json` that must never be read (it is not
valid JSON). It also contains one blank query with no foreground at all (a degenerate sample that must be
counted, not dropped), and one defective query whose mask is empty (kept as defective and counted).
"""

from __future__ import annotations

import builtins
import contextlib
import math
import pathlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

SIZE = 112  # small originals; the model input is still 518 x 518 (VGGT's preprocessing resizes)
CLASSES = ("01Alpha", "02Beta")
PALETTE = {
    "01Alpha": [(200, 200, 200), (120, 80, 40), (210, 180, 40)],
    "02Beta": [(90, 140, 200), (60, 60, 60), (200, 70, 70)],
}


def render_object(class_name: str, angle: float, scale: float = 1.0) -> Image.Image:
    """A three-part toy-like object rotated in the image plane, on pure white."""
    image = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    centre = SIZE / 2
    for part, colour in enumerate(PALETTE[class_name]):
        radius = scale * (26 - 6 * part)
        offset = scale * (10 * part)
        cx = centre + offset * math.cos(angle)
        cy = centre + offset * math.sin(angle)
        corners = [
            (cx + radius * math.cos(angle + k * math.pi / 2), cy + radius * math.sin(angle + k * math.pi / 2))
            for k in range(4)
        ]
        draw.polygon(corners, fill=colour)
    return image


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def build_dataset(root: Path) -> Path:
    """Write the fixture dataset under `root` and return the dataset root."""
    for c, class_name in enumerate(CLASSES):
        class_dir = root / class_name
        for i in range(8):
            path = class_dir / "train" / "good" / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            render_object(class_name, angle=0.4 * i + 0.1 * c).save(path)
        (class_dir / "transforms.json").write_text("NOT JSON: the method must never read this file")
        test = class_dir / "test"
        for i in range(3):
            path = test / "good" / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            render_object(class_name, angle=0.4 * i + 0.2 + 0.1 * c).save(path)
        blank = test / "good" / "3.png"
        Image.new("RGB", (SIZE, SIZE), (255, 255, 255)).save(blank)  # no foreground at all
        for i in range(2):
            # Stains: a dark blob on the object.
            image = render_object(class_name, angle=0.5 * i + 0.3)
            mask = np.zeros((SIZE, SIZE), dtype=bool)
            mask[50:62, 50:62] = True
            pixels = np.asarray(image).copy()
            pixels[mask] = (20, 20, 120)
            (test / "Stains").mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(test / "Stains" / f"{i}.png")
            save_mask(class_dir / "ground_truth" / "Stains" / f"{i}_mask.png", mask)
            # Missing: a part is erased; its mask covers pixels that now show background.
            image = render_object(class_name, angle=0.5 * i + 0.7)
            reference = np.asarray(image).copy()
            mask = np.zeros((SIZE, SIZE), dtype=bool)
            mask[45:55, 45:55] = (
                True  # inside the largest part at any angle (its inscribed circle covers 38..74)
            )
            mask &= (reference != 255).any(-1)
            assert mask.any()
            reference[mask] = 255
            (test / "Missing").mkdir(parents=True, exist_ok=True)
            Image.fromarray(reference).save(test / "Missing" / f"{i}.png")
            save_mask(class_dir / "ground_truth" / "Missing" / f"{i}_mask.png", mask)
        # A defective image whose defect is not visible: its mask is empty (MAD-Sim has 15 of these).
        render_object(class_name, angle=1.9).save(test / "Stains" / "2.png")
        save_mask(class_dir / "ground_truth" / "Stains" / "2_mask.png", np.zeros((SIZE, SIZE), dtype=bool))
    return root


def experiment_config(
    name: str, root: Path, work: Path, method: dict, *, seeds: tuple[int, ...] = (0, 1), batch_size: int = 4
) -> dict:
    """A small experiment on the fixture: ViT-S on CPU, N = 2 references."""
    return {
        "experiment": {"name": name, "output_dir": str(work / "outputs"), "seeds": list(seeds)},
        "data": {
            "dataset": "mad_sim",
            "root": str(root),
            "classes": list(CLASSES),
            "num_references": 2,
            "sampling": "per_query",
        },
        "features": {
            "backbone": "dinov2_vits14",
            "precision": "fp32",
            "device": "cpu",
            "batch_size": batch_size,
        },
        "foreground": {"pixel_threshold": 0.98, "min_patch_fraction": 0.1},
        "method": method,
        "aggregation": {"image_score": {"mode": "top_percent", "percent": 1.0}, "pixel_sigma": 4.0},
        "cache": {"dir": str(work / "cache")},
    }


GLOBAL_METHOD = {"name": "global_nn", "device": "cpu"}
ANCHORED_METHOD = {
    "name": "anchored_nn",
    "device": "cpu",
    "geometry": {"backend": "fake", "point_source": "depth", "precision": "fp32", "device": "cpu"},
    "neighbourhood": {"mode": "radius", "radius": 0.05},
    "unseen": "fallback_global",
    "min_confidence": 1.5,
}


def write_config(path: Path, config: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@contextlib.contextmanager
def transforms_json_forbidden() -> Iterator[None]:
    """Fail any attempt to open a `transforms.json` (through `open` or `Path.open`) inside the block."""
    real_open, real_path_open = builtins.open, pathlib.Path.open

    def guarded_open(file, *args, **kwargs):
        assert Path(str(file)).name != "transforms.json", "transforms.json must never be read"
        return real_open(file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        assert self.name != "transforms.json", "transforms.json must never be read"
        return real_path_open(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", guarded_open)
        patch.setattr(pathlib.Path, "open", guarded_path_open)
        yield
