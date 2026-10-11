"""Inspect a MAD-Sim copy: image counts per class, split and defect type, and the data facts.

The facts are the ones recorded in docs/DATA_NOTES.md:

- image sizes and modes;
- background purity;
- duplicates;
- empty defect masks;
- how much of each defect type's ground truth lies on background pixels;
- foreground patches per image;
- one sample's tensor shapes.

Usage:
    python scripts/inspect_data.py --config configs/killtest_global.yaml \
        [--classes 01Gorilla ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from padgeo.config import ForegroundConfig, load_config
from padgeo.data.base import GOOD, AnomalyDataset
from padgeo.data.images import PATCH_SIZE, load_model_input
from padgeo.data.registry import build_dataset
from padgeo.eval.evaluate import load_mask
from padgeo.features.foreground import patch_foreground, pixel_foreground


def inspect_class(dataset: AnomalyDataset, class_name: str, foreground: ForegroundConfig) -> dict[str, Any]:
    """Counts and data facts for one class (reads ground truth: this is a data-inspection script)."""
    split = dataset.load_split(class_name)
    targets = dataset.ground_truth(class_name)
    images = [*split.reference_pool, *split.queries]
    formats: Counter[str] = Counter()
    corners_white = 0
    for ref in images:
        with Image.open(ref.path) as image:
            formats[f"{image.mode} {image.size[0]}x{image.size[1]}"] += 1
            pixels = np.asarray(image.convert("RGB"))
        corners_white += all((pixels[r, c] == 255).all() for r in (0, -1) for c in (0, -1))
    hashes = Counter(ref.content_hash for ref in images)
    on_background: dict[str, list[float]] = defaultdict(list)
    empty_masks: Counter[str] = Counter()
    for query in split.queries:
        target = targets[query.name]
        if target.defect_type == GOOD:
            continue
        with Image.open(query.path) as image:
            pixels = np.asarray(image.convert("RGB"))
        mask = load_mask(target, (pixels.shape[0], pixels.shape[1]))
        if not mask.any():
            empty_masks[target.defect_type] += 1
            continue
        white = (pixels == 255).all(axis=-1)
        on_background[target.defect_type].append(float((mask & white).sum() / mask.sum()))
    sample = load_model_input(split.queries[0].path)
    pixel_mask = pixel_foreground(sample.pixels, foreground)
    foreground_patches = [
        int(
            patch_foreground(
                pixel_foreground(load_model_input(ref.path).pixels, foreground), PATCH_SIZE, foreground
            ).sum()
        )
        for ref in split.queries[:: max(1, len(split.queries) // 20)]
    ]
    return {
        "train_good": len(split.reference_pool),
        "test": dict(sorted(Counter(t.defect_type for t in targets.values()).items())),
        "formats": dict(formats),
        "images_with_white_corners": f"{corners_white}/{len(images)}",
        "duplicate_files": sum(count - 1 for count in hashes.values() if count > 1),
        "defective_with_empty_mask": dict(empty_masks),
        "mask_fraction_on_white_background_mean": {
            k: float(np.mean(v)) for k, v in sorted(on_background.items())
        },
        "foreground_patches_per_query_mean": float(np.mean(foreground_patches)),
        "sample": {
            "query": split.queries[0].name,
            "original_size": sample.original_size,
            "model_input": list(sample.pixels.shape),
            "pixel_foreground_fraction": float(pixel_mask.float().mean()),
            "patch_grid": [sample.pixels.shape[1] // PATCH_SIZE, sample.pixels.shape[2] // PATCH_SIZE],
        },
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Print one block per class and return all statistics."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="experiment config (dataset root, masks)")
    parser.add_argument("--classes", nargs="*", help="classes to inspect (default: every class folder)")
    parser.add_argument("--json", type=Path, help="also write the statistics to this JSON file")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    dataset = build_dataset(config.data)
    classes = args.classes or sorted(p.name for p in config.data.root.iterdir() if p.is_dir())
    stats = {class_name: inspect_class(dataset, class_name, config.foreground) for class_name in classes}
    for class_name, values in stats.items():
        print(f"== {class_name}")
        for key, value in values.items():
            print(f"   {key}: {value}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
