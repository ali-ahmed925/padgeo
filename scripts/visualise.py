"""Save a side-by-side figure per chosen query: image, ground truth, each run's heatmap, and the unseen mask.

The runs must be comparable (same data, references, masks and aggregation; checked). They must also have
used the same references for the query (checked).

Usage:
    python scripts/visualise.py --configs configs/killtest_global.yaml configs/killtest_anchored.yaml \
        --seed 0 --class-name 03Mallard --query test/Burrs/0.png --out outputs/figures
    python scripts/visualise.py ... --first-per-type --out outputs/figures   # every class of the config
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from padgeo.config import Config, load_config
from padgeo.data.registry import build_dataset
from padgeo.eval.evaluate import load_mask
from padgeo.eval.summary import check_comparable, load_run_set
from padgeo.utils.visualise import load_stored_query, render_query_figure


def load_display_image(path: Path) -> np.ndarray:
    """[H0, W0, 3] uint8 RGB, with any alpha composited onto white (as the models see it)."""
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(white, rgba).convert("RGB"))


def class_figures(
    configs: list[Config], class_name: str, seed: int, queries: list[str] | None, out_dir: Path
) -> list[Path]:
    """Figures for one class: the given queries, or the first evaluated query of each defect type."""
    dataset = build_dataset(configs[0].data)
    references = {ref.name: ref for ref in dataset.load_split(class_name).queries}
    ground_truth = dataset.ground_truth(class_name)
    run_dirs = [cfg.experiment.output_dir / cfg.experiment.name / f"seed_{seed}" for cfg in configs]
    if queries is None:
        with np.load(run_dirs[0] / "scores" / f"{class_name}.npz") as scores_file:
            evaluated = sorted(scores_file["query"].tolist())
        queries = list({ground_truth[name].defect_type: name for name in reversed(evaluated)}.values())
    written = []
    for query in sorted(queries):
        stored = [
            (cfg.experiment.name, load_stored_query(run_dir, class_name, query))
            for cfg, run_dir in zip(configs, run_dirs, strict=True)
        ]
        if len({item.references for _, item in stored}) != 1:
            raise ValueError(f"{query}: the runs used different references")
        image = load_display_image(references[query].path)
        mask = load_mask(ground_truth[query], (image.shape[0], image.shape[1]))
        out_path = out_dir / f"{class_name}_{query.replace('/', '_')}_seed{seed}.png"
        title = f"{class_name} {query} ({ground_truth[query].defect_type}), seed {seed}"
        render_query_figure(image, mask, stored, configs[0].aggregation.pixel_sigma, title, out_path)
        written.append(out_path)
    return written


def main(argv: list[str] | None = None) -> list[Path]:
    """Write one PNG per query and return their paths."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--configs", type=Path, nargs="+", required=True, help="experiment configs to show")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--class-name", help="class folder name, e.g. 03Mallard (default: every class)")
    chosen = parser.add_mutually_exclusive_group(required=True)
    chosen.add_argument("--query", nargs="+", help="query names of --class-name, e.g. test/Burrs/0.png")
    chosen.add_argument(
        "--first-per-type",
        action="store_true",
        help="the first evaluated query (by name) of each defect type",
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args(argv)
    if args.query and not args.class_name:
        parser.error("--query needs --class-name")

    configs = [load_config(path) for path in args.configs]
    check_comparable([load_run_set(cfg) for cfg in configs])
    classes = [args.class_name] if args.class_name else configs[0].data.classes
    return [
        path
        for class_name in classes
        for path in class_figures(configs, class_name, args.seed, args.query, args.out)
    ]


if __name__ == "__main__":
    main()
