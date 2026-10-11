"""Tabulate experiments side by side (mean ± std over seeds) next to published reference numbers.

Reads only result files. The experiment directories are taken from the given configs, and the runs must
differ only in their method section (checked). The markdown is printed and written to --out.

Usage:
    python scripts/make_table.py --configs configs/killtest_global.yaml configs/killtest_anchored.yaml \
        --reference configs/reference_numbers.yaml --out docs/RESULTS.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from padgeo.config import load_config
from padgeo.eval.summary import load_reference_numbers, load_run_set, render_markdown


def main(argv: list[str] | None = None) -> str:
    """Build the table from result files, write it to --out, print it, and return the markdown."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--configs", type=Path, nargs="+", required=True, help="experiment configs to compare"
    )
    parser.add_argument("--reference", type=Path, required=True, help="published reference numbers YAML")
    parser.add_argument("--out", type=Path, required=True, help="markdown output file")
    parser.add_argument("--title", default="Kill test: global_nn vs anchored_nn", help="section title")
    args = parser.parse_args(argv)
    configs = [load_config(path) for path in args.configs]
    run_sets = [load_run_set(cfg) for cfg in configs]
    markdown = render_markdown(run_sets, load_reference_numbers(args.reference), args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    print(markdown)
    return markdown


if __name__ == "__main__":
    main()
