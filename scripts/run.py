"""Run one experiment config end to end (all seeds, all classes) and write per-seed result files.

Usage:
    python scripts/run.py --config configs/killtest_anchored.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from padgeo.config import load_config
from padgeo.pipeline import ExperimentRunner
from padgeo.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> list[Path]:
    """Parse arguments, run the experiment, and return the per-seed result directories."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML config")
    args = parser.parse_args(argv)
    setup_logging()
    return ExperimentRunner(load_config(args.config)).run()


if __name__ == "__main__":
    main()
