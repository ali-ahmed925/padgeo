"""Profile the geometry backend of an anchored_nn experiment (the first step of the kill test).

Reports peak VRAM and seconds per query for each N and precision, and how much the 3D points move
between precisions.

Usage:
    python scripts/profile_geometry.py --config configs/killtest_profile.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from padgeo.config import load_profile_config
from padgeo.geometry.profiling import render_profile, run_profile
from padgeo.utils.logging import setup_logging
from padgeo.utils.provenance import collect_provenance
from padgeo.utils.seeding import seed_everything


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the profile, write `report.json` and `profile.md`, print the summary, return the report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="profile YAML config")
    args = parser.parse_args(argv)
    profile, experiment = load_profile_config(args.config)
    out_dir = profile.output_dir / profile.name
    setup_logging(out_dir / "profile.log")
    seed_everything(experiment.experiment.seeds[0])
    report = run_profile(profile, experiment)
    report["provenance"] = collect_provenance()
    report["profile_config"] = profile.model_dump(mode="json")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render_profile(report)
    (out_dir / "profile.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return report


if __name__ == "__main__":
    main()
