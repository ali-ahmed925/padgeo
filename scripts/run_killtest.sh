#!/usr/bin/env bash
# Kill test: profile the geometry backend, run global_nn and anchored_nn (3 classes x 3 seeds), tabulate both
# next to the published reference numbers, and draw one figure per defect type and class.
#
#   bash scripts/run_killtest.sh           # experiment machine: VGGT-1B + DINOv2 ViT-L -> docs/RESULTS.md
#   bash scripts/run_killtest.sh --smoke   # development machine: fake geometry + ViT-S, 8 queries per class
#
# Run from anywhere; paths are resolved from the repository root. Any failing step stops the script.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--smoke" ]]; then
  CONFIGS=configs/smoke
  TABLE=outputs/smoke_killtest_RESULTS.md
  FIGURES=outputs/smoke_killtest_figures
  TITLE="Smoke kill test (fake geometry, ViT-S; numbers are meaningless)"
elif [[ $# -eq 0 ]]; then
  CONFIGS=configs
  TABLE=docs/RESULTS.md
  FIGURES=outputs/killtest_figures
  TITLE="Kill test: global_nn vs anchored_nn"
else
  echo "usage: $0 [--smoke]" >&2
  exit 2
fi

GLOBAL="$CONFIGS/killtest_global.yaml"
ANCHORED="$CONFIGS/killtest_anchored.yaml"

echo "== 1/4 geometry profile (peak VRAM, s/query, precision effect)"
python scripts/profile_geometry.py --config "$CONFIGS/killtest_profile.yaml"

echo "== 2/4 global_nn"
python scripts/run.py --config "$GLOBAL"

echo "== 3/4 anchored_nn"
python scripts/run.py --config "$ANCHORED"

echo "== 4/4 table and figures"
python scripts/make_table.py --configs "$GLOBAL" "$ANCHORED" \
  --reference configs/reference_numbers.yaml --out "$TABLE" --title "$TITLE"
for class_name in 03Mallard 04Turtle 18Obesobeso; do
  python scripts/visualise.py --configs "$GLOBAL" "$ANCHORED" --seed 0 --class-name "$class_name" \
    --first-per-type --out "$FIGURES"
done
echo "Table: $TABLE   Figures: $FIGURES"
