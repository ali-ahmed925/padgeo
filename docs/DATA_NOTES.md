# MAD-Sim: what is on disk

Measured on the copy downloaded on 2026-10-11 (`gdown 1XlW5v_PCXMH49RSKICkskKjd2A5YDMQq`, `MAD-Sim.zip`,
1,016,999,653 bytes, 15,001 entries; every CRC verified). The per-class numbers are produced by:

```bash
python scripts/inspect_data.py --config configs/killtest_global.yaml --json outputs/data_inspection.json
```

## Layout and formats

- **Top level.** One folder `MAD-Sim/` with 20 class folders: `01Gorilla 02Unicorn 03Mallard 04Turtle 05Whale 06Bird 07Owl 08Sabertooth 09Swan 10Sheep 11Pig 12Zalika 13Pheonix 14Elephant 15Parrot 16Cat 17Scorpion 18Obesobeso 19Bear 20Puppy`. The kill-test classes are `03Mallard`, `04Turtle` and `18Obesobeso` (exact folder names).
- **Per class:**
  - `train/good/0.png … 209.png`;
  - `test/{Burrs,Missing,Stains,good}/<n>.png`;
  - `ground_truth/{Burrs,Missing,Stains}/<n>_mask.png`;
  - `transforms.json`.
  - Only PNGs and `transforms.json`; no stray files.
- **Images.** Every image (train and test, all classes) is **RGB** (no alpha), **800 × 800**, 8-bit, rendered on a **pure white background**: all four corners of every image are exactly (255, 255, 255). The background is exactly 255 everywhere; only the anti-aliased rim and rare specular highlights inside the object come close to it.
- **Masks.** Mode `L`, 800 × 800, values {0, 255}. Every defective test image has a mask file.
- **`transforms.json`.** NeRF/Blender format: `camera_angle_x` = 0.691 rad, plus 210 `frames`, each with `file_path` (`./train/good/<n>.png`) and a 4 × 4 `transform_matrix`. It covers the **training images only**. The method never reads it; the loader does not open it, and a test fails if anything does.
- **Model input.** VGGT's preprocessing turns 800 × 800 into 518 × 518 with no crop, so the patch grid is 37 × 37 and pixel maps are evaluated at 800 × 800.

## Counts

Each class has 210 `train/good` images; the remaining columns are test images.

| Class | good | Burrs | Missing | Stains | Defective with empty mask | Mean foreground patches / query |
|---|---|---|---|---|---|---|
| 01Gorilla | 20 | 67 | 137 | 35 | Missing 2, Stains 2 | 243 |
| 02Unicorn | 20 | 55 | 132 | 35 | Burrs 1 | 269 |
| **03Mallard** | 20 | 27 | 157 | 33 | Stains 1 | 270 |
| **04Turtle** | 20 | 32 | 130 | 35 | Stains 1 | 341 |
| 05Whale | 30 | 71 | 127 | 53 | – | 291 |
| 06Bird | 30 | 51 | 160 | 40 | – | 168 |
| 07Owl | 30 | 41 | 115 | 44 | – | 257 |
| 08Sabertooth | 36 | 58 | 136 | 47 | – | 300 |
| 09Swan | 36 | 66 | 143 | 41 | Stains 1 | 249 |
| 10Sheep | 36 | 39 | 150 | 63 | Missing 3 | 167 |
| 11Pig | 36 | 76 | 138 | 70 | – | 187 |
| 12Zalika | 36 | 56 | 130 | 57 | – | 329 |
| 13Pheonix | 36 | 86 | 150 | 69 | Stains 1 | 260 |
| 14Elephant | 36 | 72 | 149 | 55 | – | 332 |
| 15Parrot | 36 | 29 | 131 | 42 | – | 274 |
| 16Cat | 36 | 98 | 151 | 58 | Burrs 1 | 121 |
| 17Scorpion | 36 | 61 | 121 | 53 | – | 374 |
| **18Obesobeso** | 36 | 101 | 123 | 61 | – | 175 |
| 19Bear | 36 | 88 | 112 | 59 | – | 281 |
| 20Puppy | 36 | 63 | 125 | 47 | – | 319 |
| **Total** | 638 | 1,237 | 2,717 | 997 | 15 | |

The "Mean foreground patches / query" column counts the 1,369 patches of the 37 × 37 grid, with the foreground rule in `configs/`, on an evenly spaced sample of about 20 queries per class.

## Facts that matter for the experiment

1. **Test sets are dominated by defective images, and among them by Missing.**
   - Missing is 2,717 of 4,951 defective test images (55%).
   - In the kill-test classes it is 72% (Mallard), 66% (Turtle) and 43% (Obesobeso).
   - Good images are only 20–36 per class, so image AUROC rests on few negatives.
2. **Missing ground truth lies largely on background.** The table gives the mean fraction of each Missing mask's pixels that show pure white background in the query image; the masks mark where the absent part *would* be. Burrs and Stains masks lie on the object (0.00 in every class).

   | Class | Mean Missing-mask fraction on background |
   |---|---|
   | 01Gorilla | 0.46 |
   | 02Unicorn | 0.56 |
   | 03Mallard | 0.63 |
   | 04Turtle | 0.45 |
   | 05Whale | 0.47 |
   | 06Bird | 0.62 |
   | 07Owl | 0.37 |
   | 08Sabertooth | 0.47 |
   | 09Swan | 0.58 |
   | 10Sheep | 0.55 |
   | 11Pig | 0.45 |
   | 12Zalika | 0.47 |
   | 13Pheonix | 0.56 |
   | 14Elephant | 0.72 |
   | 15Parrot | 0.50 |
   | 16Cat | 0.58 |
   | 17Scorpion | 0.14 |
   | 18Obesobeso | 0.45 |
   | 19Bear | 0.40 |
   | 20Puppy | 0.46 |

   Background patches are excluded from scoring by the brief's rule, so these pixels score 0 for both methods. On a development run (`global_nn`, ViT-S, Mallard, seed 0), image AUROC of good vs Missing alone was 0.445, against 0.650 for Burrs and 0.742 for Stains (`outputs/m3_global_vits_mallard`).
3. **Empty masks.** 15 defective images have an all-zero mask, presumably because the defect is not visible from that viewpoint. They keep their folder label and are counted in `results.json` (`defective_with_empty_mask`).
4. **No leakage between splits.** No test image is byte-identical to a training image, and no class has duplicate files (checked by content hash with `scripts/inspect_data.py`). The loader itself raises if a test image duplicates a training image.
5. **Object size.** Foreground covers 121–374 of 1,369 patches (9–27%). A patch is therefore roughly 1/11–1/20 of the object's width in the image.
6. **Poses.** Training images come with poses (`transforms.json`); test images do not. View coverage of the 210 training renders was not analysed, because the method does not use poses.
