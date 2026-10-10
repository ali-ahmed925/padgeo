# Decisions

This file lists every choice the brief left open, with the reason for each. Any choice that affects scores
applies identically to both methods and was fixed before any real experiment ran. Sources: `UNDERSTANDING.md`
(design), `DATA_NOTES.md` (dataset facts), and the code at the commits pinned below.

## 1. Environment and pinned code

| What | Choice | Why |
|---|---|---|
| Python, torch | 3.11; torch 2.3.1 + torchvision 0.18.1 (CUDA 12.1 wheels); numpy 1.26.4 | torch 2.3.1 is the version VGGT's `requirements.txt` pins. VGGT needs numpy < 2. The maintainers say torch ≥ 2.3 is needed for flash attention through SDPA. |
| VGGT code | `pip install` from git, commit `a288dd0f14786c93483e45524328726ab7b1b4ce` (latest `main`, 2026-05-19) | The brief asks for an installed dependency rather than copied code. This commit includes the May 2026 memory fix: the aggregator caches only layers 4/11/17/23. |
| VGGT weights | `facebook/VGGT-1B`, Hugging Face revision `860abec7937da0a4c03c41d3c269c366e82abdf9`, loaded with `strict=True` | A pinned revision gives reproducibility. `huggingface_hub`'s mixin defaults to `strict=False`, which would silently ignore missing keys, hence `strict=True`. The checkpoint is CC BY-NC 4.0 (research only) and is never redistributed. |
| DINOv2 | torch.hub `facebookresearch/dinov2` pinned to commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; weights from the URLs hard-coded there | This is the same entry point AnomalyDINO uses, with a pinned commit (AnomalyDINO itself does not pin one). |
| AUPRO | MVTec AD reference implementation (`compute_pro`, `trapezoid`) from **SplatPose**, git submodule `third_party/SplatPose` at `2c07b6dad268c6ee5c98549ba323acaaa0574916`, file `aupro.py` | See section 6. |

## 2. Model input and grid alignment

- **One preprocessing for both models.** Each image goes through VGGT's own `load_and_preprocess_images(mode="crop")`, and the same tensor feeds VGGT and DINOv2. MAD-Sim's 800×800 images become 518×518, which is 37×37 patches of 14 px. Both models therefore share the patch grid by construction. Tests check the pixel ↔ token mapping, the row-major order, and equality with VGGT's function.
- **Resolution.** DINOv2 runs at 518 px, which is VGGT's input size and DINOv2's native resolution, not AnomalyDINO's 448/672. So the baseline is "AnomalyDINO-style at 518 px", not a reproduction of AnomalyDINO.
- **Portrait images are rejected.** VGGT centre-crops them, and the pixel map would then not cover the whole image. MAD-Sim is square. PIAD will need this revisited.

## 3. Geometry

- **Frame order: query first, then references sorted by content hash.**
  - VGGT gives frame 1 its own camera/register tokens and defines the world frame by it.
  - Training sampled the other frames as pose-neighbours of frame 1, and the maintainer says frame-1 predictions are slightly better.
  - The image we score therefore gets the most reliable geometry.
  - A fixed order makes the pipeline exactly invariant to the sampler's order. VGGT is only approximately equivariant over frames 2..S.
  - `ExperimentRunner.score_query` canonicalises the order itself, and a test checks bitwise invariance.
- **Point source.** The default is depth unprojected with the predicted cameras through VGGT's own `unproject_depth_map_to_point_map`. This follows the README ("usually more accurate") and the paper's Table 3 (ETH3D: Depth+Cam 0.677 vs point head 0.709 overall error). The point-map head is the alternative switch.
  - VGGT's utility takes depth shaped `(S, H, W, 1)`; passing `(S, H, W)` raises inside numpy.
- **Precision.** fp32 weights. The aggregator runs under bf16 autocast (the official recipe; the model was trained in bf16), and the heads run in fp32, as `VGGT.forward` does. Never `.half()` the model: the fp32 heads then fail. `scripts/profile_geometry.py` measures how much fp32 vs bf16 moves the points and how many neighbour relations flip.
- **Only the needed heads are kept.** The track head is never used. The depth source drops the point head; the point-map source drops the camera and depth heads.
- **Join.**
  - A patch's point and confidence are the means over its *foreground* pixels. Averaging all pixels would pull boundary patches towards the background depth, and VGGT predicts essentially random depth with low confidence on pure white backgrounds, which were never supervised in training.
  - Patches without foreground pixels get NaN, so accidental use fails a finiteness check.
- **Cache key: the exact ordered image set.** The brief's "(class, seed, N, query id)" would reuse stale entries after a change of sampling mode. Keying on the ordered content hashes, the backend identity (model, revision, package version, point source, precision) and the foreground settings covers class, seed, N and mode. Only patch-level xyz and confidence are stored.

## 4. Foreground mask (chosen after inspecting the data)

- **Rule.** A model-input pixel is foreground if its darkest RGB channel is below 0.98, and a patch is foreground if at least 10% of its pixels are.
- **Why thresholding, not PCA.** MAD-Sim renders have an exactly white background (255). The objects are shaded below that, including light-grey and white plastic; only the anti-aliasing rim and rare specular highlights come close. Thresholding is therefore exact and label-free. AnomalyDINO's PCA masking is a heuristic for photographs, and its own code switches masking off for custom datasets.
- **Why not a flood fill from the border.** It would wrongly turn enclosed see-through gaps between parts into foreground.
- **The 10% patch rule** keeps silhouette patches, where Burrs and Missing defects sit, and drops 1–2-pixel slivers.
- **References are masked too.** The brief requires background to be excluded from both memory and scoring. Official AnomalyDINO masks only test images. This is a documented deviation and applies to both methods.
- **Consequence for Missing defects (flagged).** Their ground truth lies largely on pixels that now show background (see `DATA_NOTES.md`). Those pixels belong to background patches, which score 0 by construction. Pixel AUROC and AUPRO on Missing are therefore capped for *both* methods, and Missing is the majority defect type. The comparison stays fair, but absolute numbers are not comparable with methods that score background. Detecting missing material needs the geometric residual extension, not appearance matching.

## 5. Scoring

- **Distance.** Exact cosine distance `1 − ⟨a, b⟩` on L2-normalised features (no faiss), nearest neighbour k = 1. This matches AnomalyDINO's default (`faiss` squared L2 / 2 on normalised vectors), computed exactly.
- **No rotation augmentation of references.** AnomalyDINO's default for custom datasets adds 8 rotated copies of each reference. Rotated copies have no consistent 3D points (45° rotations do not map patches to patches), and the brief compares "same features, same scoring" without augmentation. Both methods therefore use none. Rotations by multiples of 90° could carry geometry along exactly; this is a possible later extension for both methods.
- **One distance matrix.** Both methods reduce the same `[Q, M]` matrix: `global_nn` takes the row minimum, and `anchored_nn` takes the row minimum under a neighbour mask. So r = ∞ with `min_confidence: 0` reproduces `global_nn` bit for bit (tested).
- **Radius.**
  - It is relative to the diagonal of the bounding box of the *eligible* (foreground, confident) reference points of that geometry pass.
  - **Deviation from the literal brief:** the box is taken in the point cloud's principal axes, not the world axes.
    - An axis-aligned box changes with the orientation of the frame: rotating a square by 45° changes its axis-aligned diagonal from √2 to 2, and in 3D the factor can reach √3.
    - VGGT's world frame is the query camera's frame, whose orientation relative to the object is arbitrary.
    - The axis-aligned version would therefore make r vary with the viewpoint. The principal-axes box is invariant to rotation, translation and uniform scale of the points, which a test now checks; the axis-aligned version failed that test.
  - **Kill-test value: `radius: 0.1`, derived before any run with real geometry.**
    - On MAD-Sim the object spans 11–19 patches across the 37 × 37 grid (`DATA_NOTES.md`). The reference cloud's 3D bounding-box diagonal is therefore about √3 × 15 ≈ 26 patch footprints, and 0.1 of it is about 2.5 footprints.
    - Even with perfect geometry, the nearest reference patch centre can be up to about 0.7 footprints from a query patch centre (sampling offset). The rest of the radius absorbs about 2 footprints of VGGT error.
    - Look-alikes further than about 10% of the object away are excluded.
    - The first draft's 0.05 (about 1.3 footprints) would leave almost no slack and turn many patches into "no neighbour" unseen. That would hide the effect behind the global fallback.
  - No tuning on test data. A radius sweep runs from cache in minutes; any sweep is reported in full, including r = ∞.
- **kNN alternative.** The k nearest eligible reference patches in 3D. In kNN mode, unseen patches arise only from low confidence, since there are always k neighbours.
- **Confidence (`min_confidence`).**
  - A query patch below it is unseen (`unseen_low_confidence`).
  - A reference patch below it is excluded from 3D neighbourhoods, because its location is unreliable. It stays in the `fallback_global` memory.
  - Kill-test value: 1.5 for VGGT depth confidence (≥ 1 by construction). This is the lower end of the maintainers' "1.5–3 should work well" guidance. The `demo_colmap` default of 5.0 has been reported to remove everything in sparse-view scenes.
  - It is a required key, so its value is always explicit.
- **Unseen modes.** `fallback_global` (default) uses the `global_nn` score; `zero` uses 0; `max` uses 2.0, the maximum attainable cosine distance, a data-independent constant. The unseen masks (by reason) are always returned and reported as coverage.
- **Image score.** This is AnomalyDINO's `mean_top1p`: the mean of the `k = max(1, ⌊h·w·p/100⌋)` highest scores over the *full* grid, with background at 0. For 37×37 that is k = 13. k depends only on the grid, not on the object's size in the image. Background zeros enter only if fewer than k patches are foreground. `max` is the alternative.
- **Pixel map.** Bilinear upsampling (`align_corners=False`, identical to OpenCV's `INTER_LINEAR` used by AnomalyDINO) of the patch grid to the original 800×800, then `scipy.ndimage.gaussian_filter(σ = 4)` with mode reflect and truncate 4, as AnomalyDINO.

## 6. Evaluation

- **Image and pixel AUROC.** `sklearn.metrics.roc_auc_score`. Pixel AUROC pools every pixel of every query of a class at the original resolution.
- **AUPRO.** The MVTec AD reference implementation as shipped in SplatPose's `aupro.py`, integrated to FPR 0.3 and normalised by 0.3, exactly as SplatPose's `calculate_au_pro_au_roc` does. Why this over anomalib:
  1. It is the reference definition: exact over every distinct threshold, with 8-connected regions.
  2. SplatPose is one of the baselines in PADFormer's MAD-Sim table, so this is the implementation of record in this literature.
  3. It is pure numpy/scipy and one sort per class. anomalib 2.x brings Lightning, timm, kornia and more, and its AUPRO interpolates per-region ROC curves, which is a slight variant that sorts all pixels once per region.

  SplatPose states no licence for that file. It is therefore used through a pinned submodule (not copied), and its SHA-256 is checked at import. PADFormer does not say how it computes AUPRO.
- **Labels.** Labels come from the folder (`test/good` → 0, others → 1), as the brief's "good/defective label" implies. MAD-Sim has defective images whose mask is empty (presumably the defect is hidden from that viewpoint). They stay defective for image AUROC, contribute only negative pixels to the pixel metrics, and are counted per class in `results.json` (`defective_with_empty_mask`). The MVTec/SplatPose code instead derives image labels from mask non-emptiness; the counts let a reader judge the difference.
- **Standard deviation and the mean row.** Sample standard deviation (ddof = 1) over seeds. The "Mean" row averages classes within each seed, then reports mean ± std over seeds.

## 7. Protocol

- **Sampling.** `per_query` (default, PADFormer's protocol: "for each test image, we randomly select N anomaly-free images from the training set") and `per_object` (one set per class and seed).
  - The randomness comes from `SeedSequence([seed, hash(class), hash(query content)])`, so a query's references do not depend on file or folder names, on the other queries, or on iteration order.
  - The pool is in content-hash order. A reference byte-identical to its query is an error.
- **Seeds.** 0, 1, 2. Only reference sampling is random; the models are frozen and deterministic.
- **Comparability is enforced.** `make_table.py` refuses to tabulate runs whose `data`, `features`, `foreground` or `aggregation` sections, seeds or evaluated queries differ.
- **Smoke subsets.** `max_queries_per_class` evaluates an evenly spaced subset in name order. It is used only in smoke configs; every real config has `null` (all queries).

## 8. Caching, determinism, invariances

- **Features.** Cached per image content hash and model identity (hub commit, backbone, precision, layer, storage dtype) as L2-normalised **float16**, then re-normalised in float32 when used. The values used for scoring are the cached ones on both the fresh and the cached path, so cache vs fresh is bitwise identical (tested). float16 halves the cache: about 2.8 MB per image for ViT-L. That is about 3.9 GB for the 1,405 images of the kill test and about 27 GB for all 9,789 MAD-Sim images, instead of 55 GB in float32.
- **Cache format.** Each entry stores its identity and is checked on load; writes are atomic. `CACHE_FORMAT_VERSION` is bumped whenever a cached computation changes.
- **No model loads on a warm cache.** Models are loaded lazily, so a run served entirely from cache never loads either model (tested by making the loaders raise).
- **Batch size.** `features.batch_size` is a config key. GPU/CPU kernels may differ in the last bits across batch sizes, so features are compared within one float16 ulp and metrics within 1e-6 (tested). Bitwise equality across batch sizes is not something BLAS guarantees.
- **Determinism.** `seed_everything` seeds the Python, NumPy and torch RNGs, sets cuDNN deterministic, `CUBLAS_WORKSPACE_CONFIG` and `torch.use_deterministic_algorithms(warn_only=True)`. Re-running a config reproduces every number bitwise (tested from a fresh cache).
- **Provenance.** Every `results.json` records the resolved config, git commit and dirty files, package versions (VGGT with its commit), hardware, seed, timing and peak VRAM. Tables mark results produced from a dirty tree.

## 9. Robustness and failure policy

- Exceptions propagate. No sample is ever dropped.
- A query with no foreground patch (degenerate) scores 0 everywhere and is counted (`empty_foreground_queries`).
- A defective image with an empty mask is kept and counted.
- A reference set without any foreground patch raises.
- Tensor shapes, dtypes and ranges are checked at module boundaries (`padgeo.utils.checks`).

## 10. Interpretation of the brief

- **"Every module … on a path that starts at `scripts/run.py`".** The brief also requires `make_table.py`, a visualisation script, a profiling run and a data-inspection check. So every library function is reachable from one of these entry points; `AUDIT.md` gives the caller chain of each.
- **The kNN mode** has no unseen-by-distance notion; see section 5.
- **PADFormer's numbers** (81.3 / 92.9 / 86.7) were checked against the paper (arXiv 2608.04210v1, Table 1, MAD-SIM, 4-shot). They are single runs of a model **trained** on the full MAD-Sim training set; the number of classes averaged and the AUPRO implementation are not stated. The table therefore also shows UniVAD, the training-free method in the same table (58.4 / 80.0 / 67.9), and labels both as context, not like-for-like. The kill test covers 3 classes.

## 11. Concerns about the experimental design, and cheaper tests (for discussion)

1. **Missing defects dominate the test sets, and their ground truth lies mostly on background.**
   - In the kill-test classes, Missing is the majority of defective queries; see `DATA_NOTES.md`.
   - Both methods are blind to material that is absent, so the kill test mostly measures Burrs and Stains at pixel level, and image AUROC on Missing relies on boundary cues.
   - Suggestion: report metrics per defect type. Image AUROC of good vs each type can be computed from `per_query.csv` without rerunning.
2. **`anchored_nn` can only raise scores** (section 1 of `UNDERSTANDING.md`), so any gain must come from raising anomalous patches more than normal ones.
   - **A cheap mechanism check, before reading AUROCs.** For every query patch, take the 3D distance between the patch and its `global_nn` match, normalised by the diagonal. It needs no model calls on a warm cache.
   - If anomalous patches' global matches are not much farther away in 3D than normal patches' matches, the constraint cannot help at any radius.
3. **Radius sweep.** r ∈ {0.025, 0.05, 0.1, 0.2, 0.4, ∞} runs from cache in minutes. r = ∞ must reproduce `global_nn`, a built-in sanity check. Report the full curve rather than the best point.
4. **Out-of-distribution view sets for VGGT.**
   - VGGT was trained on sets whose frames overlap frame 1. Four random views of a full view sphere often do not overlap the query much.
   - Watch the unseen fraction and VGGT's confidence. A diagnostic on training images, using `transforms.json` only for that diagnostic (brief section 7), would measure registration error directly.
5. **The query influences the reference geometry.** All frames are inferred jointly, so a defective query can perturb the references' points. A two-pass variant (references alone, then the query registered separately) would remove that coupling, at twice the cost.
6. **Three classes, one N.** The kill test is a go/no-go signal, not a result. The 20-class runs for N ∈ {2, 4, 10} remain the actual experiment.
