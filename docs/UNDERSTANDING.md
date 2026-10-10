# Milestone 0: what this project is testing and how

Drafted before any code. It was later updated in four places where inspecting the data or VGGT's code
settled an open choice; each is marked *(updated)*:

- the frame order;
- the foreground mask;
- the radius normalisation;
- the data facts.

Facts about the files on disk are in `DATA_NOTES.md`. Decisions made while building are in `DECISIONS.md`.

## 1. Task, hypothesis, and why `global_nn` vs `anchored_nn` is the test

**Task (pose-agnostic anomaly detection, PAD).** For one object class we get N defect-free reference photos (`train/good`, N ∈ {2, 4, 10}) taken from unknown, arbitrary viewpoints. Given a query photo from another unknown viewpoint, we must:

1. output an image-level anomaly score (is the object defective?), and
2. output a pixel-level anomaly map (where is the defect?).

No poses are given at test time, and no parameters are trained.

**Hypothesis.** A query patch should be compared only with reference patches that image the *same place on the object's surface*. AnomalyDINO-style matching, the standard training-free approach, compares it with *every* reference patch. Under unknown pose, the "same place" is unknown in 2D. A frozen feed-forward geometry model (VGGT) can recover it: run VGGT once on the N references plus the query, and every pixel of every image gets a 3D point in one shared frame. Pooling those points to the DINOv2 patch grid gives every patch a 3D location at no training cost.

**Why the comparison tests exactly this.** The two methods share the sampled references, the DINOv2 features, the foreground mask, the distance, the image-level aggregation, the pixel-map construction and the metrics code. They differ in one thing: the set of reference patches searched for each query patch's nearest neighbour.

- `global_nn`: all foreground reference patches.
- `anchored_nn`: the foreground reference patches whose 3D point is near the query patch's 3D point (within radius r, or the k nearest in 3D).

So any metric difference comes from the 3D constraint. The constraint's own components (VGGT's accuracy, the radius, unseen-surface handling, confidence filtering) are part of the treatment.

**What the constraint can and cannot do, mechanically.** For a patch whose surface is covered by the references, the anchored candidate set is a subset of the global one. A minimum over a subset can only be larger:

`score_anchored(q) >= score_global(q)` for every seen patch. The two are equal for unseen patches under the default `fallback_global` mode.

So `anchored_nn` never lowers a patch score. It can only win by raising anomalous patches' scores more than normal patches' scores:

- **Gain (true positives).** A defect that looks like normal material from *elsewhere* on the object, for example a stain in one of the toy's own colours, a colour swap, or a burr that resembles a stud, finds a look-alike under `global_nn` and scores low. Under `anchored_nn` it is compared only with the same surface location, which looks different, so it scores high.
- **Cost (false positives).** A normal patch is penalised when its true counterpart in the references falls outside the radius. That happens through VGGT's 3D error or misregistration of the query. It also happens when the same surface looks different from the reference viewpoints (foreshortening, lighting, occlusion boundaries) while some other location happens to look closer.

The experiment measures the net effect. A null or negative result is a finding, not a bug.

## 2. Data flow for one query

Notation:

- `S = N + 1` is the number of images in one VGGT forward pass.
- `H0 × W0` is the original image size (MAD-Sim: 800 × 800 RGB on pure white; see `DATA_NOTES.md`).
- `H = W = 518` is the model input size; for square images, VGGT's preprocessing resizes to exactly 518 × 518.
- `h = w = 37` is the patch grid (518 / 14).
- `D` is the DINOv2 width (384 for ViT-S/14, 1024 for ViT-L/14).
- Pixel `(v, u)` means row `v`, column `u`. Patch `(i, j)` means patch row `i`, patch column `j`.

| Step | Quantity | Shape / dtype | Frame / convention |
|---|---|---|---|
| 0 | Image files: query from `test/*`, references from `train/good` | PNG, `H0 × W0` | Pixel grid of the original image. Ground truth is *not* loaded. |
| 1 | Reference set chosen by the seeded sampler from (seed, class, query identity) | N image refs | Order is canonicalised by a stable key before use, so the sampler's order cannot matter. |
| 1b | Model input: VGGT's own `load_and_preprocess_images` (alpha composited onto white, bicubic resize to width 518, height rounded to a multiple of 14, centre-crop only if taller than 518) | `[3, 518, 518]` float32 in [0, 1] | One tensor per image, fed to *both* models, so the two grids are aligned by construction. |
| 2 | VGGT input, query first, then the references in canonical (content-hash) order *(updated)* | `[1, S, 3, 518, 518]` | Frame 0 (the query) defines VGGT's world frame. Frame 0 gets its own tokens, VGGT's training sampled the other views around it, and its predictions are slightly better, so the image we score goes first. |
| 2a | Camera encoding → extrinsics, intrinsics | `[S, 3, 4]`, `[S, 3, 3]` | Extrinsic = **camera-from-world** `[R \| t]`. OpenCV axes: +x right, +y down, +z forward. Principal point `(W/2, H/2)`. |
| 2b | Depth and depth confidence | `[S, 518, 518]`, `[S, 518, 518]` | Depth along camera +z. Confidence ≥ 1 (VGGT's `1 + exp` activation); higher is more reliable. |
| 2c | Per-pixel 3D points: depth unprojected with VGGT's own utility (default), or the point-map head | `[S, 518, 518, 3]` float32 | Unprojection: `X_cam = d · K⁻¹ [u, v, 1]ᵀ` with integer pixel indices `u` (column) and `v` (row), as in VGGT's code. Then `X_world = Rᵀ (X_cam − t)`. World = camera frame of frame 0, arbitrary scale. |
| 3 | DINOv2 last-layer normalised patch tokens, reshaped row-major (token `k = i · 37 + j`), L2-normalised | `[37, 37, D]` per image | Patch `(i, j)` covers input pixels rows `14i … 14i+13`, cols `14j … 14j+13`. Cached per image. |
| 4 | Join: 3D points and confidence pooled to the patch grid (mean over the patch's foreground pixels) | xyz `[S, 37, 37, 3]`, conf `[S, 37, 37]` | Same world frame as 2c. Same patch ↔ pixel mapping as 3. Cached per image set. |
| 5 | Foreground mask from image content only *(updated)*: a pixel is foreground if its darkest channel is below 0.98, a patch if at least 10% of its pixels are | pixel `[518, 518]` bool → patch `[37, 37]` bool | Same mask for both methods. The background is exactly white in every MAD-Sim image. |
| 6 | Memory: foreground patches of the N references | features `[M, D]`, xyz `[M, 3]`, conf `[M]` | `M ≈ N × (foreground patches per image)`. |
| 7 | Query foreground patches | features `[Q, D]`, xyz `[Q, 3]`, conf `[Q]` | |
| 7a | Cosine distance matrix, shared by both methods | `[Q, M]` float32 | `1 − ⟨f_q, f_m⟩`, in [0, 2]. |
| 7b | `global_nn` | `[Q]` | Row-wise minimum over all of `M`. |
| 7c | `anchored_nn` | `[Q]` scores + `[Q]` unseen flags | Row-wise minimum over the neighbour mask `[Q, M]`. The mask is `‖x_q − x_m‖ ≤ r · diag` (or the k nearest in 3D) and `conf_m ≥ τ`, where diag is the diagonal of the reference cloud's bounding box in its principal axes *(updated)*. A query patch is *unseen* if it has no neighbour or `conf_q < τ`. Unseen patches get the global score, 0, or the maximum score, depending on the unseen mode. |
| 7d | Scatter back to the grid; background patches get 0 | scores `[37, 37]`, unseen `[37, 37]` | |
| 8 | Image score (mean of top 1 % of foreground patch scores, or max) | scalar | Identical code for both methods. |
| 8b | Pixel map: bilinear upsample `[37, 37] → [H0, W0]` (`align_corners=False`), then Gaussian smoothing with σ in original pixels | `[H0, W0]` float32 | The 37 × 37 grid tiles the whole 518 × 518 input. For square images that input is the whole original image, so patch `(i, j)` lands on original rows `[i · H0/37, (i+1) · H0/37)`. |
| 9 | Stored per query: image score, patch scores, unseen mask, timing, peak VRAM | | |
| 10 | Labels and masks joined only here: image AUROC (scores vs good/defective), pixel AUROC and AUPRO (maps vs ground-truth masks) | per class | |
| 11–12 | Per-seed JSON/CSV, then mean ± std over seeds next to the reference numbers | | |

## 3. Assumptions the method depends on, and how each could fail

1. **One rigid scene.** VGGT must treat the N references and the query as one static scene, even though the query may be a *different, defective* object.
   - In MAD-Sim the defective objects are modified renders of the same model. In MAD-Real they are different physical copies.
   - *Failure:* the query is misregistered, so its points land on the wrong part of the reference surface. `anchored_nn` then compares it with the wrong patches. That inflates scores on normal queries (false positives) or suppresses them on defective ones.
   - A defect could also distort the *reference* geometry, because all S frames are inferred jointly.
2. **Coverage.** The N references must actually observe the surface the query sees. With 2–4 random views of a full view sphere, parts of the query's visible surface may never appear in any reference.
   - Those patches are "unseen" and fall back to the global score. So the achievable effect is bounded by coverage.
   - We report the unseen fraction (split into "no neighbour" and "low confidence") for every run.
3. **3D accuracy at patch scale.** VGGT's error on a query patch must be smaller than r, and r must be small enough to exclude look-alike locations.
   - A patch spans roughly 1/20–1/30 of the object, so r of a few percent of the bounding-box diagonal is the interesting range.
   - *Failure:* thin or textureless parts, white backgrounds, and very few views inflate the error, which forces r up towards `global_nn`.
4. **Frame and scale.** VGGT's world frame is frame 0's camera and its scale is arbitrary per run.
   - r is therefore relative to the reference cloud's bounding-box diagonal, taken in the cloud's principal axes *(updated)*. An axis-aligned box would change with the arbitrary orientation of the query camera's frame, by up to √3. The method is then invariant to similarity transforms of the points, and a test checks this.
   - *Failure:* outlier points inflate the bounding box and silently enlarge r. Confidence filtering and foreground pooling reduce this. We log the diagonal.
5. **VGGT is not permutation-invariant.** Frame 0 gets different camera/register tokens and defines the world frame, so the input order changes the output.
   - We fix the order: query first, then references sorted by content hash *(updated)*. The pipeline is then invariant to the sampler's order by construction. VGGT itself is not.
6. **DINOv2 features are roughly viewpoint-stable.** The same surface location must look more similar across viewpoints than other locations do.
   - *Failure:* foreshortening near silhouettes, view-dependent shading, and specular highlights on rendered plastic make normal patches look anomalous under `anchored_nn`, which has fewer candidates to "hide" in.
7. **The confidence scale means something on synthetic renders.** VGGT's confidence comes from an aleatoric-uncertainty loss and is unbounded above 1.
   - The threshold is set in that scale, a priori.
   - *Failure:* confidence could be uninformative on clean renders, or low everywhere on textureless studs.
8. **A patch has one 3D location.** Boundary patches mix object and background pixels. Averaging all their pixels would place them between the object and the background depth.
   - We pool only foreground pixels.
   - *Failure:* patches straddling two depth layers of the object, such as a leg in front of the body, still average two surfaces.
9. **The foreground mask removes only background.**
   - *Failure:* for **Missing** defects, the ground-truth mask can cover pixels that now show background, where the part used to be. Those pixels are excluded from scoring by *both* methods, which caps pixel AUROC and AUPRO for that defect type. This is the same for both methods but matters when comparing with published numbers. It is what the "geometric residual" extension (section 7 of the brief) would address.
   - *(updated)* Measured: 14–72% of a Missing mask lies on background (mean per class), and Missing is 55% of all defective queries. A development run gave image AUROC 0.445 for good vs Missing, against 0.650 for Burrs and 0.742 for Stains. See `DATA_NOTES.md`.
10. **Patch-grid alignment.**
    - VGGT and DINOv2 see the identical 518 × 518 tensor.
    - VGGT's dense heads output one value per input pixel.
    - DINOv2 tokens are row-major over the 37 × 37 grid.

    Each of these three is tested explicitly, not assumed.

## 4. Ways the comparison could be unfair or leak labels, and the guard for each

| Risk | Guard |
|---|---|
| The methods see different references | One sampler, keyed by (seed, class, query content hash), is called by the shared pipeline. Reference ids are written per query, and a test asserts they are identical across the two methods. |
| Different preprocessing, masking, aggregation or evaluation | One code path. A method implements only the nearest-neighbour rule (`patch scores ← distance matrix, memory, query`). Everything else is a shared function called once. |
| Ground-truth masks or defect-type names reach scoring | The scoring path receives only tensors. Per-query randomness and cache keys use image *content hashes*, never paths, so folder names cannot affect any score. Integration test: rename the defect folders and delete `ground_truth/`, then confirm every query's score is bit-identical. Ground truth is read only by the evaluation step. |
| A query appears in the reference pool | References come only from `train/good`, queries only from `test/`. The loader asserts that the two sets of content hashes are disjoint (byte-identical files would be caught). |
| `transforms.json` used by the method | The loader never opens it. The test fixture contains an invalid `transforms.json`, so any read would fail the tests. |
| Hyperparameters tuned on test labels | All scoring hyperparameters (r, k, τ, unseen mode, aggregation, σ, mask parameters) are fixed in config before any real run. They are the same for all classes and chosen from first principles or upstream defaults, as recorded in `DECISIONS.md`. Any sweep is reported in full. |
| Cherry-picked seeds | Seeds are fixed in config (0, 1, 2). Every seed is reported, as mean ± std. |
| Stale cache entries mixing configurations | Cache keys hash the model identity, resolution, precision, point source, mask config and the exact ordered image set (content hashes). A test checks that changing any of them changes the key. |
| A skipped sample silently changes the metric | Nothing is skipped silently. Exceptions propagate. Degenerate cases (for example an empty foreground) are scored by a defined rule, counted, and reported in the results file. |
| Surprisingly good numbers caused by a bug | Treated as a leak first. The checks are the leak test above, a pose-free sanity check that r = ∞ reproduces `global_nn` exactly, and coverage statistics. |

## 5. Open questions, and places where the brief is ambiguous or (I think) wrong

None of these block work. Each is resolved provisionally in `DECISIONS.md`.

1. **"The result must not depend on the order of the reference images."** True for the scoring step (a minimum is order-invariant). Not true for VGGT, which treats frame 0 specially. We guarantee invariance by canonicalising the order before VGGT, and we say so rather than claim VGGT is invariant.
2. **"…or on batch size."** Floating-point kernels can differ in the last bits across batch sizes on GPU. Batch size is a config key for feature extraction. The test checks that features agree within a stated tolerance and that metrics agree. Bitwise equality across batch sizes is not something cuBLAS guarantees.
3. **"`anchored_nn` with r = ∞ reproduces `global_nn` exactly."** This holds only with confidence filtering off. A low-confidence *reference* patch is excluded from the anchored candidate set, because its location is unreliable, but it stays in the global memory. The exactness test therefore runs with τ disabled. A second test checks equality when all patches pass τ.
4. **`max` unseen mode.** "Treat as anomalous" needs a value. We use the maximum attainable cosine distance (2.0), a fixed, data-independent constant. A per-query maximum would make the value depend on the other patches.
5. **Confidence threshold scale.** VGGT's depth and point heads have different, unbounded confidence scales (≥ 1). We use an absolute threshold in the scale of the selected head. The default comes from VGGT's own code, not from our test data.
6. **"Every module … on a path that starts at `scripts/run.py`."** The brief also requires `make_table.py`, a visualisation script and a profiling run. We read this as "every library function is reachable from an entry-point script in `scripts/`". `AUDIT.md` lists the entry point for each function.
7. **Cache key "(class, seed, N, query id)".** That key goes stale if the sampling mode changes, because the same tuple then maps to different references. We key on the exact ordered image set instead, which subsumes class, seed, N and sampling mode.
8. **Kill-test classes versus reference numbers.** The PADFormer figures are 20-class means. The kill test uses 3 classes, so the table labels them as not directly comparable. Whether those exact figures appear in the paper for the 4-reference setting is being checked against the paper itself.
9. **Resolution.** AnomalyDINO runs DINOv2 at 448 or 672 px. Here both methods use 518 px so that the DINOv2 grid aligns with VGGT's input. The baseline is "AnomalyDINO-style at 518 px", not a reproduction of AnomalyDINO's numbers.
10. **DINOv2 ViT-L without registers has high-norm artefact tokens**, mostly in low-information background regions. Background patches are masked out, which limits the effect. The `_reg` backbones can be selected via the existing backbone config key, so no extra code is needed.
11. **"Steps 2 and 3 are the only ones that need the GPU models."** Correct. Scoring runs from cache without loading either model, and a test checks this.
12. **A cheaper first test of the mechanism.** For each query patch, measure the 3D distance between it and its `global_nn` match, using the cached geometry and no extra model calls. If anomalous patches' global matches are not disproportionately far away in 3D compared with normal patches' matches, the constraint cannot help, whatever r is. A radius sweep from cache (r ∈ {…, ∞}) is similarly cheap, and r = ∞ is a built-in sanity check. Both are proposed in `DECISIONS.md`.
13. **The brief's "bounding-box diagonal" is frame-dependent** *(updated)*. An axis-aligned box changes with the orientation of VGGT's world frame, so the box is taken in the cloud's principal axes; see `DECISIONS.md`.
