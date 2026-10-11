# Milestone 6: integration audit

This file audits the codebase against section 1b of the brief: nothing dangles, every key and option is wired
and tested, and correctness is demonstrated.

## 1. How the audit was done

1. **Static call-graph scan.** An AST scan covered every module-level function, class, method, dataclass
   field and constant in `padgeo/` and `scripts/` (364 definitions) and recorded every reference from library,
   script and test code. Every definition has a caller in library or script code. The one exception is
   `ExperimentConfig._unique_non_negative`, a pydantic validator that pydantic invokes (exercised by
   `test_config.py::test_invalid_experiment_values`).
2. **`vulture` (minimum confidence 0).**
   - Findings: pydantic `model_config` attributes and validators, attribute assignments that configure
     torch (`cudnn.benchmark`, `cudnn.deterministic`) and VGGT (dropping heads), and one real item.
   - The real item, `ReferenceSampler._mode`, was stored but never read and was **removed**.
3. **Config-key scan.** For every field of every config model, the scan recorded where the field is read and
   which tests and configs set it. Section 4 holds the curated result.
4. **Independent review.** Four reviewers worked with distinct lenses (geometry and numerics; fairness and
   leakage; spec compliance and integration; test adequacy). Each was followed by a skeptic that tried to refute
   every finding. The verified findings and their resolutions are in section 8.
5. **Commands.** `make check` (format, lint, types, tests) and a fresh clone set up by following only the
   README (section 9).

## 2. Entry points

| Entry point | What it runs |
|---|---|
| `scripts/run.py` | One experiment config end to end: `padgeo.pipeline.ExperimentRunner.run`. |
| `scripts/make_table.py` | Cross-seed tables from result files: `padgeo.eval.summary`. |
| `scripts/visualise.py` | Side-by-side figures from result files: `padgeo.utils.visualise`. |
| `scripts/profile_geometry.py` | Geometry profiling: `padgeo.geometry.profiling`. |
| `scripts/inspect_data.py` | Dataset facts for `DATA_NOTES.md` (the Milestone 1 check). |
| `scripts/run_killtest.sh` | Calls `profile_geometry.py`, then `run.py` twice, then `make_table.py` and `visualise.py`. |

The brief asks that every function be reachable from `scripts/run.py`. It also requires the table,
visualisation and profiling scripts, so the criterion applied is "reachable from an entry point" (see
`DECISIONS.md` section 10).

## 3. Modules: public functions and classes, and their callers

Private helpers (leading underscore) are called only inside their own module; the scan confirms each has
a caller.

| Module | Public definitions | Called from |
|---|---|---|
| `config.py` | `Config` and its section models; `ProfileConfig` | Schema of every config |
| | `load_yaml` (rejects duplicate keys) | `load_config`, `load_profile_config`, `summary.load_reference_numbers` |
| | `load_config` | `run.py`, `make_table.py`, `visualise.py`, `inspect_data.py`, `load_profile_config` |
| | `load_profile_config` | `profile_geometry.py` |
| | `config_to_yaml` | `results.write_seed_results` |
| `pipeline.py` | `ExperimentRunner` (`prepare_images`, `score_query`, `score_class`, `run_class`, `run`) | `run.py` |
| | `PreparedImage`, `ScoredClass` | Internal data carriers |
| `data/base.py` | `ImageRef`, `ClassSplit`, `GroundTruth`, `AnomalyDataset` | Dataset interface (loaders, sampler, pipeline, evaluation) |
| | `hash_file` | `MADSim` |
| `data/mad_sim.py` | `MADSim` | `data/registry.build_dataset` |
| `data/registry.py` | `build_dataset` | `pipeline`, `profiling`, `visualise.py`, `inspect_data.py` |
| `data/sampling.py` | `ReferenceSampler` | `pipeline.score_class`, `profiling.run_profile` |
| | `select_queries` | `pipeline.score_class`, `profiling.run_profile` |
| `data/images.py` | `load_model_input` | `pipeline.prepare_images`, `geometry/store.get`, `profiling`, `inspect_data.py` |
| | `model_input_height` | `load_model_input` |
| | `ModelInput`, `MODEL_INPUT_WIDTH`, `PATCH_SIZE` | Same callers |
| `features/dinov2.py` | `DINOv2Extractor`, `FeatureStore`, `unpack_features` | `pipeline` |
| `features/foreground.py` | `pixel_foreground`, `patch_foreground` | `pipeline.prepare_images`, `geometry/store.get`, `profiling`, `inspect_data.py` |
| `geometry/__init__.py` | `build_geometry_backend` | `pipeline`, `profiling` |
| `geometry/base.py` | `GeometryBackend`, `PixelGeometry` | Backend interface |
| `geometry/camera.py` | `unproject_depth` | `FakeGeometryBackend.render`, `VGGTBackend.predict` |
| `geometry/fake.py` | `FakeGeometryBackend` | `build_geometry_backend` |
| | `look_at_camera`, `raycast_unit_sphere`, `relative_extrinsic`, `viewpoint_for_image`, `SyntheticCamera` | `FakeGeometryBackend` |
| `geometry/vggt_backend.py` | `VGGTBackend` | `build_geometry_backend` |
| | `keep_needed_heads` | `VGGTBackend._load_model` |
| `geometry/patches.py` | `pool_to_patch_grid`, `PatchGeometry` | `geometry/store.get`, `profiling` |
| `geometry/store.py` | `GeometryStore` | `pipeline` |
| `geometry/profiling.py` | `run_profile`, `render_profile` | `profile_geometry.py` |
| `methods/__init__.py` | `build_method` | `pipeline` |
| `methods/base.py` | `Method`, `PatchSet`, `Patches`, `NearestNeighbourScores`, `MethodOutput`, `foreground_patches`, `build_memory`, `cosine_distances` | `Method.score`; used by both methods |
| | `global_nearest_neighbour` | `GlobalNN`; `AnchoredNN` (fallback) |
| `methods/global_nn.py` | `GlobalNN` | `build_method` |
| `methods/anchored_nn.py` | `AnchoredNN` | `build_method` |
| | `neighbour_mask` (returns the mask and the radius used), `bounding_box_diagonal` | `AnchoredNN`, `profiling` |
| | `median_patch_spacing` | `AnchoredNN` (radius in query patches) |
| `methods/aggregation.py` | `aggregate_image_score` | `Method.score` |
| | `make_pixel_map` | `Method.score`, `utils/visualise.render_query_figure` |
| `eval/metrics.py` | `image_auroc`, `pixel_auroc`, `aupro` | `eval/evaluate.evaluate_class` |
| `eval/evaluate.py` | `evaluate_class`, `ClassMetrics` | `pipeline.run_class` |
| | `load_mask` | `evaluate_class`, `visualise.py`, `inspect_data.py` |
| `eval/results.py` | `QueryRecord`, `ClassResult`, `summarise_class`, `write_class_scores`, `write_seed_results` | `pipeline` |
| | `coverage_summary`, `timing_summary`, `ClassSummary` | `summarise_class` |
| `eval/summary.py` | `load_run_set`, `load_reference_numbers`, `render_markdown` | `make_table.py` |
| | `check_comparable` | `render_markdown`, `visualise.py` |
| | `mean_std`, `metric_values` | `render_markdown` |
| `utils/cache.py` | `TensorCache` | `pipeline` (feature and geometry caches) |
| | `CacheIdentityError` | Raised in `TensorCache.load` |
| `utils/checks.py` | `check_tensor` | Module boundaries throughout |
| `utils/logging.py` | `setup_logging` | `run.py`, `pipeline.run`, `profile_geometry.py` |
| `utils/precision.py` | `torch_dtype` | `FakeGeometryBackend` |
| | `autocast` | `DINOv2Extractor`, `VGGTBackend` |
| `utils/provenance.py` | `collect_provenance` | `pipeline.run`, `profile_geometry.py` |
| | `distribution_version` (version plus git commit) | `package_versions`; `VGGTBackend.identity`, `DINOv2Extractor.identity` (cache keys) |
| `utils/resources.py` | `reset_peak_vram`, `peak_vram_mb`, `stopwatch` | `pipeline`, `profiling` |
| `utils/seeding.py` | `seed_everything` | `pipeline.run`, `profile_geometry.py` |
| `utils/visualise.py` | `load_stored_query`, `render_query_figure` | `visualise.py` (`class_figures`) |

## 4. Config keys: where each is read, and the test that exercises it

All config models forbid unknown keys. Option-specific keys exist only inside their option, so a key that
would have no effect is rejected (`test_config.py::test_unknown_or_irrelevant_keys_are_rejected`,
`test_geometry_section_is_rejected_for_global_nn`, `test_percent_is_rejected_for_max_image_score`).

### `experiment` and `data`

| Key | Read in | Exercised by |
|---|---|---|
| `experiment.name`, `experiment.output_dir` | `ExperimentRunner.run`; `write_seed_results`; run directories in `make_table.py` and `visualise.py` | `test_integration.py` (every run) |
| `experiment.seeds` | `ExperimentRunner.run` (loop, sampler seed); `profiling.run_profile` (first seed) | `test_integration.py` (seeds 0, 1 draw different references); `test_config.py::test_invalid_experiment_values` |
| `data.dataset`, `data.root` | `registry.build_dataset` → `MADSim(root)` | `test_data.py`, `test_integration.py` |
| `data.classes` | `ExperimentRunner.run`; `profiling.run_profile` (first class) | `test_integration.py` (2 classes); `test_config.py::test_killtest_configs_differ_only_in_method` |
| `data.num_references` | `ReferenceSampler` in `score_class` | `test_sampling.py`, `test_integration.py` (N = 2) |
| `data.sampling` | `ReferenceSampler` in `score_class` and `run_profile` | `test_sampling.py` (both modes); `test_integration.py::test_per_object_sampling_reuses_one_reference_set_per_class` |
| `data.max_queries_per_class` | `select_queries` in `score_class` | `test_integration.py::test_max_queries_per_class_evaluates_an_evenly_spaced_subset`; `test_sampling.py::test_select_queries` |

### `features`, `foreground` and `cache`

| Key | Read in | Exercised by |
|---|---|---|
| `features.backbone` | `DINOv2Extractor._load_model`, `.identity` | `test_features.py` (ViT-S) |
| `features.precision` | `DINOv2Extractor.extract` (autocast), `.identity` | `test_features.py::test_half_precision_features_are_close_to_full_precision` (bf16 on CPU, fp16 on CUDA) |
| `features.device` | `DINOv2Extractor` | `test_features.py` (CPU and CUDA) |
| `features.batch_size` | `ExperimentRunner.prepare_images` | `test_integration.py::test_batch_size_does_not_change_results` |
| `foreground.pixel_threshold` | `pixel_foreground` | `test_foreground.py` |
| `foreground.min_patch_fraction` | `patch_foreground` | `test_foreground.py` |
| `cache.dir` | `TensorCache` roots in `ExperimentRunner` | `test_integration.py` (fresh vs warm cache); `test_cache.py` |

### `method`

| Key | Read in | Exercised by |
|---|---|---|
| `method.name` | `build_method` dispatch; geometry store creation in `ExperimentRunner`; `"method"` in results | `test_methods.py::test_build_method_dispatches_on_config`; `test_integration.py` |
| `method.device` | `Method.device` | `test_methods.py` (CPU); development runs (CUDA) |
| `method.geometry.backend` | `build_geometry_backend` dispatch | `test_geometry.py::test_vggt_backend_builds_without_loading_weights`; `test_integration.py` (fake) |
| `method.geometry.point_source` | `FakeGeometryBackend.render`; `keep_needed_heads`, `VGGTBackend.predict`; backend identity | `test_geometry.py::test_fake_depth_and_pointmap_sources_agree`; `test_vggt_backend.py` (both branches); `test_geometry.py::test_keep_needed_heads` |
| `method.geometry.precision` | Fake: ray-casting dtype; VGGT: aggregator autocast; backend identity | `test_geometry.py::test_fake_half_precision_is_close_to_full`; `test_vggt_backend.py` (fp32, bf16); profile in `test_integration.py` (fp32, bf16) |
| `method.geometry.device` | Fake ray casting; VGGT model placement | `test_geometry.py`, `test_vggt_backend.py` |
| `method.geometry.model`, `method.geometry.revision` | `VGGT.from_pretrained(model, revision, strict=True)`; backend identity | `test_vggt_backend.py::test_depth_branch_decodes_cameras_and_unprojects_like_the_scene` |
| `method.neighbourhood.mode`, `.radius`, `.k` | `neighbour_mask` | `test_methods.py` (radius, kNN, ∞); `test_anchoring_synthetic.py` |
| `method.unseen` | `AnchoredNN._unseen_scores` | `test_methods.py::test_unseen_modes` (all three); `test_fallback_global_differs_from_zero_when_global_score_is_positive` |
| `method.min_confidence` | Eligibility in `AnchoredNN`; `profiling._point_differences` | `test_methods.py::test_confidence_filter_query_and_reference`; `test_infinite_radius_with_confidence_filter_differs_only_through_confidence` |

### `aggregation`, profile and reference numbers

| Key | Read in | Exercised by |
|---|---|---|
| `aggregation.image_score.mode`, `.percent` | `aggregate_image_score` | `test_aggregation.py` |
| `aggregation.pixel_sigma` | `make_pixel_map` (via `Method.score`); `visualise.py` | `test_aggregation.py`; `test_integration.py` |
| Profile: `name`, `output_dir`, `experiment_config`, `num_references`, `precisions`, `queries_per_setting` | `profile_geometry.py`; `load_profile_config`; `run_profile` | `test_integration.py` (profile in the main run); `test_config.py::test_repository_profile_configs_validate` |
| Reference numbers: `source`, `note`, `by_num_references` and each method's fields | `summary.load_reference_numbers`, `_reference_table` | `test_summary.py` (the 4-shot values are checked against the paper) |

## 5. Every option in section 5 of the brief, and the tests that run it

| Option | Tests |
|---|---|
| Point source: depth unprojection / point map | Fake: `test_geometry.py::test_fake_depth_and_pointmap_sources_agree`. VGGT forward path (stub network): `test_vggt_backend.py::test_depth_branch_…` and `test_pointmap_branch_…`. Head selection: `test_keep_needed_heads[depth/pointmap]`; real architecture: `test_vggt_architecture_builds_on_cpu_and_keeps_only_depth_heads`. |
| Radius / kNN | `test_methods.py::test_radius_scales_with_the_diagonal_of_eligible_reference_points_only`, `test_bounding_box_diagonal_ignores_a_few_stray_points`, `test_anchored_reports_radius_in_query_patches`, `test_anchored_radius_hand_computed`, `test_anchored_knn_hand_computed`, `test_neighbour_mask_respects_eligibility_and_k`, `test_anchored_invariant_to_similarity_transform[radius/knn]`, `test_reference_order_does_not_change_scores[radius/knn]`; `test_anchoring_synthetic.py` |
| Unseen modes: fallback_global / zero / max | `test_methods.py::test_unseen_modes[...]` (all three, hand-computed values) |
| Confidence threshold | `test_methods.py::test_confidence_filter_query_and_reference` (query patch → unseen; reference patch → excluded) |
| Image score: top 1% mean / max | `test_aggregation.py::test_top_percent_uses_full_grid_count`, `test_top_percent_minimum_one_patch_and_max`, `test_top_percent_includes_background_zeros_only_when_foreground_is_small` |
| Pixel map: bilinear + Gaussian | `test_aggregation.py::test_pixel_map_places_patch_scores_on_the_right_pixels`, `test_pixel_map_non_square_and_smoothing`, `test_pixel_map_matches_bilinear_interpolation_convention` |
| Sampling: per_query / per_object | `test_sampling.py`; `test_integration.py::test_per_object_sampling_reuses_one_reference_set_per_class` |
| Precision: fp32 / fp16 / bf16 | Geometry: fake fp16 and bf16, profile fp32 vs bf16. VGGT stub: `test_precision_sets_autocast_for_the_aggregator_and_never_for_the_heads` checks that the setting reaches autocast. Features: bf16 (CPU) and fp16 (CUDA), which must differ from fp32. |
| Caching | `test_cache.py` (including a view saving only itself); `test_store.py` (every setting changes the key, including the VGGT commit and the image order); `test_features.py::test_feature_store_fresh_equals_cached`; `test_integration.py::test_cached_run_is_identical_and_loads_no_model` |
| Metrics (image AUROC, pixel AUROC, AUPRO) | `test_metrics.py`: hand-computed values, including AUPRO with ties, region weighting and per-type AUROC |

## 6. Other brief requirements and the tests that demonstrate them

### Conventions

| Requirement | Test |
|---|---|
| Camera-from-world (not world-from-camera) | `test_geometry.py::test_unproject_depth_camera_from_world_and_opencv_axes` (the opposite convention is shown to differ) |
| OpenCV axes | The same test and `test_unproject_depth_pixel_centre_and_axis_directions` (+x right, +y down); `test_fake_camera_is_a_proper_rotation_looking_at_origin` |
| Pixel-centre convention | `test_unproject_depth_pixel_centre_and_axis_directions`: integer pixel coordinates, principal point (W/2, H/2), as VGGT's utility. Depth vs point-map agreement in the fake backend would break with a half-pixel offset. |
| Same tensor for VGGT and DINOv2 | `test_data.py::test_model_input_is_vggt_preprocessing` |
| DINOv2 token grid row-major and aligned with pixels | `test_features.py::test_token_grid_is_row_major_and_aligned_with_pixels` (asymmetric patch (7, 29)) |
| Join grid alignment, row/column order | `test_geometry.py::test_pool_to_patch_grid_row_column_order_and_alignment` |
| Pixel map back on original pixels | `test_aggregation.py::test_pixel_map_places_patch_scores_on_the_right_pixels` |
| Foreground-weighted pooling, NaN for background | `test_geometry.py::test_pool_to_patch_grid_uses_only_foreground_pixels` |
| Cross-view consistency of the fake backend | `test_geometry.py::test_fake_cross_view_consistency` |
| VGGT camera decoding + unprojection | `test_vggt_backend.py` (a non-square image catches (H, W) mix-ups) |

### Milestone 4 checks

| Requirement | Test |
|---|---|
| A look-alike defect is missed by `global_nn` and caught by `anchored_nn` | `test_anchoring_synthetic.py::test_lookalike_defect_missed_by_global_caught_by_anchored` |
| r = ∞ reproduces `global_nn` exactly | `test_methods.py::test_infinite_radius_reproduces_global_exactly` (3 random scenes, `torch.equal`); `test_anchoring_synthetic.py::test_infinite_radius_matches_global_on_the_synthetic_scene` |

### Integration, invariances and reproducibility

| Requirement | Test |
|---|---|
| End-to-end integration (data, geometry, features, both methods, 3 metrics, results, table, figures, profile) | `test_integration.py` (12 tests through the real entry-point scripts) |
| Invariance to reference order | Method level: `test_methods.py::test_reference_order_does_not_change_scores`. Pipeline: `test_integration.py::test_reference_order_does_not_change_results` (bitwise). |
| Invariance to batch size | `test_integration.py::test_batch_size_does_not_change_results`; `test_features.py::test_batch_size_does_not_change_features` |
| Cache vs fresh | `test_integration.py::test_cached_run_is_identical_and_loads_no_model` (bitwise, and no model is loaded) |
| Same config, seed and commit give the same numbers | `test_integration.py::test_rerun_from_a_fresh_cache_reproduces_every_number` (bitwise) |
| Provenance in results | `test_integration.py::test_result_files_are_complete`; `test_utils.py::test_provenance_records_commit_versions_and_hardware` |

### Leak guards and failure policy

| Requirement | Test |
|---|---|
| Scores blind to folder names, file names and ground truth | `test_integration.py::test_scores_are_blind_to_folder_names_file_names_and_ground_truth` (rename every test folder, `good` included, and every file; delete `ground_truth/`; score from a fresh cache → bit-identical scores) |
| Query first, then references by content hash, in the geometry pass | `test_integration.py::test_geometry_pass_gets_the_query_first_then_references_by_content_hash` |
| Coverage numbers exact; confidence filter active end to end | `test_results.py`; `test_integration.py::test_anchoring_is_exercised_and_the_confidence_filter_triggers` |
| `transforms.json` never read | `test_data.py::test_transforms_json_is_never_opened`. The whole integration run executes under a guard that fails any open of `transforms.json`. |
| A query never in its reference pool | `test_data.py::test_query_identical_to_training_image_is_rejected`; `test_sampling.py::test_pool_too_small_and_identical_query_are_errors` |
| Same references for both methods | `test_integration.py::test_both_methods_saw_identical_references` |
| Degenerate samples counted, not dropped | `test_integration.py::test_degenerate_query_is_counted_not_dropped`. Empty defect masks: `test_metrics.py`, and the fixture's defective query with an empty mask is counted in `test_result_files_are_complete`. |
| Fail-loud boundaries | `test_utils.py::test_check_tensor_fails_loudly`; missing class or mask in `test_data.py`; references without foreground and anchoring without geometry in `test_methods.py` |
| Tables only compare runs that differ in the method, and only results of the current config | `test_summary.py::test_runs_differing_outside_the_method_are_rejected`, `test_runs_with_different_seeds_or_queries_are_rejected`, `test_load_run_set_reads_exactly_the_configured_seeds` |
| Duplicate YAML keys are errors | `test_config.py::test_duplicate_keys_are_rejected` |

## 7. Verified only with the fake backend or a stub, still to check with real VGGT on the 24 GB machine

1. **A real VGGT-1B forward pass through `VGGTBackend`.**
   - Locally, the forward path is tested with a stub network that returns the fake scene through VGGT's own camera encoding. That stub test covers camera decoding, output indexing, unprojection and both point sources.
   - Still unchecked: the real network's outputs, memory, and bf16 autocast on CUDA.
   - Check: run `PADGEO_RUN_VGGT=1 pytest -m vggt_weights`, and the profile step of the kill test.
2. **Whether VGGT's geometry is good enough on MAD-Sim for anchoring.** That is, whether query patches land near the matching reference patches. The coverage table of the kill test reports the unseen fraction under `radius: 0.1` and `min_confidence: 1.5`. If most patches are unseen, `anchored_nn` degenerates to `global_nn` and the kill test says little.
3. **VGGT's depth-confidence distribution on MAD-Sim renders.** It decides how many patches the 1.5 threshold drops (the "Low confidence" column of the coverage table).
4. **fp32 vs bf16 point shifts and neighbour flips**, and **peak VRAM and seconds per query for N = 2, 4, 10.** The profile step measures all of these.
5. **DINOv2 ViT-L.** Only ViT-S was run locally. ViT-L uses the same code path, but a different hub entry and more memory.
6. **Full-scale run time and memory.** Locally, one class (237 queries) took about 60 s for features (ViT-S), about 0.07 s per query for scoring, and about 90 s for evaluation at 800 × 800.

## 8. Independent review: findings and resolutions

Four reviewers produced 33 findings: geometry 4, fairness 7, spec 12, tests 10. None was rated high
severity, and many overlap. Skeptics independently re-checked the geometry and fairness findings:

- 8 confirmed;
- 3 partially correct;
- none refuted.

The spec and test skeptics hit the API usage limit, so I verified those 22 findings against the code while
fixing them. All 33 were acted on except the second half of F6, the noise from the finite test set, which is
stated as a limitation below.

### Verified by skeptics: geometry (G) and fairness (F) lenses

| Id | Finding | Verdict | Resolution |
|---|---|---|---|
| G1, F3 | Cache keys miss the VGGT code commit (the package always reports 0.0.1); features miss it too, although VGGT's preprocessing makes their input | confirmed | Identities use `distribution_version` (version plus the git commit from pip). `CACHE_FORMAT_VERSION` = 2. `test_store.py` |
| G2, F7 | The radius normaliser is a non-robust max–min extent, and the diagonal said to be logged is not | confirmed | Extent trimmed by 1% per principal axis. `radius_in_query_patches` and `eligible_reference_fraction` are logged per query and summarised in the coverage table. Tests in `test_methods.py` and `test_results.py` |
| G3 | The profile normalises shifts by a different diagonal (all points, not the eligible ones) than the method | confirmed | The profile now uses the eligible points and the radius returned by `neighbour_mask` |
| G4 | "No neighbour" also counts patches whose nearby references were removed by the confidence filter | partially correct | Docstrings state the exact meaning. `eligible_reference_fraction` is reported, so the filter's share is visible. The zero-diagonal case is documented |
| F1 | `make_table` could mix seeds from different configs or stale seed directories | confirmed | `load_run_set(config)` reads exactly the configured seeds and refuses any seed whose stored config differs. Test added |
| F2 | With `min_confidence` 1.5, r = ∞ does not reproduce `global_nn`, so the planned sanity check was misstated | partially correct | `DECISIONS.md` and `UNDERSTANDING.md` corrected: the check needs `min_confidence: 0`, and the tests use exactly that |
| F4 | Each feature cache entry serialised its whole batch (8–16× disk) | confirmed | `TensorCache.save` stores a copy of each tensor. Regression test in `test_cache.py` |
| F5 | Tables did not show the method parameters | confirmed | Run legend with each run's method settings |
| F6 | Unpaired mean ± std over seeds is a weak uncertainty for the comparison | partially correct | Paired per-seed difference table added. Uncertainty from the finite test set (for example a bootstrap over queries) is **not** added; this is a stated limitation |

### Verified by me while fixing: spec (S) and test (T) lenses

| Id | Finding | Resolution |
|---|---|---|
| S1 | Tables can mix seeds or stale directories | Same as F1 |
| S2 | Effective radius not logged | Same as G2 |
| S3 | `vggt_package` in the key is a constant | Same as G1 |
| S4 | Duplicate YAML keys silently accepted | Strict loader (`load_yaml`); `test_duplicate_keys_are_rejected` |
| S5 | `UNDERSTANDING.md` described the image score over foreground patches | Doc corrected: top 1% of the full grid, as implemented |
| S6 | Profile report lacked the resolved experiment config and seed, and used a different diagonal | Both added; diagonal fixed (G3) |
| S7 | False "determinism may not apply" warning: provenance initialised CUDA before seeding | Provenance is now collected after the first `seed_everything` |
| S8 | `requirements.txt` cites a lock file that did not exist | `requirements-lock.txt` generated (`pip freeze` of the reference environment) |
| S9 | `DATA_NOTES.md` claimed the loader rejects duplicates within a split | Doc corrected: `inspect_data.py` checks this; the loader rejects test–train duplicates |
| S10 | `inspect_data.py --json outputs/...` fails when `outputs/` does not exist | Parent directory is created |
| S11 | Table columns labelled by method only, so a sweep is illegible | Columns labelled by run name |
| S12 | `run_killtest.sh` hard-codes the figure classes | `visualise.py` defaults to every class of the config |
| T1 | Cache-key guard not tested for backend, precision, point source, foreground or order | `test_store.py` |
| T2 | Leak test used a warm cache, so nothing was recomputed from the renamed files | Renamed copy is scored from a fresh cache |
| T3 | Table tests could not catch column mix-ups | Exact row assertions (paired-difference row, coverage row, run-labelled columns) |
| T4 | Fixture exercised 3D matching for only about 4% of patches and never the confidence filter | Fake viewpoints drawn from a cone, so views overlap. New test asserts less than 90% unseen and that a strict threshold triggers the filter |
| T5 | Precision tests would pass if precision were ignored | Half-precision features must differ from fp32; the VGGT stub records the autocast state of the aggregator (on) and the heads (off) |
| T6 | No test that the radius uses only eligible points | `test_radius_scales_with_the_diagonal_of_eligible_reference_points_only` |
| T7 | Leak test left `test/good` and file names unchanged | Every test folder and file renamed |
| T8 | No test of the image order in the geometry pass | `test_geometry_pass_gets_the_query_first_then_references_by_content_hash` |
| T9 | Coverage only checked to lie in [0, 1] | `test_results.py` with hand-computed fractions and medians |
| T10 | A defective query with an empty mask was never exercised | Added to the fixture; its count is checked in the integration test |

## 9. Lint, tests and fresh-clone check

`make check` on the development machine (RTX 4050 6 GB, Python 3.11, environment as in
`requirements-lock.txt`), at the commit that adds this section:

```
ruff format --check .   ->  58 files already formatted
ruff check .            ->  All checks passed!
mypy                    ->  Success: no issues found in 64 source files
pytest                  ->  163 passed, 1 skipped in 206 s
```

The skipped test is the opt-in real-VGGT forward pass (`PADGEO_RUN_VGGT=1 pytest -m vggt_weights`), meant
for the experiment machine.

The remaining warnings come from third-party code:

- xFormers is not installed for DINOv2 (it falls back to standard attention);
- SplatPose's `aupro.py` imports `scipy.ndimage.measurements`, which is deprecated but works with the pinned SciPy;
- matplotlib uses deprecated pyparsing names.

**Fresh-clone check: not run, at the user's request.** It would have needed a second full environment
(about 2.5 GB of downloads). The first setup on the experiment machine, following the README, serves as that
check.
