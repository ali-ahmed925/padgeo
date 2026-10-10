"""One experiment end to end, with the same code for both methods.

Only the method differs, plus the geometry pass for anchored_nn.

For each seed and class:

1. Sample N references for every query with the shared, seeded sampler (the pipeline's only randomness).
2. (Steps 3 and 5 of the brief) For every image used, read DINOv2 features from cache or compute them,
   and compute the foreground mask from the image.
3. For each query:
   a. anchored_nn only (steps 2 and 4): patch geometry of [query, references...], from cache or from the
      geometry backend;
   b. (steps 6-8) `method.score` gives patch scores, unseen masks, the image score and the pixel map;
   c. (step 9) record timing and peak VRAM.
4. (Step 10) Evaluate against ground truth. This is the first and only place where labels or masks are
   read.

Results of each seed are then written to disk (step 11, `padgeo.eval.results`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from padgeo.config import AnchoredNNConfig, Config
from padgeo.data.base import ImageRef
from padgeo.data.images import PATCH_SIZE, load_model_input
from padgeo.data.registry import build_dataset
from padgeo.data.sampling import ReferenceSampler, select_queries
from padgeo.eval.evaluate import ClassMetrics, evaluate_class
from padgeo.eval.results import (
    ClassResult,
    QueryRecord,
    summarise_class,
    write_class_scores,
    write_seed_results,
)
from padgeo.features.dinov2 import DINOv2Extractor, FeatureStore, unpack_features
from padgeo.features.foreground import patch_foreground, pixel_foreground
from padgeo.geometry import build_geometry_backend
from padgeo.geometry.patches import PatchGeometry
from padgeo.geometry.store import GeometryStore
from padgeo.methods import build_method
from padgeo.methods.base import PatchSet
from padgeo.utils.cache import TensorCache
from padgeo.utils.logging import setup_logging
from padgeo.utils.provenance import collect_provenance
from padgeo.utils.resources import peak_vram_mb, reset_peak_vram, stopwatch
from padgeo.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedImage:
    """Per-image quantities shared by every query that uses the image.

    Attributes:
        features: [h, w, D] float16 features exactly as stored in the cache.
        foreground: [h, w] bool patch foreground mask.
        original_size: (H0, W0) of the image file.
    """

    features: torch.Tensor
    foreground: torch.Tensor
    original_size: tuple[int, int]


@dataclass(frozen=True)
class ScoredClass:
    """Scores of every query of one class and seed, before any ground truth is read.

    Attributes:
        records: One record per query, in query order.
        num_images_prepared: Distinct images whose features and masks were prepared.
        feature_seconds: Time to prepare them.
        feature_peak_vram_mb: Peak CUDA memory while preparing them, or None without CUDA.
    """

    records: tuple[QueryRecord, ...]
    num_images_prepared: int
    feature_seconds: float
    feature_peak_vram_mb: float | None


class ExperimentRunner:
    """Runs one config: all seeds, all classes, all queries.

    Args:
        config: The validated experiment config.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset = build_dataset(config.data)
        self.method = build_method(config.method, config.aggregation)
        self.feature_store = FeatureStore(
            DINOv2Extractor(config.features), TensorCache(config.cache.dir, "features")
        )
        self.geometry_store: GeometryStore | None = None
        if isinstance(config.method, AnchoredNNConfig):
            self.geometry_store = GeometryStore(
                build_geometry_backend(config.method.geometry),
                TensorCache(config.cache.dir, "geometry"),
                config.foreground,
            )

    def prepare_images(self, images: Sequence[ImageRef]) -> dict[str, PreparedImage]:
        """Features and foreground masks of `images`, keyed by content hash.

        Features are read from cache where possible; misses are computed in batches of
        `features.batch_size`, in content-hash order.
        """
        foregrounds: dict[str, torch.Tensor] = {}
        sizes: dict[str, tuple[int, int]] = {}
        features: dict[str, torch.Tensor] = {}
        pending: list[tuple[ImageRef, torch.Tensor]] = []

        def flush() -> None:
            hashes = [ref.content_hash for ref, _ in pending]
            stored = self.feature_store.compute_and_store(hashes, torch.stack([px for _, px in pending]))
            features.update(zip(hashes, stored, strict=True))
            pending.clear()

        for ref in sorted({ref.content_hash: ref for ref in images}.values(), key=lambda r: r.content_hash):
            model_input = load_model_input(ref.path)
            input_hw = (model_input.pixels.shape[1], model_input.pixels.shape[2])
            foregrounds[ref.content_hash] = patch_foreground(
                pixel_foreground(model_input.pixels, self.config.foreground),
                PATCH_SIZE,
                self.config.foreground,
            )
            sizes[ref.content_hash] = model_input.original_size
            cached = self.feature_store.lookup(ref.content_hash, input_hw)
            if cached is not None:
                features[ref.content_hash] = cached
                continue
            pending.append((ref, model_input.pixels))
            if len(pending) == self.config.features.batch_size:
                flush()
        if pending:
            flush()
        return {
            key: PreparedImage(features=features[key], foreground=foregrounds[key], original_size=sizes[key])
            for key in foregrounds
        }

    @staticmethod
    def _patch_set(image: PreparedImage, geometry: PatchGeometry | None, index: int) -> PatchSet:
        """The PatchSet of image `index` of a query's image set (geometry only for anchored_nn)."""
        features = unpack_features(image.features)
        if geometry is None:
            return PatchSet(features=features, foreground=image.foreground)
        return PatchSet(
            features=features,
            foreground=image.foreground,
            xyz=geometry.xyz[index],
            confidence=geometry.confidence[index],
        )

    def score_query(
        self, query: ImageRef, references: tuple[ImageRef, ...], prepared: dict[str, PreparedImage]
    ) -> QueryRecord:
        """Steps 2, 4 and 6-9 for one query (the order of `references` does not matter)."""
        reset_peak_vram()
        # Query first: it defines VGGT's world frame (frame 1, slightly more accurate, and the frame VGGT's
        # training sampled the other views around). References follow in canonical content-hash order, so
        # the geometry pass, and hence every score, is independent of the order they arrive in.
        references = tuple(sorted(references, key=lambda ref: ref.content_hash))
        images = [query, *references]
        geometry: PatchGeometry | None = None
        cache_hit: bool | None = None
        with stopwatch() as geometry_seconds:
            if self.geometry_store is not None:
                geometry, cache_hit = self.geometry_store.get(images)
        with stopwatch() as scoring_seconds:
            sets = [self._patch_set(prepared[ref.content_hash], geometry, i) for i, ref in enumerate(images)]
            output = self.method.score(sets[0], sets[1:], prepared[query.content_hash].original_size)
        return QueryRecord(
            query=query,
            references=references,
            foreground=prepared[query.content_hash].foreground,
            output=output,
            geometry_seconds=geometry_seconds[0],
            scoring_seconds=scoring_seconds[0],
            peak_vram_mb=peak_vram_mb(),
            geometry_cache_hit=cache_hit,
        )

    def score_class(self, class_name: str, seed: int) -> ScoredClass:
        """Steps 1-9 for every query of one class and seed. Reads no labels and no ground truth."""
        split = self.dataset.load_split(class_name)
        queries = select_queries(split.queries, self.config.data.max_queries_per_class)
        sampler = ReferenceSampler(split, self.config.data.num_references, self.config.data.sampling, seed)
        references = {query.name: sampler.sample(query) for query in queries}
        used = [*queries, *(ref for refs in references.values() for ref in refs)]
        reset_peak_vram()
        with stopwatch() as feature_seconds:
            prepared = self.prepare_images(used)
        feature_peak_vram = peak_vram_mb()
        logger.info(
            "%s: %d queries, %d images prepared in %.1fs",
            class_name,
            len(queries),
            len(prepared),
            feature_seconds[0],
        )
        records = []
        for index, query in enumerate(queries):
            records.append(self.score_query(query, references[query.name], prepared))
            if (index + 1) % 25 == 0 or index + 1 == len(queries):
                logger.info("%s: scored %d/%d queries", class_name, index + 1, len(queries))
        return ScoredClass(
            records=tuple(records),
            num_images_prepared=len(prepared),
            feature_seconds=feature_seconds[0],
            feature_peak_vram_mb=feature_peak_vram,
        )

    def run_class(self, class_name: str, seed: int) -> ClassResult:
        """Score every query of one class (`score_class`), then evaluate against ground truth (step 10)."""
        scored = self.score_class(class_name, seed)
        metrics: ClassMetrics = evaluate_class(
            [record.query.name for record in scored.records],
            [record.output.image_score for record in scored.records],
            [record.output.pixel_map for record in scored.records],
            self.dataset.ground_truth(class_name),
        )
        logger.info(
            "%s: image AUROC %.4f, pixel AUROC %.4f, AUPRO %.4f",
            class_name,
            metrics.image_auroc,
            metrics.pixel_auroc,
            metrics.aupro,
        )
        return ClassResult(
            class_name=class_name,
            metrics=metrics,
            records=scored.records,
            num_images_prepared=scored.num_images_prepared,
            feature_seconds=scored.feature_seconds,
            feature_peak_vram_mb=scored.feature_peak_vram_mb,
        )

    def run(self) -> list[Path]:
        """Run every seed; returns the per-seed result directories."""
        provenance = collect_provenance()
        if provenance["git"]["dirty"]:
            logger.warning("Uncommitted changes in tracked files: %s", provenance["git"]["dirty_files"])
        run_dirs = []
        for seed in self.config.experiment.seeds:
            run_dir = self.config.experiment.output_dir / self.config.experiment.name / f"seed_{seed}"
            setup_logging(run_dir / "run.log")
            seed_everything(seed)
            self.feature_store.cache.reset_counters()
            if self.geometry_store is not None:
                self.geometry_store.cache.reset_counters()
            logger.info(
                "Experiment %s, method %s, seed %d",
                self.config.experiment.name,
                self.config.method.name,
                seed,
            )
            summaries = []
            with stopwatch() as seed_seconds:
                for class_name in self.config.data.classes:
                    result = self.run_class(class_name, seed)
                    write_class_scores(run_dir, result)
                    summaries.append(summarise_class(result))
                    del result  # release this class's pixel maps before the next class
            write_seed_results(
                run_dir,
                config=self.config,
                seed=seed,
                summaries=summaries,
                provenance=provenance,
                total_seconds=seed_seconds[0],
                cache_stats=self._cache_stats(),
            )
            run_dirs.append(run_dir)
        return run_dirs

    def _cache_stats(self) -> dict[str, int]:
        stats = {
            "feature_hits": self.feature_store.cache.hits,
            "feature_misses": self.feature_store.cache.misses,
        }
        if self.geometry_store is not None:
            stats["geometry_hits"] = self.geometry_store.cache.hits
            stats["geometry_misses"] = self.geometry_store.cache.misses
        return stats
