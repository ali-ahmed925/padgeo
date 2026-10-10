"""Experiment configuration: a strict, typed schema loaded from YAML (one file = one experiment).

Rules enforced here:

- Unknown or misspelled keys raise (every model forbids extra fields).
- Option-specific keys live inside discriminated unions, so a key that would have no effect for the chosen
  option is rejected instead of ignored: `k` only exists in kNN mode, `percent` only for top-percent
  image scores, and the geometry section only exists for `anchored_nn`.
- Defaults are the values stated in the project brief; every run saves its fully resolved config.

`docs/AUDIT.md` lists where each field is read and which test exercises it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Precision = Literal["fp32", "fp16", "bf16"]
PointSource = Literal["depth", "pointmap"]
Backbone = Literal[
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
    "dinov2_vitl14_reg",
    "dinov2_vitg14_reg",
]


class _Strict(BaseModel):
    # ser_json_inf_nan="strings": `radius: .inf` is written to JSON as "Infinity" rather than null.
    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")


class ExperimentConfig(_Strict):
    """Run identity and outputs.

    Attributes:
        name: Experiment name; results go to `<output_dir>/<name>/seed_<seed>/`.
        output_dir: Root directory for results.
        seeds: Seeds to run, each producing its own result files.
    """

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    output_dir: Path = Path("outputs")
    seeds: list[int] = Field(default=[0, 1, 2], min_length=1)

    @field_validator("seeds")
    @classmethod
    def _unique_non_negative(cls, seeds: list[int]) -> list[int]:
        if len(set(seeds)) != len(seeds) or min(seeds) < 0:
            raise ValueError(f"seeds must be unique and non-negative, got {seeds}")
        return seeds


class DataConfig(_Strict):
    """Dataset, classes and few-shot protocol.

    Attributes:
        dataset: Dataset loader name.
        root: Dataset root directory.
        classes: Class folder names to run, in order.
        num_references: N, defect-free references per query.
        sampling: "per_query" (fresh draw per query) or "per_object" (one draw per class and seed).
        max_queries_per_class: Smoke tests only: evaluate an evenly spaced subset of this many queries.
            None evaluates every query (all real experiments).
    """

    dataset: Literal["mad_sim"] = "mad_sim"
    root: Path
    classes: list[str] = Field(min_length=1)
    num_references: int = Field(default=4, ge=1)
    sampling: Literal["per_query", "per_object"] = "per_query"
    max_queries_per_class: int | None = Field(default=None, ge=2)


class FeatureConfig(_Strict):
    """Frozen DINOv2 patch features.

    Attributes:
        backbone: torch.hub entry point of the pinned DINOv2 repository.
        precision: Autocast precision of the forward pass ("fp32" disables autocast).
        device: Torch device for the forward pass.
        batch_size: Images per forward pass.
    """

    backbone: Backbone = "dinov2_vitl14"
    precision: Precision = "fp32"
    device: str = "cuda"
    batch_size: int = Field(default=8, ge=1)


class ForegroundConfig(_Strict):
    """Background thresholding for renders on a pure white background (see `padgeo.features.foreground`).

    Attributes:
        pixel_threshold: A model-input pixel is foreground if its darkest RGB channel (in [0, 1]) is below
            this value; pure white background is 1.0.
        min_patch_fraction: A patch is foreground if at least this fraction of its pixels is foreground.
    """

    pixel_threshold: float = Field(default=0.98, gt=0, le=1)
    min_patch_fraction: float = Field(default=0.1, gt=0, le=1)


class FakeGeometryConfig(_Strict):
    """Model-free geometry of a known synthetic scene, for tests and development without VGGT.

    Attributes:
        backend: Discriminator.
        point_source: "depth" unprojects synthetic depth with the synthetic cameras through the same
            code path as VGGT's depth branch; "pointmap" returns the ray-cast points directly.
        precision: Arithmetic precision of the ray casting (exercises the precision code path).
        device: Torch device for the ray casting.
    """

    backend: Literal["fake"]
    point_source: PointSource = "depth"
    precision: Precision = "fp32"
    device: str = "cpu"


class VGGTGeometryConfig(_Strict):
    """VGGT feed-forward geometry.

    Attributes:
        backend: Discriminator.
        model: Hugging Face model id.
        revision: Hugging Face commit of the checkpoint (pinned for reproducibility).
        point_source: "depth" unprojects the depth head with the predicted cameras (more accurate according
            to the VGGT README); "pointmap" uses the point-map head.
        precision: "fp32", or the autocast dtype of the aggregator ("bf16"/"fp16"); the heads always run
            in fp32, as in VGGT's own forward pass.
        device: Torch device.
    """

    backend: Literal["vggt"]
    model: str = "facebook/VGGT-1B"
    revision: str
    point_source: PointSource = "depth"
    precision: Precision = "bf16"
    device: str = "cuda"


GeometryConfig = Annotated[FakeGeometryConfig | VGGTGeometryConfig, Field(discriminator="backend")]


class RadiusNeighbourhood(_Strict):
    """Reference patches within `radius` times the reference cloud's bounding-box diagonal (principal axes).

    `radius: .inf` makes the constraint vacuous (anchored_nn then equals global_nn when
    `min_confidence` is 0).
    """

    mode: Literal["radius"]
    radius: float = Field(gt=0)


class KNNNeighbourhood(_Strict):
    """The `k` reference patches nearest in 3D."""

    mode: Literal["knn"]
    k: int = Field(ge=1)


Neighbourhood = Annotated[RadiusNeighbourhood | KNNNeighbourhood, Field(discriminator="mode")]


class GlobalNNConfig(_Strict):
    """Baseline: nearest neighbour among all foreground reference patches.

    Attributes:
        name: Discriminator.
        device: Torch device for the distance computations.
    """

    name: Literal["global_nn"]
    device: str = "cpu"


class AnchoredNNConfig(_Strict):
    """Ours: nearest neighbour among foreground reference patches close in 3D.

    Attributes:
        name: Discriminator.
        device: Torch device for the distance computations.
        geometry: Geometry backend producing the 3D points.
        neighbourhood: Radius or kNN definition of "close in 3D".
        unseen: Score of a query patch with no usable neighbour: the global_nn score
            ("fallback_global"), 0 ("zero"), or the maximum cosine distance 2 ("max").
        min_confidence: Patches whose geometry confidence is below this are not anchored: a query patch is
            treated as unseen, a reference patch is excluded from neighbourhoods. 0 disables the filter.
    """

    name: Literal["anchored_nn"]
    device: str = "cpu"
    geometry: GeometryConfig
    neighbourhood: Neighbourhood
    unseen: Literal["fallback_global", "zero", "max"] = "fallback_global"
    min_confidence: float = Field(ge=0)


MethodConfig = Annotated[GlobalNNConfig | AnchoredNNConfig, Field(discriminator="name")]


class TopPercentImageScore(_Strict):
    """Image score = mean of the top `percent` % of the full patch grid (AnomalyDINO's `mean_top1p`).

    k = max(1, floor(h * w * percent / 100)) highest scores, background patches scoring 0.
    """

    mode: Literal["top_percent"]
    percent: float = Field(gt=0, le=100)


class MaxImageScore(_Strict):
    """Image score = maximum patch score."""

    mode: Literal["max"]


ImageScore = Annotated[TopPercentImageScore | MaxImageScore, Field(discriminator="mode")]


class AggregationConfig(_Strict):
    """Patch scores to image score and pixel map (identical for both methods).

    Attributes:
        image_score: Image-level aggregation.
        pixel_sigma: Gaussian smoothing sigma of the pixel map, in original-image pixels (0 disables).
    """

    image_score: ImageScore = TopPercentImageScore(mode="top_percent", percent=1.0)
    pixel_sigma: float = Field(default=4.0, ge=0)


class CacheConfig(_Strict):
    """On-disk cache of DINOv2 features and patch-level geometry."""

    dir: Path = Path("cache")


class Config(_Strict):
    """One experiment."""

    experiment: ExperimentConfig
    data: DataConfig
    features: FeatureConfig = FeatureConfig()
    foreground: ForegroundConfig = ForegroundConfig()
    method: MethodConfig
    aggregation: AggregationConfig = AggregationConfig()
    cache: CacheConfig = CacheConfig()


class ProfileConfig(_Strict):
    """Geometry profiling run: peak VRAM, seconds per query, and the effect of precision on the points.

    The data, geometry backend, foreground and neighbourhood settings come from `experiment_config` (an
    anchored_nn config), so the profile measures exactly what that experiment runs. The profile then
    sweeps the number of references and the precision, which replace that config's `num_references` and
    `geometry.precision`.

    Attributes:
        name: Output sub-directory name.
        output_dir: Root directory for the report.
        experiment_config: Path of the anchored_nn experiment config to profile.
        num_references: Values of N to profile.
        precisions: Precisions to profile; the first is the reference for point differences.
        queries_per_setting: Queries of the first class to time per (N, precision).
    """

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    output_dir: Path = Path("outputs")
    experiment_config: Path
    num_references: list[int] = Field(default=[2, 4, 10], min_length=1)
    precisions: list[Precision] = Field(default=["fp32", "bf16"], min_length=1)
    queries_per_setting: int = Field(default=3, ge=1)


def load_profile_config(path: Path) -> tuple[ProfileConfig, Config]:
    """Load a profile config and the anchored_nn experiment config it points to.

    Raises:
        ValueError: If the referenced experiment is not an anchored_nn experiment.
    """
    with path.open(encoding="utf-8") as handle:
        profile = ProfileConfig.model_validate(yaml.safe_load(handle))
    experiment = load_config(profile.experiment_config)
    if not isinstance(experiment.method, AnchoredNNConfig):
        raise ValueError(f"{profile.experiment_config} must be an anchored_nn config")
    return profile, experiment


def load_config(path: Path) -> Config:
    """Load and validate a YAML experiment config.

    Raises:
        pydantic.ValidationError: On unknown keys, missing required keys, or invalid values.
    """
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Config.model_validate(raw)


def config_to_yaml(config: BaseModel) -> str:
    """Serialise a (resolved) config to YAML, with paths as strings and infinities as `.inf`."""
    data = config.model_dump(mode="python")
    return yaml.safe_dump(_plain(data), sort_keys=False)


def _plain(value: object) -> object:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value  # PyYAML writes float("inf") as .inf, which load_config reads back
