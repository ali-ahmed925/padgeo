"""Strict config loading: unknown keys and option-irrelevant keys are errors; repo configs all validate."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from padgeo.config import (
    AnchoredNNConfig,
    Config,
    TopPercentImageScore,
    VGGTGeometryConfig,
    config_to_yaml,
    load_config,
    load_profile_config,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "configs").rglob("*.yaml"))
EXPERIMENT_CONFIGS = [p for p in CONFIGS if p.name != "reference_numbers.yaml" and "profile" not in p.name]
PROFILE_CONFIGS = [p for p in CONFIGS if "profile" in p.name]

BASE = {
    "experiment": {"name": "t", "seeds": [0]},
    "data": {"root": "data/MAD-Sim", "classes": ["03Mallard"]},
    "method": {
        "name": "anchored_nn",
        "geometry": {"backend": "fake"},
        "neighbourhood": {"mode": "radius", "radius": 0.05},
        "min_confidence": 1.5,
    },
}


def validate(raw: dict) -> Config:
    return Config.model_validate(raw)


def with_change(path: list[str], value: object) -> dict:
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    node = raw
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    return raw


def test_base_config_and_brief_defaults():
    config = validate(BASE)
    assert config.features.backbone == "dinov2_vitl14"
    assert config.data.sampling == "per_query"
    assert config.data.num_references == 4
    assert isinstance(config.method, AnchoredNNConfig)
    assert config.method.geometry.point_source == "depth"
    assert config.method.unseen == "fallback_global"
    assert isinstance(config.aggregation.image_score, TopPercentImageScore)
    assert config.aggregation.image_score.percent == 1.0


@pytest.mark.parametrize(
    "path",
    [
        ["typo_section"],
        ["data", "num_refs"],
        ["method", "radius"],
        ["method", "geometry", "model"],  # fake backend has no model
        ["method", "neighbourhood", "k"],  # k is not a radius-mode key
        ["aggregation", "image_score", "percnt"],
    ],
)
def test_unknown_or_irrelevant_keys_are_rejected(path):
    raw = with_change(path, 3)
    if path[:2] == ["aggregation", "image_score"]:
        raw["aggregation"]["image_score"].update(mode="top_percent", percent=1.0)  # only the typo is wrong
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate(raw)


def test_geometry_section_is_rejected_for_global_nn():
    raw = with_change(["method"], {"name": "global_nn", "geometry": {"backend": "fake"}})
    with pytest.raises(ValidationError):
        validate(raw)


def test_percent_is_rejected_for_max_image_score():
    raw = with_change(["aggregation"], {"image_score": {"mode": "max", "percent": 1.0}})
    with pytest.raises(ValidationError):
        validate(raw)


def test_anchored_requires_radius_and_confidence_explicitly():
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    del raw["method"]["min_confidence"]
    with pytest.raises(ValidationError):
        validate(raw)
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    del raw["method"]["neighbourhood"]
    with pytest.raises(ValidationError):
        validate(raw)


@pytest.mark.parametrize("bad", [{"seeds": [0, 0]}, {"seeds": [-1]}, {"seeds": []}, {"name": "has space"}])
def test_invalid_experiment_values(bad):
    raw = with_change(["experiment"], {"name": "t", "seeds": [0], **bad})
    with pytest.raises(ValidationError):
        validate(raw)


def test_infinite_radius_round_trips_through_yaml(tmp_path):
    raw = with_change(["method", "neighbourhood"], {"mode": "radius", "radius": math.inf})
    config = validate(raw)
    path = tmp_path / "c.yaml"
    path.write_text(config_to_yaml(config))
    assert ".inf" in path.read_text()
    assert load_config(path) == config
    assert '"Infinity"' in config.model_dump_json()


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_repository_experiment_configs_validate(path):
    load_config(path)


@pytest.mark.parametrize("path", PROFILE_CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_repository_profile_configs_validate(path):
    profile, experiment = load_profile_config(path)
    assert experiment.method.name == "anchored_nn"
    assert profile.precisions[0] == "fp32"  # differences are measured against full precision


def test_killtest_configs_differ_only_in_method():
    global_cfg = load_config(REPO / "configs" / "killtest_global.yaml")
    anchored_cfg = load_config(REPO / "configs" / "killtest_anchored.yaml")
    for section in ("data", "features", "foreground", "aggregation", "cache"):
        assert getattr(global_cfg, section) == getattr(anchored_cfg, section)
    assert global_cfg.experiment.seeds == anchored_cfg.experiment.seeds == [0, 1, 2]
    assert global_cfg.data.classes == ["03Mallard", "04Turtle", "18Obesobeso"]
    assert global_cfg.data.num_references == 4
    assert global_cfg.features.backbone == "dinov2_vitl14"
    assert isinstance(anchored_cfg.method, AnchoredNNConfig)
    assert isinstance(anchored_cfg.method.geometry, VGGTGeometryConfig)
    assert anchored_cfg.method.geometry.model == "facebook/VGGT-1B"


def test_duplicate_keys_are_rejected(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text(yaml.safe_dump(BASE) + "experiment:\n  name: again\n")  # a second top-level key
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate keys"):
        load_config(path)
