"""Cache identities: every setting a cached value depends on changes its key."""

from __future__ import annotations

from pathlib import Path

from padgeo.config import FakeGeometryConfig, FeatureConfig, ForegroundConfig, VGGTGeometryConfig
from padgeo.data.base import ImageRef
from padgeo.features.dinov2 import DINOv2Extractor, FeatureStore
from padgeo.geometry.fake import FakeGeometryBackend
from padgeo.geometry.store import GeometryStore
from padgeo.geometry.vggt_backend import VGGTBackend
from padgeo.utils.cache import TensorCache, canonical_json

PINNED_VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"


def refs(*hashes: str) -> list[ImageRef]:
    return [ImageRef(name=h, path=Path(h), content_hash=h) for h in hashes]


def geometry_key(images: list[ImageRef], foreground: ForegroundConfig | None = None, **fake: str) -> str:
    backend = FakeGeometryBackend(FakeGeometryConfig(backend="fake", **fake))
    store = GeometryStore(
        backend, TensorCache(Path("/nonexistent"), "geometry"), foreground or ForegroundConfig()
    )
    return canonical_json(store._identity(images))


def test_geometry_key_changes_with_every_setting_and_with_image_order():
    images = refs("q", "a", "b")
    variants = [
        geometry_key(images),
        geometry_key(images, precision="bf16"),
        geometry_key(images, point_source="pointmap"),
        geometry_key(images, foreground=ForegroundConfig(pixel_threshold=0.9)),
        geometry_key(images, foreground=ForegroundConfig(min_patch_fraction=0.5)),
        geometry_key(refs("q", "b", "a")),  # order matters: image 0 defines the frame
        geometry_key(refs("x", "a", "b")),  # another query
    ]
    assert len(set(variants)) == len(variants)


def test_vggt_identity_names_model_revision_code_commit_point_source_and_precision():
    def identity(**update: str) -> dict:
        config = VGGTGeometryConfig(backend="vggt", revision="r1").model_copy(update=update)
        return VGGTBackend(config).identity

    base = identity()
    assert PINNED_VGGT_COMMIT in base["vggt_package"]  # not just "0.0.1", which every commit reports
    for update in ({"revision": "r2"}, {"precision": "fp32"}, {"point_source": "pointmap"}, {"model": "m"}):
        assert identity(**update) != base


def test_feature_key_changes_with_backbone_precision_input_size_and_preprocessing_code():
    def key(content_hash: str = "img", input_hw: tuple[int, int] = (518, 518), **config: str) -> str:
        extractor = DINOv2Extractor(FeatureConfig(**{"backbone": "dinov2_vits14", "device": "cpu", **config}))
        store = FeatureStore(extractor, TensorCache(Path("/nonexistent"), "features"))
        return canonical_json(store._identity(content_hash, input_hw))

    keys = [
        key(),
        key(backbone="dinov2_vitl14"),
        key(precision="bf16"),
        key(input_hw=(252, 518)),
        key("other"),
    ]
    assert len(set(keys)) == len(keys)
    assert PINNED_VGGT_COMMIT in key()  # VGGT's preprocessing produces the input tensor
