"""DINOv2 extractor (ViT-S, CPU): grid layout, alignment with the input pixels, batching and caching."""

from __future__ import annotations

import pytest
import torch

from padgeo.config import FeatureConfig
from padgeo.features.dinov2 import DINOv2Extractor, FeatureStore, unpack_features
from padgeo.utils.cache import TensorCache

CONFIG = FeatureConfig(backbone="dinov2_vits14", precision="fp32", device="cpu", batch_size=2)


@pytest.fixture(scope="module")
def extractor() -> DINOv2Extractor:
    return DINOv2Extractor(CONFIG)


@pytest.fixture(scope="module")
def images() -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    base = torch.ones(3, 3, 518, 518)
    base[:, :, 150:370, 150:370] = torch.rand(3, 3, 1, 1, generator=gen)  # a coloured square on white
    return base


def test_feature_grid_shape_norm_and_storage(extractor, images):
    stored = extractor.extract(images)
    assert stored.shape == (3, 37, 37, 384)
    assert stored.dtype == torch.float16
    features = unpack_features(stored[0])
    assert features.dtype == torch.float32
    assert torch.allclose(features.norm(dim=-1), torch.ones(37, 37), atol=1e-5)


def test_token_grid_is_row_major_and_aligned_with_pixels(extractor):
    """Changing the pixels of one patch changes that patch's token most (row i, column j)."""
    image = torch.full((1, 3, 518, 518), 0.5)
    perturbed = image.clone()
    i, j = 7, 29  # deliberately not symmetric
    perturbed[0, :, 14 * i : 14 * i + 14, 14 * j : 14 * j + 14] = 0.0
    before = unpack_features(extractor.extract(image)[0])
    after = unpack_features(extractor.extract(perturbed)[0])
    change = (before - after).norm(dim=-1)
    row, col = divmod(int(change.argmax()), 37)
    assert (row, col) == (i, j)


def test_batch_size_does_not_change_features(extractor, images):
    together = extractor.extract(images)
    one_by_one = torch.cat([extractor.extract(images[k : k + 1]) for k in range(3)])
    # Stored in float16; different batch sizes may pick different kernels, so allow one float16 ulp.
    assert torch.allclose(together.float(), one_by_one.float(), atol=2e-3)


def test_feature_store_fresh_equals_cached(tmp_path, extractor, images):
    store = FeatureStore(extractor, TensorCache(tmp_path, "features"))
    hashes = ["a", "b", "c"]
    assert store.lookup("a", (518, 518)) is None
    fresh = store.compute_and_store(hashes, images)
    cached = [store.lookup(h, (518, 518)) for h in hashes]
    for f, c in zip(fresh, cached, strict=True):
        assert c is not None
        assert torch.equal(f, c)
    other_input = store.lookup("a", (252, 518))
    assert other_input is None  # the input resolution is part of the key


def test_identity_names_everything_features_depend_on(extractor):
    identity = extractor.identity
    assert identity["backbone"] == "dinov2_vits14"
    assert identity["precision"] == "fp32"
    assert "7764ea0f912e53c92e82eb78a2a1631e92725fc8" in identity["hub_repo"]
