"""Content-addressed tensor cache."""

from __future__ import annotations

import pytest
import torch

from padgeo.utils.cache import CacheIdentityError, TensorCache


def test_round_trip_is_exact_and_counted(tmp_path):
    cache = TensorCache(tmp_path, "features")
    identity = {"model": "m", "input": 518, "precision": "fp32", "image": "abc"}
    assert cache.load(identity) is None
    tensor = torch.randn(3, 4, dtype=torch.float32)
    cache.save(identity, {"x": tensor})
    loaded = cache.load(identity)
    assert loaded is not None
    assert torch.equal(loaded["x"], tensor)
    assert (cache.hits, cache.misses) == (1, 1)
    cache.reset_counters()
    assert (cache.hits, cache.misses) == (0, 0)
    assert not list(tmp_path.rglob("*.tmp"))  # atomic write leaves no temporary file


@pytest.mark.parametrize("field", ["model", "input", "precision", "image"])
def test_every_identity_field_changes_the_key(tmp_path, field):
    cache = TensorCache(tmp_path, "features")
    identity = {"model": "m", "input": 518, "precision": "fp32", "image": "abc"}
    cache.save(identity, {"x": torch.zeros(1)})
    changed = {**identity, field: "other"}
    assert cache.load(changed) is None


def test_key_order_does_not_matter(tmp_path):
    cache = TensorCache(tmp_path, "geometry")
    cache.save({"a": 1, "b": [1, 2]}, {"x": torch.ones(2)})
    assert cache.load({"b": [1, 2], "a": 1}) is not None


def test_mismatched_identity_inside_file_raises(tmp_path):
    cache = TensorCache(tmp_path, "features")
    identity = {"image": "abc"}
    cache.save(identity, {"x": torch.zeros(1)})
    path = next(tmp_path.rglob("*.pt"))
    entry = torch.load(path, weights_only=True)
    entry["identity"] = '{"image":"something else"}'
    torch.save(entry, path)
    with pytest.raises(CacheIdentityError):
        cache.load(identity)
