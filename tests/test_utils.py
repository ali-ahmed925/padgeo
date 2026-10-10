"""Seeding, provenance, resources and tensor checks."""

from __future__ import annotations

import pytest
import torch

from padgeo.utils.checks import check_tensor
from padgeo.utils.provenance import collect_provenance
from padgeo.utils.resources import stopwatch
from padgeo.utils.seeding import seed_everything


def test_seed_everything_makes_global_rngs_reproducible():
    seed_everything(7)
    first = torch.rand(5)
    seed_everything(7)
    assert torch.equal(first, torch.rand(5))
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)


def test_provenance_records_commit_versions_and_hardware():
    provenance = collect_provenance()
    assert len(provenance["git"]["commit"]) == 40
    assert isinstance(provenance["git"]["dirty"], bool)
    assert provenance["packages"]["torch"].startswith("2.3.1")
    assert "a288dd0f14786c93483e45524328726ab7b1b4ce" in provenance["packages"]["vggt"]
    assert "python" in provenance["hardware"]


def test_stopwatch_measures_elapsed_time():
    with stopwatch() as elapsed:
        sum(range(10_000))
    assert elapsed[0] > 0


@pytest.mark.parametrize(
    ("tensor", "kwargs", "message"),
    [
        (torch.zeros(2, 3), {"shape": (2, 4), "dtype": torch.float32}, "expected shape"),
        (torch.zeros(2, 3), {"shape": (2, None), "dtype": torch.float64}, "expected dtype"),
        (torch.tensor([float("nan")]), {"shape": (1,), "dtype": torch.float32, "finite": True}, "non-finite"),
        (torch.tensor([2.0]), {"shape": (1,), "dtype": torch.float32, "value_range": (0.0, 1.0)}, "outside"),
    ],
)
def test_check_tensor_fails_loudly(tensor, kwargs, message):
    with pytest.raises(ValueError, match=message):
        check_tensor("x", tensor, **kwargs)
