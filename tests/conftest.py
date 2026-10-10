"""Pytest fixtures shared by the test modules (helpers live in tests/helpers.py)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.helpers import build_dataset, transforms_json_forbidden


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture dataset (MAD-Sim format), built once per session."""
    return build_dataset(tmp_path_factory.mktemp("MAD-Sim-fixture"))


@pytest.fixture
def forbid_transforms_json() -> Iterator[None]:  # noqa: PT004 - used for its side effect
    """Fail the test if anything opens a transforms.json."""
    with transforms_json_forbidden():
        yield
