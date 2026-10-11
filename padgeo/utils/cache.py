"""Content-addressed on-disk cache for tensors.

An entry is identified by an *identity*: a JSON-serialisable mapping listing everything the cached value
depends on (model name and revision, input resolution, precision, image content hashes, ...). The file
name is the SHA-256 of the canonical JSON of the identity. The identity is also stored inside the entry
and compared on load, so a stale or colliding entry is never returned silently. Writes are atomic
(temporary file, then rename), so an interrupted run cannot leave a truncated entry behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

# Bump when the meaning of any cached value changes; it is part of every identity.
# 2: VGGT code commit added to the feature and geometry identities.
CACHE_FORMAT_VERSION = 2


class CacheIdentityError(RuntimeError):
    """A cache file exists under the expected name but records a different identity."""


def canonical_json(identity: Mapping[str, Any]) -> str:
    """Serialise an identity mapping deterministically (sorted keys, no whitespace)."""
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)


class TensorCache:
    """A directory of `.pt` files, each holding a dict of CPU tensors plus the identity that produced it.

    Args:
        root: Cache root directory (from the config's `cache.dir`).
        namespace: Sub-directory separating kinds of entries, e.g. "features" or "geometry".
    """

    def __init__(self, root: Path, namespace: str) -> None:
        self.directory = root / namespace
        self.hits = 0
        self.misses = 0

    def reset_counters(self) -> None:
        """Zero the hit and miss counters (each seed reports its own)."""
        self.hits = 0
        self.misses = 0

    def _path(self, identity_json: str) -> Path:
        digest = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
        return self.directory / digest[:2] / f"{digest}.pt"

    def load(self, identity: Mapping[str, Any]) -> dict[str, torch.Tensor] | None:
        """Return the cached tensors for `identity`, or None on a miss.

        Raises:
            CacheIdentityError: If the file exists but was written for a different identity.
        """
        identity_json = canonical_json({**identity, "cache_format_version": CACHE_FORMAT_VERSION})
        path = self._path(identity_json)
        if not path.exists():
            self.misses += 1
            return None
        entry = torch.load(path, map_location="cpu", weights_only=True)
        if entry["identity"] != identity_json:
            raise CacheIdentityError(f"{path} records identity {entry['identity']}, expected {identity_json}")
        self.hits += 1
        tensors: dict[str, torch.Tensor] = entry["tensors"]
        return tensors

    def save(self, identity: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> None:
        """Atomically write `tensors` (moved to CPU) under `identity`, replacing any existing entry."""
        identity_json = canonical_json({**identity, "cache_format_version": CACHE_FORMAT_VERSION})
        path = self._path(identity_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "identity": identity_json,
            # clone: a view (e.g. one image of a batch) would otherwise serialise its whole parent storage.
            "tensors": {name: tensor.detach().to("cpu").clone() for name, tensor in tensors.items()},
        }
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            torch.save(payload, tmp_path)
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
