"""Seeded few-shot reference sampling.

Both methods call the same sampler with the same arguments, so for a given (seed, class, query) they see
exactly the same reference images. Randomness is derived from content hashes, never from file names or
iteration order, so a query's references do not depend on which other queries are run, on folder names
(which encode defect types), or on the order in which the loader lists files.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Literal

import numpy as np

from padgeo.data.base import ClassSplit, ImageRef

SamplingMode = Literal["per_query", "per_object"]


def _hash_to_int(text: str) -> int:
    """A stable 64-bit integer derived from `text` (Python's `hash` is salted per process)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


class ReferenceSampler:
    """Draws N distinct references from a class's defect-free pool.

    Args:
        split: The class split; only its `reference_pool` and `class_name` are used.
        num_references: N, the number of references per query.
        mode: "per_query" draws a fresh set for every query; "per_object" draws one set per (class, seed)
            and reuses it for every query.
        seed: Experiment seed.

    Raises:
        ValueError: If the pool has fewer than N images.
    """

    def __init__(self, split: ClassSplit, num_references: int, mode: SamplingMode, seed: int) -> None:
        if num_references < 1:
            raise ValueError(f"num_references must be >= 1, got {num_references}")
        if len(split.reference_pool) < num_references:
            raise ValueError(
                f"{split.class_name}: pool has {len(split.reference_pool)} images, need {num_references}"
            )
        self._pool = split.reference_pool
        self._num_references = num_references
        self._entropy = [seed, _hash_to_int(split.class_name)]
        self._fixed = self._draw(self._entropy) if mode == "per_object" else None

    def _draw(self, entropy: list[int]) -> tuple[ImageRef, ...]:
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        indices = rng.choice(len(self._pool), size=self._num_references, replace=False)
        # Canonical order (by content hash): the geometry model treats its first frame specially, so the
        # order passed downstream must not depend on the draw order.
        return tuple(sorted((self._pool[i] for i in indices), key=lambda ref: ref.content_hash))

    def sample(self, query: ImageRef) -> tuple[ImageRef, ...]:
        """References for `query`, sorted by content hash.

        Raises:
            ValueError: If a drawn reference is byte-identical to the query.
        """
        if self._fixed is not None:
            references = self._fixed
        else:
            references = self._draw([*self._entropy, _hash_to_int(query.content_hash)])
        if any(ref.content_hash == query.content_hash for ref in references):
            raise ValueError(f"query {query.name} is byte-identical to one of its references")
        return references


def select_queries(queries: Sequence[ImageRef], max_queries: int | None) -> tuple[ImageRef, ...]:
    """All queries, or (smoke tests only) `max_queries` of them evenly spaced in name order."""
    if max_queries is None or max_queries >= len(queries):
        return tuple(queries)
    indices = np.linspace(0, len(queries) - 1, num=max_queries).round().astype(int)
    return tuple(queries[i] for i in sorted(set(indices.tolist())))
