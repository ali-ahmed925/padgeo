"""Seeded reference sampling and query selection."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from padgeo.data.base import ClassSplit, ImageRef
from padgeo.data.sampling import ReferenceSampler, select_queries


def ref(name: str, content: str) -> ImageRef:
    return ImageRef(name=name, path=Path("/nonexistent") / name, content_hash=content)


def split(pool_size: int = 30, queries: int = 6) -> ClassSplit:
    pool = tuple(
        sorted(
            (ref(f"train/good/{i}.png", f"{i:064x}") for i in range(pool_size)), key=lambda r: r.content_hash
        )
    )
    query_refs = tuple(ref(f"test/good/{i}.png", f"{1000 + i:064x}") for i in range(queries))
    return ClassSplit(class_name="01Toy", reference_pool=pool, queries=query_refs)


def test_per_query_is_deterministic_seed_dependent_and_canonically_ordered():
    s = split()
    first = ReferenceSampler(s, 4, "per_query", seed=0)
    again = ReferenceSampler(s, 4, "per_query", seed=0)
    other_seed = ReferenceSampler(s, 4, "per_query", seed=1)
    for query in s.queries:
        refs = first.sample(query)
        assert refs == again.sample(query)
        assert len(set(refs)) == 4
        assert [r.content_hash for r in refs] == sorted(r.content_hash for r in refs)
    assert [first.sample(q) for q in s.queries] != [other_seed.sample(q) for q in s.queries]
    # Fresh draws per query.
    assert len({first.sample(q) for q in s.queries}) > 1


def test_per_object_reuses_one_draw():
    s = split()
    sampler = ReferenceSampler(s, 4, "per_object", seed=3)
    draws = {sampler.sample(q) for q in s.queries}
    assert len(draws) == 1


def test_draw_depends_on_content_not_on_names_or_other_queries():
    s = split()
    query = s.queries[2]
    renamed = dataclasses.replace(query, name="test/Burrs/renamed.png", path=Path("/elsewhere/x.png"))
    sampler = ReferenceSampler(s, 4, "per_query", seed=0)
    assert sampler.sample(query) == sampler.sample(renamed)
    fewer_queries = ClassSplit(s.class_name, s.reference_pool, s.queries[2:3])
    assert ReferenceSampler(fewer_queries, 4, "per_query", seed=0).sample(query) == sampler.sample(query)


def test_class_name_changes_the_draw():
    s = split()
    other_class = ClassSplit("02Other", s.reference_pool, s.queries)
    q = s.queries[0]
    assert ReferenceSampler(s, 4, "per_query", 0).sample(q) != ReferenceSampler(
        other_class, 4, "per_query", 0
    ).sample(q)


def test_pool_too_small_and_identical_query_are_errors():
    with pytest.raises(ValueError, match="need 5"):
        ReferenceSampler(split(pool_size=4), 5, "per_query", seed=0)
    s = split(pool_size=2)
    duplicate = ref("test/good/dup.png", s.reference_pool[0].content_hash)
    with pytest.raises(ValueError, match="byte-identical"):
        ReferenceSampler(s, 2, "per_query", seed=0).sample(duplicate)


def test_select_queries():
    queries = tuple(ref(f"q{i:02d}", f"{i:064x}") for i in range(10))
    assert select_queries(queries, None) == queries
    assert select_queries(queries, 20) == queries
    picked = select_queries(queries, 4)
    assert [q.name for q in picked] == ["q00", "q03", "q06", "q09"]
