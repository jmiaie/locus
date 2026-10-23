"""Generic second-stage reranker hook (any object with rerank(query, chunks, limit) and key())."""
from dataclasses import replace
from pathlib import Path

import pytest

from locus.core import LocusEngine

DOCS = Path(__file__).resolve().parent.parent / "examples" / "construction" / "docs"


class LengthReranker:
    depth = 10

    def key(self):
        return "len"

    def rerank(self, query, chunks, limit=None):
        head, tail = chunks[: self.depth], chunks[self.depth:]
        head = sorted(head, key=lambda c: -len(c.content))
        out = [replace(c, provenance=c.provenance + "+rr") for c in head] + tail
        return out[:limit]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    p = tmp_path_factory.mktemp("hk") / ".locus"
    LocusEngine(store_path=p).index(DOCS)
    return p


def test_hook_reorders_and_toggles(store):
    plain = LocusEngine(store_path=store).retrieve("concrete slab", limit=10, use_cache=False)
    eng = LocusEngine(store_path=store, reranker=LengthReranker())
    got = eng.retrieve("concrete slab", limit=5, use_cache=False)
    assert [c.chunk_id for c in got] == [c.chunk_id for c in sorted(plain, key=lambda c: -len(c.content))[:5]]
    assert all(c.provenance.endswith("+rr") for c in got)
    off = eng.retrieve("concrete slab", limit=5, rerank=False, use_cache=False)
    assert [c.chunk_id for c in off] == [c.chunk_id for c in plain[:5]]
    with pytest.raises(ValueError):
        LocusEngine(store_path=store).retrieve("x", rerank=True)


def test_hook_with_parent_and_filters(store):
    eng = LocusEngine(store_path=store, reranker=LengthReranker())
    res = eng.retrieve("concrete", limit=3, parent="doc", filters={"path": "specs/*"}, use_cache=False)
    assert res and all(r.doc_path.startswith("specs/") and r.spans for r in res)
