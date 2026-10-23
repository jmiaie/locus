"""Optional reranking hook (no model download: scorer is stubbed)."""
import importlib.util
from pathlib import Path

import pytest

from locus.core import LocusEngine
from locus.rerank import CrossEncoderReranker

DOCS = Path(__file__).resolve().parent.parent / "examples" / "construction" / "docs"


class StubCE(CrossEncoderReranker):
    """Scores by number of characters: deterministic, no model."""
    calls = 0

    def score(self, query, texts):
        StubCE.calls += 1
        return [float(len(t)) for t in texts]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    p = tmp_path_factory.mktemp("rr") / ".locus"
    LocusEngine(store_path=p).index(DOCS)
    return p


def test_rerank_reorders_top_depth(store):
    plain = LocusEngine(store_path=store).retrieve("concrete slab", limit=20, use_cache=False)
    eng = LocusEngine(store_path=store, reranker=StubCE(depth=10))
    got = eng.retrieve("concrete slab", limit=5, use_cache=False)
    head = sorted(plain[:10], key=lambda c: -len(c.content))
    assert [c.chunk_id for c in got] == [c.chunk_id for c in head[:5]]
    assert all(c.provenance.endswith("+ce") for c in got)
    assert got[0].score == float(len(got[0].content))


def test_rerank_toggle_and_errors(store):
    eng = LocusEngine(store_path=store, reranker=StubCE(depth=10))
    plain = LocusEngine(store_path=store)
    assert [c.chunk_id for c in eng.retrieve("concrete slab", limit=5, rerank=False, use_cache=False)] == \
           [c.chunk_id for c in plain.retrieve("concrete slab", limit=5, use_cache=False)]
    with pytest.raises(ValueError):
        plain.retrieve("x", rerank=True)
    with pytest.raises(ValueError):
        CrossEncoderReranker(depth=0)


def test_rerank_cached_separately(store):
    eng = LocusEngine(store_path=store, reranker=StubCE(depth=10))
    a = eng.retrieve("roof drain", limit=3)
    b = eng.retrieve("roof drain", limit=3, rerank=False)
    assert [c.provenance for c in a] != [c.provenance for c in b]


def test_rerank_with_parent_and_filters(store):
    eng = LocusEngine(store_path=store, reranker=StubCE(depth=20))
    res = eng.retrieve("concrete", limit=3, parent="doc", filters={"path": "specs/*"}, use_cache=False)
    assert res and all(r.doc_path.startswith("specs/") and r.spans for r in res)


def test_rerank_tail_keeps_order():
    from locus.retrieval.bm25 import ScoredChunk
    cs = [ScoredChunk(str(i), "d", 1.0 / (i + 1), "x" * (i + 1), "bm25") for i in range(6)]
    out = StubCE(depth=3).rerank("q", cs)
    assert [c.chunk_id for c in out] == ["2", "1", "0", "3", "4", "5"]


@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is not None,
                    reason="extra installed")
def test_missing_extra_message():
    with pytest.raises(ImportError, match=r"locus\[rerank\]"):
        CrossEncoderReranker().score("q", ["t"])


def test_core_does_not_import_rerank():
    import subprocess, sys
    code = "import sys, locus, locus.core; print('locus.rerank' in sys.modules, 'torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False False"


@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is not None,
                    reason="extra installed")
def test_cli_rerank_without_extra_fails_cleanly(store):
    import os, subprocess, sys
    root = Path(__file__).resolve().parent.parent
    p = subprocess.run([sys.executable, "-m", "locus", "--store", str(store), "retrieve", "slab", "--rerank"],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(root)))
    assert p.returncode != 0 and "locus[rerank]" in p.stderr
