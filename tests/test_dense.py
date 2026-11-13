"""Optional dense re-rank (DenseReranker) with an injected encoder: no model download, no extra deps."""
import importlib.util
import re

import pytest

from locus.core import LocusEngine
from locus.dense import ADVICE_RE, DenseReranker, _join_chunks, text_windows
from locus.retrieval.bm25 import ScoredChunk

VOCAB = ["hotel", "beach", "miami", "budget", "luxury", "concrete", "slab", "rebar", "pool", "suggest"]


def bow(texts):
    """Deterministic bag-of-words vectors over a tiny vocabulary (+1 bias dim so no zero vectors)."""
    out = []
    for t in texts:
        toks = re.findall(r"\w+", t.lower())
        out.append([float(toks.count(w)) for w in VOCAB] + [0.01])
    return out


class Counting:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return bow(texts)


DOCS = {
    "a.md": "Concrete slab pour schedule. The slab needs rebar inspection before the pour. concrete slab",
    "b.md": "Notes: I prefer luxury hotel stays with a pool, near the beach in Miami. Budget is flexible.",
    "c.md": "Concrete mix design for the slab and the footing; slab curing.",
    "d.md": "Grocery list and errands for the weekend; slab of chocolate.",
}


@pytest.fixture()
def store(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for k, v in DOCS.items():
        (d / k).write_text(v + "\n")
    p = tmp_path / ".locus"
    LocusEngine(store_path=p).index(d)
    return p


def _docs(res):
    return list(dict.fromkeys(c.doc_path.split("/")[-1] for c in res))


def test_windows():
    assert text_windows("a b c") == ["a b c"]
    w = text_windows(" ".join(str(i) for i in range(500)), 200, 150)
    assert [x.split()[0] for x in w] == ["0", "150", "300"] and w[-1].split()[-1] == "499"
    assert len(w[0].split()) == 200


def test_rrf_fusion_math():
    pool = [ScoredChunk(f"{d}#0", d, 1.0, txt, "bm25") for d, txt in
            [("w", "slab"), ("x", "concrete slab"), ("y", "hotel beach miami"), ("z", "rebar")]]
    rr = DenseReranker(encoder=bow, rrf_k=60)
    out = rr.rerank("suggest a beach hotel", pool, limit=4)
    # bm = w,x,y,z ; dense = y, then w,x,z (bias-only tie -> bm order)
    # w: 1/61+1/62 > y: 1/63+1/61 > x: 1/62+1/63 > z: 1/64+1/64
    assert [c.doc_path for c in out] == ["w", "y", "x", "z"]
    assert all(c.provenance.endswith("+dense") for c in out)
    assert out[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_engine_dense_moves_semantic_match_up(store):
    q = "can you suggest a hotel for my miami trip?"
    plain = _docs(LocusEngine(store_path=store).retrieve(q, limit=10, use_cache=False))
    eng = LocusEngine(store_path=store, reranker=DenseReranker(encoder=bow))
    got = _docs(eng.retrieve(q, limit=10, use_cache=False))
    assert set(got) == set(plain)
    assert got.index("b.md") <= plain.index("b.md")
    # rerank=False == engine without a reranker
    assert _docs(eng.retrieve(q, limit=10, use_cache=False, rerank=False)) == plain


def test_bound_engine_uses_full_doc_text(store):
    rr = DenseReranker(encoder=bow)
    eng = LocusEngine(store_path=store, reranker=rr)
    assert rr._engine is eng
    docs = eng.corpus.list_docs()
    texts = rr._doc_texts(docs, [])
    assert any("luxury hotel" in t for t in texts.values())


def test_advice_route_untouched_and_lazy(store):
    enc = Counting()
    rr = DenseReranker(encoder=enc, route="advice")
    eng = LocusEngine(store_path=store, reranker=rr)
    plain = LocusEngine(store_path=store)
    q = "concrete slab pour"
    assert [c.chunk_id for c in eng.retrieve(q, limit=5, use_cache=False)] == \
           [c.chunk_id for c in plain.retrieve(q, limit=5, use_cache=False)]
    assert enc.calls == 0
    eng.retrieve("any tips for a miami hotel?", limit=5, use_cache=False)
    assert enc.calls == 1
    assert ADVICE_RE.search("Can you recommend something?") and not ADVICE_RE.search("what did I say?")


def test_embedding_cache_reused(store):
    enc = Counting()
    rr = DenseReranker(encoder=enc)
    eng = LocusEngine(store_path=store, reranker=rr)
    eng.retrieve("slab hotel", limit=5, use_cache=False)
    n = len(rr._cache)
    eng.retrieve("slab hotel beach", limit=5, use_cache=False)
    assert len(rr._cache) == n and enc.calls == 2  # second call encodes only the query


def test_key_and_validation():
    assert DenseReranker().key() != DenseReranker(route="advice").key()
    assert DenseReranker().key() != DenseReranker(depth=50).key()
    with pytest.raises(ValueError):
        DenseReranker(route="nope")
    with pytest.raises(ValueError):
        DenseReranker(depth=0)
    with pytest.raises(ValueError):
        DenseReranker(stride_words=300, window_words=200)


def test_join_chunks_drops_overlap():
    class C:
        def __init__(self, content, start_word):
            self.content, self.start_word = content, start_word
    assert _join_chunks([C("a b c d", 0), C("c d e f", 2), C("g", 6)]) == "a b c d e f g"


def test_core_does_not_import_dense():
    import subprocess
    import sys
    code = "import sys, locus, locus.core; print('locus.dense' in sys.modules, 'sentence_transformers' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is None, reason="needs the dense extra")
def test_real_model_smoke(store):  # pragma: no cover - only with the extra installed and the model cached
    eng = LocusEngine(store_path=store, reranker=DenseReranker(threads=1))
    res = eng.retrieve("can you suggest a hotel for my trip?", limit=5, use_cache=False)
    assert res and "b.md" in _docs(res)


def test_small_cache_eviction_does_not_break_scoring(store):
    rr = DenseReranker(encoder=bow, cache_docs=1)
    eng = LocusEngine(store_path=store, reranker=rr)
    for q in ("slab hotel", "concrete beach", "slab hotel"):
        assert eng.retrieve(q, limit=5, use_cache=False)
    assert len(rr._cache) == 1
