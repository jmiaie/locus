"""Chunking options, parent-child retrieval and RM3 expansion (correctness)."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from locus.core import LocusEngine
from locus.mcp.server import _call_tool
from locus.memory.chunker import Chunker, normalize_chunking
from locus.retrieval.prf import RM3Config, expansion_model, rm3_search

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "examples" / "construction" / "docs"
norm = lambda t: " ".join(t.split())


@pytest.fixture(scope="module")
def eng(tmp_path_factory):
    e = LocusEngine(store_path=tmp_path_factory.mktemp("rp") / ".locus")
    e.index(DOCS)
    return e


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    e = LocusEngine(store_path=tmp_path_factory.mktemp("rs") / ".locus",
                    chunking={"chunk_words": 40, "overlap_words": 8, "heading_context": True, "boundaries": "lines"})
    e.index(DOCS)
    return e


# ---------------------------------------------------------------- chunking

def test_default_chunk_ids_unchanged():
    text = "# Title\n\nshort body text here\n\n## Next\n\n" + " ".join(f"w{i}" for i in range(900))
    chunks = Chunker().chunk_text(text, source="d.md")
    assert chunks[0].id == hashlib.sha256(b"d.md:section:Title").hexdigest()[:16]
    assert [c.start_word for c in chunks[1:]] == [0, 350, 700]
    assert chunks[1].id == hashlib.sha256(b"d.md:Next:0").hexdigest()[:16]
    assert chunks[1].content.startswith("Next w0 w1")


def test_normalize_chunking_validation():
    assert normalize_chunking(None)["chunk_words"] == 400
    with pytest.raises(ValueError):
        normalize_chunking({"chunk_words": 100, "overlap_words": 100})
    with pytest.raises(ValueError):
        normalize_chunking({"boundaries": "sentences"})
    with pytest.raises(ValueError):
        normalize_chunking({"nope": 1})


def test_heading_context_breadcrumb_on_every_window():
    text = "# Spec\n\nintro text for the spec\n\n## Slabs\n\n" + " ".join(f"w{i}" for i in range(100))
    chunks = Chunker(chunk_words=30, overlap_words=5, heading_context=True).chunk_text(text, "s.md")
    slab = [c for c in chunks if c.metadata["section"] == "Slabs"]
    assert len(slab) >= 3 and all(c.content.startswith("Spec > Slabs\n\n") for c in slab)


def test_line_boundaries_end_on_line_breaks():
    lines = [" ".join(f"t{i}_{j}" for j in range(7)) for i in range(40)]
    chunks = Chunker(chunk_words=30, overlap_words=0, boundaries="lines").chunk_text("\n".join(lines), "c.md")
    assert len(chunks) > 3
    for c in chunks[:-1]:
        assert re.match(r"t\d+_6$", c.content.split()[-1])  # window ended at the end of a line


def test_spans_point_into_parent_section():
    text = "# A\n\n" + "\n".join(" ".join(f"a{i}{j}" for j in range(9)) for i in range(30))
    ch = Chunker(chunk_words=25, overlap_words=5, heading_context=True)
    chunks, sections = ch.chunk_text_with_sections(text, "p.md")
    sec = {s.id: s for s in sections}
    for c in chunks:
        s = sec[c.metadata["parent_id"]]
        a, b = c.metadata["span"]
        assert norm(s.content[a:b]) == norm(c.content.split("\n\n", 1)[1])


def test_chunking_change_rechunks_on_next_index(tmp_path):
    store = tmp_path / ".locus"
    e = LocusEngine(store_path=store)
    e.index(DOCS)
    n_default = e.corpus.chunk_count()
    e2 = LocusEngine(store_path=store, chunking={"chunk_words": 40, "overlap_words": 0})
    e2.index(DOCS)
    assert e2.corpus.chunk_count() > n_default
    assert LocusEngine(store_path=store).corpus.chunking["chunk_words"] == 40  # persisted


# ---------------------------------------------------------------- parent-child

def test_parent_section_returns_sections_with_cited_spans(small):
    q = "Level 2 slab compressive strength"
    res = small.retrieve(q, limit=3, parent="section", use_cache=False)
    assert len(res) == 3 and len({r.chunk_id for r in res}) == 3
    for r in res:
        assert r.spans and r.provenance.endswith("+parent")
        for s in r.spans:
            assert 0 <= s["start"] < s["end"] <= len(r.content)
    top = res[0]
    cited = top.content[top.spans[0]["start"]:top.spans[0]["end"]].lower()
    assert "slab" in cited or "strength" in cited
    # Parent score is its best chunk's score
    assert top.score == max(s["score"] for s in top.spans)


def test_parent_doc_unique_docs_and_offsets(small):
    res = small.retrieve("curtain wall embeds", limit=4, parent="doc", use_cache=False)
    docs = [r.doc_path for r in res]
    assert len(docs) == len(set(docs)) == 4
    chunks = small.corpus.get_chunks_batch([s["chunk_id"] for r in res for s in r.spans])
    for r in res:
        for s in r.spans:
            body = chunks[s["chunk_id"]].content.split("\n\n", 1)[-1]
            assert norm(r.content[s["start"]:s["end"]]) == norm(body)


def test_parent_doc_ranking_equals_chunk_dedup(eng):
    q = "roof drainage change order"
    dedup = list(dict.fromkeys(c.doc_path for c in eng.retrieve(q, limit=60, use_cache=False)))[:5]
    assert [r.doc_path for r in eng.retrieve(q, limit=5, parent="doc", use_cache=False)] == dedup


def test_parent_with_filters(small):
    res = small.retrieve("concrete", limit=3, parent="section", filters={"path": "specs/*"}, use_cache=False)
    assert res and all(r.doc_path.startswith("specs/") for r in res)


def test_invalid_parent_and_expansion(eng):
    with pytest.raises(ValueError):
        eng.retrieve("x", parent="paragraph")
    with pytest.raises(ValueError):
        eng.retrieve("x", expansion="llm")


# ---------------------------------------------------------------- RM3

def test_score_terms_sums_to_bm25(eng):
    terms = eng.corpus.analyze("slab compressive strength")
    ranked = eng.corpus.search_bm25(terms, limit=10)
    per = eng.corpus.score_terms(terms, [c for c, _ in ranked])
    for cid, s in ranked:
        assert abs(sum(per.get(t, {}).get(cid, 0.0) for t in terms) - s) < 1e-6


def test_rm3_model_contains_query_and_feedback_terms(eng):
    m = expansion_model(eng.corpus, "slab compressive strength", RM3Config(fb_docs=5, fb_terms=10, orig_weight=0.5))
    assert abs(sum(m.values()) - 1.0) < 1e-9
    q = set(eng.corpus.analyze("slab compressive strength"))
    assert q <= set(m) and len(set(m) - q) > 0
    assert sum(m[t] for t in q) >= 0.5 - 1e-9


def test_rm3_orig_weight_one_reproduces_bm25_order(eng):
    q = "curtain wall embeds slab edge"
    plain = [c for c, _ in eng.corpus.search_bm25(eng.corpus.analyze(q), limit=10)]
    rm3 = [c for c, _ in rm3_search(eng.corpus, q, 10, {"orig_weight": 1.0})]
    assert rm3 == plain


def test_rm3_deterministic_and_backend_independent(tmp_path, eng):
    leg = LocusEngine(store_path=tmp_path / ".l", index_backend="legacy")
    leg.index(DOCS)
    q = "Level 2 slab compressive strength"
    a = [(c.chunk_id, round(c.score, 6)) for c in eng.retrieve(q, limit=10, expansion="rm3", use_cache=False)]
    b = [(c.chunk_id, round(c.score, 6)) for c in leg.retrieve(q, limit=10, expansion="rm3", use_cache=False)]
    assert a == b and a == [(c.chunk_id, round(c.score, 6)) for c in
                            eng.retrieve(q, limit=10, expansion="rm3", use_cache=False)]
    assert all(c.provenance == "bm25+rm3" for c in eng.retrieve(q, limit=3, expansion="rm3"))


def test_rm3_cache_key_and_config(eng):
    q = "weekend work hours noise"
    a = eng.retrieve(q, limit=5)
    b = eng.retrieve(q, limit=5, expansion="rm3")
    assert [c.provenance for c in a] != [c.provenance for c in b]
    with pytest.raises(ValueError):
        RM3Config.from_value({"fb_docs": 0})
    with pytest.raises(ValueError):
        RM3Config.from_value({"lambda": 0.5})


def test_rm3_hybrid_mode_runs(eng):
    assert eng.retrieve("slab strength", limit=3, mode="hybrid", expansion="rm3", use_cache=False)


# ---------------------------------------------------------------- CLI / MCP

def test_cli_chunking_expand_parent(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    store = tmp_path / ".locus"
    subprocess.run([sys.executable, "-m", "locus", "--store", str(store), "index", str(DOCS),
                    "--chunk-words", "60", "--overlap-words", "10", "--heading-context"],
                   check=True, capture_output=True, env=env)
    assert LocusEngine(store_path=store).corpus.chunking == {
        "chunk_words": 60, "overlap_words": 10, "heading_context": True, "boundaries": "words"}
    out = subprocess.run([sys.executable, "-m", "locus", "--store", str(store), "retrieve", "slab strength",
                          "--raw", "--limit", "2", "--parent", "section", "--expand", "rm3"],
                         check=True, capture_output=True, text=True, env=env).stdout
    rows = json.loads(out)
    assert len(rows) == 2 and all(r["spans"] and r["provenance"] == "bm25+rm3+parent" for r in rows)


def test_mcp_parent_and_expansion(small):
    out = _call_tool("locus_retrieve", {"query": "slab strength", "limit": 2, "track": False,
                                        "parent": "doc", "expansion": "rm3"}, engine=small)
    assert len(out["results"]) == 2 and all(r["spans"] for r in out["results"])
