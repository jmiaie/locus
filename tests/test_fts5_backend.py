"""FTS5 index backend, legacy fallback/migration, incremental updates,
read-only + deterministic retrieval, and the bm25/hybrid pipeline split."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import locus.memory.corpus as corpus_mod
from locus.core import LocusEngine, RETRIEVAL_MODES
from locus.memory.corpus import Corpus, fts5_available

needs_fts5 = pytest.mark.skipif(not fts5_available(), reason="SQLite built without FTS5")

DOCS = {
    "auth.md": "---\ndate: 2025-01-15\ntags: security\ntype: guide\n---\n# Authentication\n\n"
               "JWT tokens are used for authentication. Alice leads the Auth team. See [[ops]].",
    "ops.md": "# Ops\n\nOperational runbook for the authentication service and token rotation.",
    "deploy.md": "---\ndate: 2023-03-01\ntype: runbook\n---\n# Deployment\n\n"
                 "Kubernetes manages container orchestration. Bob owns the DeployPipeline.",
    "db.md": "# Database\n\nPostgreSQL schema migrations and index tuning. Carol leads the database team.",
    "cooking.md": "# Cooking\n\nPasta, risotto and garlic. Nothing about tokens or kubernetes here.",
}


def _write(root: Path, docs: dict = DOCS) -> Path:
    d = root / "docs"
    d.mkdir(exist_ok=True)
    for name, text in docs.items():
        (d / name).write_text(text)
    return d


def _engine(tmp_path, **kw) -> LocusEngine:
    e = LocusEngine(store_path=tmp_path / ".locus", **kw)
    e.index(_write(tmp_path))
    return e


def _store_digest(store: Path) -> dict:
    out = {}
    for f in sorted(store.rglob("*.sqlite3")):
        conn = sqlite3.connect(f)
        dump = "\n".join(conn.iterdump())
        conn.close()
        out[f.name] = hashlib.sha256(dump.encode()).hexdigest()
    return out


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

@needs_fts5
def test_fts5_is_default_backend(tmp_path):
    e = _engine(tmp_path)
    assert e.corpus.backend == "fts5"
    assert e.corpus.stats()["index_backend"] == "fts5"
    with sqlite3.connect(e.corpus.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM term_index").fetchone()[0] == 0
        assert conn.execute("SELECT value FROM meta WHERE key='index_backend'").fetchone()[0] == "fts5"


def test_fallback_without_fts5(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_mod, "_FTS5_AVAILABLE", False)
    e = _engine(tmp_path)
    assert e.corpus.backend == "legacy"
    hits = e.retrieve("JWT authentication", limit=3)
    assert hits and hits[0].doc_path == "auth.md"


def test_unknown_backend_rejected(tmp_path):
    with pytest.raises(ValueError):
        Corpus(tmp_path / "c", backend="lucene")


@needs_fts5
def test_legacy_and_fts5_rank_identically(tmp_path):
    fts = _engine(tmp_path / "a", index_backend="fts5")
    leg = _engine(tmp_path / "b", index_backend="legacy")
    assert leg.corpus.backend == "legacy"
    for q in ["authentication tokens", "kubernetes container", "database schema index",
              "garlic pasta", "team leads"]:
        a = fts.retrieve(q, limit=5)
        b = leg.retrieve(q, limit=5)
        assert [c.chunk_id for c in a] == [c.chunk_id for c in b], q
        for x, y in zip(a, b):
            assert x.score == pytest.approx(y.score, rel=1e-6, abs=1e-9)


@needs_fts5
def test_migrates_pre_fts5_store(tmp_path):
    store = tmp_path / ".locus"
    old = LocusEngine(store_path=store, index_backend="legacy")
    old.index(_write(tmp_path))
    expected = [c.chunk_id for c in old.retrieve("authentication tokens", limit=5)]
    # Make it look exactly like a Locus <= 1.0.0 store: no meta table, no n_tokens values.
    with sqlite3.connect(old.corpus.db_path) as conn:
        conn.execute("DROP TABLE meta")
        conn.execute("UPDATE chunks SET n_tokens = NULL")
        assert conn.execute("SELECT COUNT(*) FROM term_index").fetchone()[0] > 0
    new = LocusEngine(store_path=store)  # default backend -> fts5, migrates in place
    assert new.corpus.backend == "fts5"
    assert [c.chunk_id for c in new.retrieve("authentication tokens", limit=5)] == expected
    with sqlite3.connect(new.corpus.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM term_index").fetchone()[0] == 0


@needs_fts5
def test_reindex_switches_backend_and_back(tmp_path):
    e = _engine(tmp_path)
    before = [c.chunk_id for c in e.retrieve("kubernetes", limit=3)]
    r = e.reindex(backend="legacy")
    assert r["backend"] == "legacy" and r["chunks"] == e.corpus.chunk_count()
    assert [c.chunk_id for c in e.retrieve("kubernetes", limit=3)] == before
    e.reindex(backend="fts5")
    assert [c.chunk_id for c in e.retrieve("kubernetes", limit=3)] == before


# ---------------------------------------------------------------------------
# Incremental updates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["fts5", "legacy"])
def test_incremental_update_replaces_postings(tmp_path, backend):
    if backend == "fts5" and not fts5_available():
        pytest.skip("no FTS5")
    e = _engine(tmp_path, index_backend=backend)
    docs = tmp_path / "docs"
    assert e.index(docs)["chunks"] == 0  # unchanged -> skipped
    (docs / "cooking.md").write_text("# Cooking\n\nSourdough fermentation and croissant lamination.")
    assert e.index(docs)["chunks"] >= 1
    assert e.retrieve("sourdough", limit=3)[0].doc_path == "cooking.md"
    assert all(c.doc_path != "cooking.md" for c in e.retrieve("risotto garlic", limit=5))
    e.forget("cooking.md")
    assert e.retrieve("sourdough", limit=3) == []
    assert "cooking.md" not in e.corpus.list_docs()


def test_reindex_does_not_duplicate_kg_and_keeps_manual_facts(tmp_path):
    e = _engine(tmp_path)
    e.add_fact("Alice", "role", "Engineer", source="auth.md")
    n1 = e.kg.stats()["triple_count"]
    docs = tmp_path / "docs"
    e.index(docs)  # unchanged
    assert e.kg.stats()["triple_count"] == n1
    (docs / "auth.md").write_text(DOCS["auth.md"] + " Dave reviews the Auth team.")
    e.index(docs)
    facts = e.query_entity("Alice")["facts"]
    assert any(f["predicate"] == "role" for f in facts)  # manual fact kept
    # extracted triples for auth.md were replaced, not appended
    mentioned = [f for f in e.kg_match("ops", "mentioned_in", "auth.md")["triples"]]
    assert len(mentioned) == 1
    e.forget("auth.md")
    assert e.kg_match("ops", "mentioned_in", "auth.md")["count"] == 0
    assert any(f["predicate"] == "role" for f in e.query_entity("Alice")["facts"])


# ---------------------------------------------------------------------------
# Read-only + deterministic retrieval
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
def test_retrieve_does_not_write(tmp_path, mode):
    e = _engine(tmp_path)
    store = tmp_path / ".locus"
    before = _store_digest(store)
    tracked = e.bulletin.stats()["total_tracked"]
    for q in ["authentication tokens", "Q1 2025 runbook", "who leads the database team"]:
        e.retrieve(q, limit=5, mode=mode, use_cache=False)
        list(e.stream_retrieve(q, limit=5))
        e.prepare_context(q, limit=3, mode=mode)
    assert _store_digest(store) == before
    assert e.bulletin.stats()["total_tracked"] == tracked


def test_track_true_records_hits(tmp_path):
    e = _engine(tmp_path)
    e.retrieve("authentication tokens", limit=3, track=True)
    assert e.bulletin.stats()["total_tracked"] >= 1


def test_ties_broken_by_index_order(tmp_path):
    same = "Identical text about tokens and authentication flows."
    e = LocusEngine(store_path=tmp_path / ".locus")
    e.index(_write(tmp_path, {f"{n}.md": same for n in ["c", "a", "b"]}))
    hits = e.retrieve("tokens", limit=3)
    assert [h.doc_path for h in hits] == ["a.md", "b.md", "c.md"]
    assert len({h.score for h in hits}) == 1


def test_two_stores_same_files_same_results(tmp_path):
    a = _engine(tmp_path / "x")
    b = _engine(tmp_path / "y")
    for mode in RETRIEVAL_MODES:
        for q in ["authentication runbook", "guide for security 2025", "team"]:
            ra = [(c.chunk_id, round(c.score, 9), c.provenance) for c in a.retrieve(q, mode=mode)]
            rb = [(c.chunk_id, round(c.score, 9), c.provenance) for c in b.retrieve(q, mode=mode)]
            assert ra == rb


_HASHSEED_SCRIPT = r"""
import json, sys
from pathlib import Path
from locus.core import LocusEngine
store, docs = sys.argv[1], sys.argv[2]
e = LocusEngine(store_path=store)
e.index(docs)
out = {}
for q in ["incident runbook review", "meeting decision report 2025", "guide runbook spec", "Alice Bob team"]:
    out[q] = [(c.chunk_id, c.provenance) for c in e.retrieve(q, limit=5, mode="hybrid")]
print(json.dumps(out))
"""


def test_hybrid_independent_of_hash_seed(tmp_path):
    docs = _write(tmp_path)
    outs = set()
    for i, seed in enumerate(["1", "2", "12345"]):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run(
            [sys.executable, "-c", _HASHSEED_SCRIPT, str(tmp_path / f"s{i}"), str(docs)],
            capture_output=True, text=True, env=env, check=True,
        )
        outs.add(r.stdout.strip())
    assert len(outs) == 1


# ---------------------------------------------------------------------------
# Pipeline: bm25 default, hybrid re-rank is opt-in
# ---------------------------------------------------------------------------

def test_default_mode_is_bm25(tmp_path):
    e = _engine(tmp_path)
    assert e.retrieval_mode == "bm25"
    hits = e.retrieve("authentication", limit=5)
    assert hits and all(h.provenance == "bm25" for h in hits)
    assert hits == e._bm25.search("authentication", limit=5)


def test_hybrid_only_reranks_bm25_pool(tmp_path):
    e = _engine(tmp_path)
    for q in ["authentication runbook", "Alice team 2025", "kubernetes"]:
        pool = {c.chunk_id for c in e._bm25.search(q, limit=e.rerank_depth)}
        hyb = e.retrieve(q, limit=5, mode="hybrid")
        assert {c.chunk_id for c in hyb} <= pool


def test_as_of_selects_hybrid_and_mode_validation(tmp_path):
    e = _engine(tmp_path)
    assert isinstance(e.retrieve("Alice", as_of="2025-01-01"), list)
    with pytest.raises(ValueError):
        e.retrieve("Alice", mode="vector")
    with pytest.raises(ValueError):
        LocusEngine(store_path=tmp_path / "z", retrieval_mode="dense")


def test_env_var_selects_hybrid(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCUS_RETRIEVAL_MODE", "hybrid")
    assert LocusEngine(store_path=tmp_path / ".l").retrieval_mode == "hybrid"


def test_mcp_retrieve_mode_and_track(tmp_path):
    from locus.mcp.server import _call_tool
    _write(tmp_path)
    store = str(tmp_path / ".locus")
    _call_tool("locus_index", {"path": str(tmp_path / "docs"), "store_path": store})
    r = _call_tool("locus_retrieve", {"query": "authentication", "store_path": store,
                                      "mode": "hybrid", "track": False})
    if isinstance(r, str):
        r = json.loads(r)
    assert r["results"]
