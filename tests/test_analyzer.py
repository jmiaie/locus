"""Analyzer (stemming / stop-words) tests: determinism, FTS5 parity, migration."""
import sqlite3

import pytest

from locus.core import LocusEngine
from locus.memory.corpus import ANALYZERS, DEFAULT_ANALYZER, Corpus, analyze, fts5_available
from locus.memory.stemmer import stem

# Classic Porter (1980) examples, plus the author's later bli/logi revisions.
PORTER_CASES = {
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
    "feed": "feed", "agreed": "agre", "plastered": "plaster", "bled": "bled",
    "motoring": "motor", "sing": "sing", "conflated": "conflat", "troubled": "troubl",
    "sized": "size", "hopping": "hop", "tanned": "tan", "falling": "fall", "hissing": "hiss",
    "fizzed": "fizz", "failing": "fail", "filing": "file", "happy": "happi", "sky": "sky",
    "relational": "relat", "conditional": "condit", "rational": "ration",
    "valenci": "valenc", "digitizer": "digit", "conformabli": "conform",
    "radicalli": "radic", "differentli": "differ", "vileli": "vile", "analogousli": "analog",
    "vietnamization": "vietnam", "predication": "predic", "operator": "oper",
    "feudalism": "feudal", "decisiveness": "decis", "hopefulness": "hope",
    "callousness": "callous", "formaliti": "formal", "sensitiviti": "sensit",
    "sensibiliti": "sensibl", "triplicate": "triplic", "formative": "form",
    "formalize": "formal", "electriciti": "electr", "electrical": "electr",
    "hopeful": "hope", "goodness": "good", "revival": "reviv", "allowance": "allow",
    "inference": "infer", "airliner": "airlin", "gyroscopic": "gyroscop",
    "adjustable": "adjust", "defensible": "defens", "irritant": "irrit",
    "replacement": "replac", "adjustment": "adjust", "dependent": "depend",
    "adoption": "adopt", "homologou": "homolog", "communism": "commun",
    "activate": "activ", "angulariti": "angular", "homologous": "homolog",
    "effective": "effect", "bowdlerize": "bowdler", "probate": "probat",
    "rate": "rate", "cease": "ceas", "controll": "control", "roll": "roll",
    "etiology": "etiolog", "incredibly": "incred",
}


@pytest.mark.parametrize("word,expected", sorted(PORTER_CASES.items()))
def test_porter_reference_cases(word, expected):
    assert stem(word) == expected


def test_stem_keeps_non_alpha_tokens():
    assert stem("covid19") == "covid19"
    assert stem("snake_case") == "snake_case"
    assert stem("café") == "café"


@pytest.mark.skipif(not fts5_available(), reason="SQLite without FTS5")
def test_stemmer_matches_sqlite_fts5_porter():
    words = sorted(set(PORTER_CASES) | {
        "running", "runs", "ran", "generalizations", "oscillators", "proteins", "mutations",
        "inflammatory", "cardiovascular", "hypertension", "dietary", "vitamins", "children",
        "relationships", "conditionally", "connected", "connecting", "connection", "argued",
        "analysis", "analyses", "studies", "studying", "measurement", "probabilities",
    })
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE f USING fts5(b, tokenize='porter ascii')")
    con.execute("CREATE VIRTUAL TABLE v USING fts5vocab(f, instance)")
    con.executemany("INSERT INTO f(rowid, b) VALUES (?, ?)", [(i + 1, w) for i, w in enumerate(words)])
    fts = {doc: term for term, doc in con.execute("SELECT term, doc FROM v")}
    assert [stem(w) for w in words] == [fts[i + 1] for i in range(len(words))]


def test_analyzers():
    text = "The children were running studies on their proteins"
    assert analyze(text, "plain") == ["children", "running", "studies", "their", "proteins"]
    assert analyze(text, "porter") == ["children", "run", "studi", "their", "protein"]
    assert analyze(text, "porter+english") == ["children", "run", "studi", "protein"]
    with pytest.raises(ValueError):
        analyze(text, "snowball")
    assert DEFAULT_ANALYZER in ANALYZERS


def _docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# Running\n\nShe runs every morning; running improves endurance.")
    (d / "b.md").write_text("# Cooking\n\nRecipes for baking bread and cakes.")
    (d / "c.md").write_text("# Runner\n\nA runner's guide to marathons.")
    return d


@pytest.mark.parametrize("backend", ["fts5", "legacy"])
def test_stemmed_query_matches_inflections(tmp_path, backend):
    if backend == "fts5" and not fts5_available():
        pytest.skip("SQLite without FTS5")
    eng = LocusEngine(store_path=tmp_path / ".locus", index_backend=backend, analyzer="porter")
    eng.index(_docs(tmp_path))
    hits = eng.retrieve("run", limit=3, use_cache=False)
    assert hits and hits[0].doc_path.endswith("a.md")
    plain = LocusEngine(store_path=tmp_path / ".plain", index_backend=backend, analyzer="plain")
    plain.index(tmp_path / "docs")
    assert plain.retrieve("run", limit=3, use_cache=False) == []


def test_backends_rank_identically_with_stemming(tmp_path):
    if not fts5_available():
        pytest.skip("SQLite without FTS5")
    d = _docs(tmp_path)
    a = LocusEngine(store_path=tmp_path / ".f", index_backend="fts5", analyzer="porter")
    b = LocusEngine(store_path=tmp_path / ".l", index_backend="legacy", analyzer="porter")
    a.index(d)
    b.index(d)
    for q in ["running endurance", "baking", "marathon runners"]:
        ra = [c.chunk_id for c in a.retrieve(q, limit=3, use_cache=False, mode="bm25")]
        rb = [c.chunk_id for c in b.retrieve(q, limit=3, use_cache=False, mode="bm25")]
        assert ra == rb


def test_deterministic_across_runs(tmp_path):
    d = _docs(tmp_path)
    runs = []
    for i in range(2):
        eng = LocusEngine(store_path=tmp_path / f".s{i}")
        eng.index(d)
        runs.append([(c.chunk_id, round(c.score, 9)) for c in eng.retrieve("runs", limit=3, use_cache=False)])
    assert runs[0] == runs[1]


def test_analyzer_switch_rebuilds_and_persists(tmp_path):
    d = _docs(tmp_path)
    store = tmp_path / ".locus"
    eng = LocusEngine(store_path=store, analyzer="plain")
    eng.index(d)
    assert eng.corpus.analyzer == "plain"
    assert eng.retrieve("run", limit=3, use_cache=False) == []
    # Re-open without a request: keeps the store's analyzer.
    assert LocusEngine(store_path=store).corpus.analyzer == "plain"
    # Explicit request re-indexes from stored chunks (no file reads).
    eng2 = LocusEngine(store_path=store, analyzer="porter")
    assert eng2.corpus.analyzer == "porter"
    assert eng2.retrieve("run", limit=3, use_cache=False)
    assert LocusEngine(store_path=store).corpus.stats()["analyzer"] == "porter"
    # And via reindex().
    r = eng2.reindex(analyzer="plain")
    assert r["analyzer"] == "plain"
    assert LocusEngine(store_path=store).corpus.analyzer == "plain"


def test_pre_analyzer_store_is_upgraded_to_default(tmp_path):
    d = _docs(tmp_path)
    store = tmp_path / "corpus"
    c = Corpus(store, analyzer="plain")
    c.add_directory(d)
    with sqlite3.connect(c.db_path) as conn:  # simulate a store written before analyzers
        conn.execute("DELETE FROM meta WHERE key = 'analyzer'")
    assert Corpus(store).analyzer == DEFAULT_ANALYZER


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCUS_ANALYZER", "porter+english")
    assert Corpus(tmp_path / "c").analyzer == "porter+english"
    monkeypatch.setenv("LOCUS_ANALYZER", "bogus")
    assert Corpus(tmp_path / "c2").analyzer == DEFAULT_ANALYZER


def test_public_term_apis_accept_plain_words(tmp_path):
    c = Corpus(tmp_path / "c", analyzer="porter")
    c.add_directory(_docs(tmp_path))
    assert c.get_posting_list("running") == c.get_posting_list("runs")
    assert c.term_chunk_frequency("running") >= 1
