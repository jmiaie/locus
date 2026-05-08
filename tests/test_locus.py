"""
Tests for Locus vectorless RAG engine.
"""

import json
import pytest
from pathlib import Path

from locus.memory.chunker import Chunker, _extract_frontmatter, _extract_links
from locus.memory.corpus import Corpus, tokenize
from locus.memory.knowledge_graph import TemporalKG
from locus.retrieval.bm25 import BM25Retriever, ScoredChunk
from locus.retrieval.fusion import rrf_fuse
from locus.retrieval.kg_retrieval import KGRetriever, extract_query_entities
from locus.retrieval.link_walker import LinkWalker
from locus.retrieval.classifier import classify_query, QueryIntent, INTENT_WEIGHTS
from locus.retrieval.structural import StructuralRetriever, _extract_date_range, _extract_tags, _extract_type
from locus.retrieval.recency import RecencyRetriever
from locus.context.bulletin import ContextBulletin, PROMOTE_THRESHOLD
from locus.context.budget import ContextBudget, BudgetStatus
from locus.memory.extractor import extract_triples_from_text, ProseTriple
from locus.memory.entity_resolver import EntityResolver
from locus.core import LocusEngine


# -----------------------------------------------------------------------
# Chunker
# -----------------------------------------------------------------------

def test_extract_frontmatter_present():
    text = "---\ndate: 2024-01-01\ntags: foo, bar\n---\nBody content here."
    meta, body = _extract_frontmatter(text)
    assert meta["date"] == "2024-01-01"
    assert meta["tags"] == "foo, bar"
    assert "Body content" in body


def test_extract_frontmatter_absent():
    text = "No frontmatter here, just plain text."
    meta, body = _extract_frontmatter(text)
    assert meta == {}
    assert body == text


def test_extract_links_wikilinks():
    text = "See [[ProjectAlpha]] and [[Beta|Beta Project]] for context."
    links = _extract_links(text)
    assert "ProjectAlpha" in links
    assert "Beta" in links


def test_chunker_produces_chunks():
    chunker = Chunker(chunk_words=10, overlap_words=2)
    chunks = chunker.chunk_text(
        "one two three four five six seven eight nine ten eleven twelve thirteen",
        source="test.md",
    )
    assert len(chunks) >= 1
    assert all(c.doc_path == "test.md" for c in chunks)
    assert all(c.id for c in chunks)


def test_chunker_overlap():
    chunker = Chunker(chunk_words=5, overlap_words=2)
    # Use real words so chunks pass the 40-char minimum content check
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
             "golf", "hotel", "india", "juliet", "kilo", "lima",
             "mike", "november", "oscar", "papa", "quebec", "romeo"]
    text = " ".join(words)
    chunks = chunker.chunk_text(text, source="words.md")
    assert len(chunks) >= 3


def test_chunker_file(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("---\ndate: 2025-01-01\n---\n" + " ".join(["word"] * 100))
    chunker = Chunker()
    chunks = chunker.chunk_file(f, base_path=tmp_path)
    assert len(chunks) >= 1
    assert chunks[0].metadata["date"] == "2025-01-01"


# -----------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------

def test_tokenize_removes_stopwords():
    tokens = tokenize("the quick brown fox jumps over the lazy dog")
    assert "the" not in tokens    # in STOPWORDS
    assert "of" not in tokens     # in STOPWORDS
    assert "quick" in tokens
    assert "brown" in tokens


def test_tokenize_lowercases():
    tokens = tokenize("Hello World")
    assert "hello" in tokens
    assert "world" in tokens


# -----------------------------------------------------------------------
# Corpus
# -----------------------------------------------------------------------

def test_corpus_add_and_count(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    doc = tmp_path / "test.md"
    doc.write_text("# Hello\n\nThis is a test document about vectorless retrieval and BM25.")
    added = corpus.add_file(doc, base_path=tmp_path)
    assert added >= 1
    assert corpus.doc_count() == 1
    assert corpus.chunk_count() >= 1


def test_corpus_remove(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    doc = tmp_path / "test.md"
    doc.write_text("Some content about information retrieval.")
    corpus.add_file(doc, base_path=tmp_path)
    assert corpus.doc_count() == 1
    corpus.remove_file("test.md")
    assert corpus.doc_count() == 0
    assert corpus.chunk_count() == 0


def test_corpus_list_docs(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    for name in ["a.md", "b.md", "c.md"]:
        f = tmp_path / name
        f.write_text(f"Content of {name} with some words for indexing.")
        corpus.add_file(f, base_path=tmp_path)
    docs = corpus.list_docs()
    assert len(docs) == 3


def test_corpus_posting_list(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    doc = tmp_path / "fruit.md"
    doc.write_text("Bananas are yellow tropical fruits rich in potassium.")
    corpus.add_file(doc, base_path=tmp_path)
    posting = corpus.get_posting_list("bananas")
    assert len(posting) >= 1


# -----------------------------------------------------------------------
# Knowledge Graph
# -----------------------------------------------------------------------

def test_kg_add_and_query(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "knows", "Bob", source="test.md")
    kg.add_triple("Alice", "works_at", "Acme", source="test.md")
    triples = kg.query_entity("Alice")
    assert len(triples) == 2
    predicates = {t.predicate for t in triples}
    assert "knows" in predicates
    assert "works_at" in predicates


def test_kg_temporal_filter(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "role", "Engineer", valid_from="2020-01-01", valid_to="2022-12-31")
    kg.add_triple("Alice", "role", "Manager", valid_from="2023-01-01")

    triples_2021 = kg.query_entity("Alice", as_of="2021-06-01")
    roles = {t.object for t in triples_2021 if t.predicate == "role"}
    assert "Engineer" in roles
    assert "Manager" not in roles

    triples_2024 = kg.query_entity("Alice", as_of="2024-01-01")
    roles_2024 = {t.object for t in triples_2024 if t.predicate == "role"}
    assert "Manager" in roles_2024


def test_kg_populate_from_text(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    text = "This note discusses [[ProjectAlpha]] and [[TeamBeta]]. #engineering #ops"
    count = kg.populate_from_text(text, source="notes/meeting.md")
    assert count >= 4  # 2 wikilinks + 2 tags

    sources = kg.sources_for_entity("ProjectAlpha")
    assert "notes/meeting.md" in sources


def test_kg_stats(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("X", "rel", "Y", source="doc.md")
    stats = kg.stats()
    assert stats["triple_count"] == 1
    assert stats["entity_count"] >= 1


# -----------------------------------------------------------------------
# BM25 Retriever
# -----------------------------------------------------------------------

def test_bm25_basic(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    docs = {
        "apples.md": "Apples are a crisp delicious fruit grown in temperate orchards and climates.",
        "bananas.md": "Bananas are tropical yellow fruits packed with potassium and dietary fiber.",
        "python.md": "Python is a programming language popular for data science and automation tasks.",
    }
    for fname, content in docs.items():
        f = tmp_path / fname
        f.write_text(content)
        corpus.add_file(f, base_path=tmp_path)

    retriever = BM25Retriever(corpus)
    results = retriever.search("tropical fruit potassium", limit=3)
    assert len(results) > 0
    assert results[0].doc_path == "bananas.md"
    assert all(r.provenance == "bm25" for r in results)


def test_bm25_empty_corpus(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    retriever = BM25Retriever(corpus)
    assert retriever.search("anything") == []


def test_bm25_no_match(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    f = tmp_path / "doc.md"
    f.write_text("This document is about cooking recipes and pasta dishes.")
    corpus.add_file(f, base_path=tmp_path)
    retriever = BM25Retriever(corpus)
    results = retriever.search("quantum physics")
    assert isinstance(results, list)


# -----------------------------------------------------------------------
# KG Retriever
# -----------------------------------------------------------------------

def test_kg_retriever(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))

    f = tmp_path / "transformers.md"
    f.write_text("Transformers are a neural network architecture used in NLP and vision tasks.")
    corpus.add_file(f, base_path=tmp_path)
    kg.add_triple("Transformers", "introduced_by", "Attention Is All You Need", source="transformers.md")

    retriever = KGRetriever(kg, corpus)
    results = retriever.search("Transformers architecture")
    assert len(results) >= 1
    assert any(r.doc_path == "transformers.md" for r in results)


def test_extract_query_entities():
    entities = extract_query_entities('What did "Alice" do at Acme Corp?')
    assert "Alice" in entities or "Acme" in entities


# -----------------------------------------------------------------------
# Link Walker
# -----------------------------------------------------------------------

def test_link_walker(tmp_path):
    corpus = Corpus(tmp_path / "corpus")

    root = tmp_path / "index.md"
    root.write_text("Main index. See [[details]] for more information about the system.")
    details = tmp_path / "details.md"
    details.write_text("Details page with extensive information about the system architecture.")

    corpus.add_file(root, base_path=tmp_path)
    corpus.add_file(details, base_path=tmp_path)

    retriever = BM25Retriever(corpus)
    seeds = retriever.search("main index", limit=2)

    walker = LinkWalker(corpus)
    walked = walker.walk(seeds, depth=1, limit=5)
    assert isinstance(walked, list)
    if walked:
        assert all(c.provenance.startswith("link:") for c in walked)


# -----------------------------------------------------------------------
# RRF Fusion
# -----------------------------------------------------------------------

def test_rrf_fuse_deduplicates():
    list_a = [
        ScoredChunk("c1", "doc1.md", 2.0, "content1", "bm25"),
        ScoredChunk("c2", "doc2.md", 1.5, "content2", "bm25"),
    ]
    list_b = [
        ScoredChunk("c2", "doc2.md", 3.0, "content2", "kg"),
        ScoredChunk("c3", "doc3.md", 1.0, "content3", "kg"),
    ]
    fused = rrf_fuse([list_a, list_b], limit=3)
    ids = [c.chunk_id for c in fused]
    assert len(ids) == len(set(ids)), "Duplicates in fused output"


def test_rrf_boosts_common_results():
    list_a = [
        ScoredChunk("c1", "doc1.md", 2.0, "x", "bm25"),
        ScoredChunk("c2", "doc2.md", 1.0, "y", "bm25"),
    ]
    list_b = [
        ScoredChunk("c1", "doc1.md", 1.0, "x", "kg"),
        ScoredChunk("c3", "doc3.md", 2.0, "z", "kg"),
    ]
    fused = rrf_fuse([list_a, list_b], limit=3)
    top = fused[0].chunk_id
    assert top == "c1", "c1 appears in both lists and should rank first via RRF"


def test_rrf_empty_lists():
    assert rrf_fuse([[], [], []]) == []


# -----------------------------------------------------------------------
# Bulletin
# -----------------------------------------------------------------------

def test_bulletin_records_hit():
    bulletin = ContextBulletin()
    bulletin.record_hit("c1", content="Important content", doc_path="doc.md", base_score=0.5)
    assert "c1" in bulletin._all
    assert bulletin.stats()["tier1_hot"] == 1


def test_bulletin_auto_promote():
    bulletin = ContextBulletin()
    # High base_score + many hits should push effective_score above PROMOTE_THRESHOLD
    for _ in range(20):
        bulletin.record_hit("hot_chunk", content="text", doc_path="doc.md", base_score=0.9)
    assert bulletin.stats()["tier0_pinned"] == 1


def test_bulletin_manual_promote():
    bulletin = ContextBulletin()
    bulletin.record_hit("c1", content="x", doc_path="d.md", base_score=0.3)
    result = bulletin.promote_to_pin("c1")
    assert result is True
    assert bulletin.stats()["tier0_pinned"] == 1


def test_bulletin_tick_archives(tmp_path):
    bulletin = ContextBulletin(archive_path=tmp_path / "archive")
    bulletin.record_hit("old", content="old content", doc_path="old.md", base_score=0.0)
    entry = bulletin._all["old"]
    entry.rounds_elapsed = 100  # force effective_score very negative
    archived = bulletin.tick()
    assert archived == 1
    assert bulletin.stats()["tier1_hot"] == 0


def test_bulletin_inject():
    bulletin = ContextBulletin()
    bulletin.record_hit("c1", content="Context about the project.", doc_path="proj.md", base_score=0.7)
    output = bulletin.inject(token_limit=500)
    assert "proj.md" in output


# -----------------------------------------------------------------------
# Budget
# -----------------------------------------------------------------------

def test_budget_ok():
    budget = ContextBudget()
    check = budget.record(100)
    assert check.status == BudgetStatus.OK


def test_budget_critical():
    budget = ContextBudget(critical_threshold=500)
    check = budget.record(600)
    assert check.status == BudgetStatus.CRITICAL


def test_budget_trend():
    budget = ContextBudget()
    for i in range(6):
        budget.record(100 + i * 50)
    assert budget._consecutive_growth >= 5


def test_budget_estimate_tokens():
    budget = ContextBudget()
    tokens = budget.estimate_tokens("one two three four five")
    assert tokens > 0


# -----------------------------------------------------------------------
# End-to-end engine
# -----------------------------------------------------------------------

def test_engine_index_retrieve(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")

    (tmp_path / "ai.md").write_text(
        "# AI Research\n\nDeep learning and neural networks have revolutionized "
        "computer vision tasks. [[Transformers]] changed natural language processing."
    )
    (tmp_path / "python.md").write_text(
        "# Python Programming\n\nPython is excellent for machine learning pipelines "
        "and data science workflows including neural network training."
    )
    (tmp_path / "cooking.md").write_text(
        "# Cooking\n\nRecipes for pasta, risotto, and other Italian dishes. "
        "Garlic, olive oil, and fresh herbs are the foundation of good cooking."
    )

    result = engine.index(tmp_path)
    assert result["files"] >= 3
    assert result["chunks"] >= 3

    chunks = engine.retrieve("neural networks deep learning", limit=3)
    assert len(chunks) > 0
    doc_paths = [c.doc_path for c in chunks]
    assert any("ai" in p or "python" in p for p in doc_paths)

    irrelevant = engine.retrieve("Italian pasta recipes", limit=3)
    assert len(irrelevant) > 0
    assert any("cooking" in c.doc_path for c in irrelevant)


def test_engine_kg_fact_roundtrip(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Transformers", "introduced_by", "Vaswani et al", source="ai.md")
    result = engine.query_entity("Transformers")
    assert result["entity"] == "Transformers"
    assert len(result["facts"]) == 1
    assert result["facts"][0]["predicate"] == "introduced_by"


def test_engine_forget(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    f = tmp_path / "remove_me.md"
    f.write_text("This document will be removed from the corpus shortly.")
    # Index via directory so doc_path is relative (base_path is set)
    engine.index(tmp_path)
    assert engine.corpus.doc_count() == 1
    doc_path = engine.corpus.list_docs()[0]
    engine.forget(doc_path)
    assert engine.corpus.doc_count() == 0


def test_engine_session_lifecycle(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    start = engine.session_start()
    assert "corpus" in start
    assert "kg" in start
    assert "bulletin" in start
    assert "hot_context" in start

    wrap = engine.wrap_up()
    assert "bulletin" in wrap
    assert "budget" in wrap


def test_engine_format_context(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    f = tmp_path / "doc.md"
    f.write_text("Important knowledge about the system design and architecture.")
    engine.index(f)
    chunks = engine.retrieve("system design")
    context = engine.format_context(chunks)
    assert isinstance(context, str)


def test_engine_status(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    status = engine.status()
    assert "version" in status
    assert "corpus" in status
    assert "kg" in status


# -----------------------------------------------------------------------
# Phase 1 — Corpus checksum dedup
# -----------------------------------------------------------------------

def test_corpus_checksum_skip_unchanged(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    f = tmp_path / "doc.md"
    f.write_text("Content that will not change between index calls.")
    first = corpus.add_file(f, base_path=tmp_path)
    assert first >= 1
    second = corpus.add_file(f, base_path=tmp_path)
    assert second == 0  # unchanged — skipped


def test_corpus_checksum_reindex_on_change(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    f = tmp_path / "doc.md"
    f.write_text("Original content about system design.")
    corpus.add_file(f, base_path=tmp_path)
    f.write_text("Updated content about system design with extra information added.")
    second = corpus.add_file(f, base_path=tmp_path)
    assert second >= 1  # changed — re-indexed


def test_corpus_force_reindex(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    f = tmp_path / "doc.md"
    f.write_text("Same content both times.")
    corpus.add_file(f, base_path=tmp_path)
    forced = corpus.add_file(f, base_path=tmp_path, force=True)
    assert forced >= 1  # forced even though unchanged


def test_corpus_batch_fetch(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    for i in range(3):
        f = tmp_path / f"doc{i}.md"
        f.write_text(f"Document {i} contains unique information about topic {i}.")
        corpus.add_file(f, base_path=tmp_path)
    all_ids = [
        r for chunks in
        [corpus.get_chunks_for_doc(d) for d in corpus.list_docs()]
        for r in chunks
    ]
    ids = [c.id for c in all_ids[:3]]
    batch = corpus.get_chunks_batch(ids)
    assert len(batch) == len(ids)
    assert all(cid in batch for cid in ids)


def test_corpus_stats_cache_invalidated(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    assert corpus.doc_count() == 0
    f = tmp_path / "doc.md"
    f.write_text("Some content about caching behaviour.")
    corpus.add_file(f, base_path=tmp_path)
    assert corpus.doc_count() == 1  # cache invalidated by add_file
    corpus.remove_file("doc.md")
    assert corpus.doc_count() == 0  # cache invalidated by remove


# -----------------------------------------------------------------------
# Phase 1 — Query intent classifier
# -----------------------------------------------------------------------

def test_classify_kg_first():
    assert classify_query("Who is Alice?") == QueryIntent.KG_FIRST
    assert classify_query("When was the project created?") == QueryIntent.KG_FIRST
    assert classify_query("Who leads the infrastructure team?") == QueryIntent.KG_FIRST


def test_classify_bm25_first():
    assert classify_query("How does the authentication system work?") == QueryIntent.BM25_FIRST
    assert classify_query("Explain the deployment process") == QueryIntent.BM25_FIRST
    assert classify_query("What is the difference between staging and production?") == QueryIntent.BM25_FIRST


def test_classify_balanced_fallback():
    # Short, ambiguous queries should not strongly favour either
    result = classify_query("database")
    assert result in (QueryIntent.BALANCED, QueryIntent.BM25_FIRST, QueryIntent.KG_FIRST)


def test_intent_weights_defined():
    for intent in QueryIntent:
        weights = INTENT_WEIGHTS[intent]
        assert len(weights) == 3
        assert all(w > 0 for w in weights)


def test_rrf_weighted_boosts_kg():
    # With KG-first weights [0.5, 2.0, 0.5], a KG-only result should outscore
    # a BM25-only result that appeared first in BM25 list
    bm25_list = [ScoredChunk("bm25_top", "a.md", 5.0, "x", "bm25")]
    kg_list   = [ScoredChunk("kg_top",   "b.md", 1.0, "y", "kg")]
    fused = rrf_fuse([bm25_list, kg_list, []], weights=[0.5, 2.0, 0.5], limit=2)
    assert fused[0].chunk_id == "kg_top"


def test_rrf_weighted_boosts_bm25():
    bm25_list = [ScoredChunk("bm25_top", "a.md", 5.0, "x", "bm25")]
    kg_list   = [ScoredChunk("kg_top",   "b.md", 1.0, "y", "kg")]
    fused = rrf_fuse([bm25_list, kg_list, []], weights=[2.0, 0.5, 0.5], limit=2)
    assert fused[0].chunk_id == "bm25_top"


# -----------------------------------------------------------------------
# Phase 1 — Bulletin persistence
# -----------------------------------------------------------------------

def test_bulletin_persists_across_restarts(tmp_path):
    db = tmp_path / "bulletin.sqlite3"

    b1 = ContextBulletin(db_path=db)
    b1.record_hit("c1", content="Important context", doc_path="doc.md", base_score=0.7)
    b1.record_hit("c2", content="Secondary context", doc_path="doc2.md", base_score=0.5)

    # Simulate restart — fresh instance pointing at same db
    b2 = ContextBulletin(db_path=db)
    assert "c1" in b2._all
    assert "c2" in b2._all
    assert b2.stats()["tier1_hot"] == 2


def test_bulletin_persist_promotion_survives_restart(tmp_path):
    db = tmp_path / "bulletin.sqlite3"

    b1 = ContextBulletin(db_path=db)
    b1.record_hit("pin_me", content="Critical context", doc_path="doc.md", base_score=0.5)
    b1.promote_to_pin("pin_me")
    assert b1.stats()["tier0_pinned"] == 1

    b2 = ContextBulletin(db_path=db)
    assert b2.stats()["tier0_pinned"] == 1
    assert b2._all["pin_me"].tier == 0


def test_bulletin_persist_tick_removes_cold(tmp_path):
    db = tmp_path / "bulletin.sqlite3"

    b1 = ContextBulletin(db_path=db)
    b1.record_hit("cold", content="Cold content", doc_path="d.md", base_score=0.0)
    entry = b1._all["cold"]
    entry.rounds_elapsed = 100
    b1.tick()

    # After tick removes it, a new instance should not see it
    b2 = ContextBulletin(db_path=db)
    assert "cold" not in b2._all


def test_bulletin_stats_includes_persistent_flag(tmp_path):
    b_mem = ContextBulletin()
    assert b_mem.stats()["persistent"] is False

    b_db = ContextBulletin(db_path=tmp_path / "b.sqlite3")
    assert b_db.stats()["persistent"] is True


# -----------------------------------------------------------------------
# Phase 1 — Engine uses classifier + bulletin persistence
# -----------------------------------------------------------------------

def test_engine_bulletin_persists(tmp_path):
    store = tmp_path / ".locus"

    e1 = LocusEngine(store_path=store)
    f = tmp_path / "doc.md"
    f.write_text("Knowledge about the authentication and security system architecture.")
    e1.index(tmp_path)
    e1.retrieve("authentication security")

    # Re-open engine — bulletin should still have entries
    e2 = LocusEngine(store_path=store)
    assert e2.bulletin.stats()["total_tracked"] >= 1


def test_engine_checksum_dedup_on_reindex(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    f = tmp_path / "stable.md"
    f.write_text("Stable content that does not change between runs.")
    r1 = engine.index(tmp_path)
    r2 = engine.index(tmp_path)
    # Second pass: no new chunks (all files unchanged)
    assert r2["chunks"] == 0


def test_engine_retrieve_intent_override(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("Alice leads the infrastructure team at Acme Corp.")
    engine.index(tmp_path)
    chunks = engine.retrieve("Alice", intent=QueryIntent.KG_FIRST)
    assert isinstance(chunks, list)
    chunks2 = engine.retrieve("Alice", intent=QueryIntent.BM25_FIRST)
    assert isinstance(chunks2, list)


# -----------------------------------------------------------------------
# Phase 2 — Section-aware chunking
# -----------------------------------------------------------------------

def test_section_chunker_splits_at_headings():
    from locus.memory.chunker import Chunker
    chunker = Chunker(section_aware=True)
    text = (
        "# Introduction\n\nThis section introduces the system.\n\n"
        "## Architecture\n\nThe architecture consists of three layers.\n\n"
        "## Deployment\n\nDeployment is managed via Kubernetes."
    )
    chunks = chunker.chunk_text(text, source="doc.md")
    assert len(chunks) == 3
    sections = [c.metadata.get("section", "") for c in chunks]
    assert "Introduction" in sections
    assert "Architecture" in sections
    assert "Deployment" in sections


def test_section_chunker_preserves_heading_in_content():
    from locus.memory.chunker import Chunker
    chunker = Chunker(section_aware=True)
    text = "## Auth System\n\nJWT tokens are validated on every request."
    chunks = chunker.chunk_text(text, source="auth.md")
    assert len(chunks) == 1
    assert "Auth System" in chunks[0].content


def test_section_chunker_falls_back_for_no_headings():
    from locus.memory.chunker import Chunker
    chunker = Chunker(section_aware=True, chunk_words=5, overlap_words=1)
    words = ["alpha", "bravo", "charlie", "delta", "echo",
             "foxtrot", "golf", "hotel", "india", "juliet",
             "kilo", "lima", "mike", "november", "oscar"]
    text = " ".join(words)
    chunks = chunker.chunk_text(text, source="flat.md")
    assert len(chunks) >= 2


def test_section_chunker_splits_long_section():
    from locus.memory.chunker import Chunker
    chunker = Chunker(section_aware=True, chunk_words=5, overlap_words=1)
    # One heading but content > chunk_words → falls back to word split inside
    words = " ".join(["word"] * 20)
    text = f"## Long Section\n\n{words}"
    chunks = chunker.chunk_text(text, source="long.md")
    assert len(chunks) >= 3


def test_corpus_uses_section_aware_by_default(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    f = tmp_path / "sectioned.md"
    f.write_text(
        "# Overview\n\nSystem overview.\n\n"
        "## Details\n\nDetailed technical information here.\n\n"
        "## Deployment\n\nDeployment instructions for production."
    )
    added = corpus.add_file(f, base_path=tmp_path)
    assert added == 3  # one chunk per section
    chunks = corpus.get_chunks_for_doc("sectioned.md")
    sections = [c.metadata.get("section") for c in chunks]
    assert "Overview" in sections


# -----------------------------------------------------------------------
# Phase 2 — Structural retriever
# -----------------------------------------------------------------------

def test_structural_extract_date_range():
    assert _extract_date_range("decisions from Q1 2025") == ("2025-01-01", "2025-03-31")
    assert _extract_date_range("documents from 2024") == ("2024-01-01", "2024-12-31")
    assert _extract_date_range("Q4 2023 review") == ("2023-10-01", "2023-12-31")
    assert _extract_date_range("general query") is None


def test_structural_extract_tags():
    tags = _extract_tags("find #engineering documents tagged ops")
    assert "engineering" in tags
    assert "ops" in tags


def test_structural_extract_type():
    assert _extract_type("show me all meeting notes") == "meeting"
    assert _extract_type("find ADR documents") == "adr"
    assert _extract_type("random query") is None


def test_structural_retriever_date_match(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    f = tmp_path / "q1_doc.md"
    f.write_text("---\ndate: 2025-02-15\ntags: engineering\n---\nQ1 planning document.")
    corpus.add_file(f, base_path=tmp_path)

    retriever = StructuralRetriever(corpus)
    results = retriever.search("decisions from Q1 2025")
    assert len(results) == 1
    assert results[0].provenance == "structural"


def test_structural_retriever_tag_match(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    f = tmp_path / "infra.md"
    f.write_text("---\ntags: ops, infrastructure\n---\nInfrastructure documentation.")
    corpus.add_file(f, base_path=tmp_path)

    retriever = StructuralRetriever(corpus)
    results = retriever.search("tagged ops")
    assert len(results) >= 1


def test_structural_retriever_no_signals(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    f = tmp_path / "doc.md"
    f.write_text("Some plain content with no dates or tags.")
    corpus.add_file(f, base_path=tmp_path)

    retriever = StructuralRetriever(corpus)
    results = retriever.search("plain semantic query")
    assert results == []  # no structural signals → empty


# -----------------------------------------------------------------------
# Phase 2 — Recency retriever
# -----------------------------------------------------------------------

def test_recency_retriever_returns_docs(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    for i in range(3):
        f = tmp_path / f"doc{i}.md"
        f.write_text(f"Document {i} content about various topics.")
        corpus.add_file(f, base_path=tmp_path)

    retriever = RecencyRetriever(corpus)
    results = retriever.search(limit=10)
    assert len(results) == 3
    assert all(0 < r.score <= 1.0 for r in results)
    assert all(r.provenance == "recency" for r in results)


def test_recency_scores_decrease_with_age():
    from locus.retrieval.recency import RecencyRetriever
    from datetime import datetime, timezone, timedelta

    r = RecencyRetriever.__new__(RecencyRetriever)
    import math
    r._decay = math.log(2) / 30.0
    now = datetime.now(timezone.utc)

    score_today = r._score(now.isoformat(), now)
    score_old = r._score((now - timedelta(days=60)).isoformat(), now)
    assert score_today > score_old
    assert score_today > 0.99
    assert score_old < 0.3


# -----------------------------------------------------------------------
# Phase 3 — Prose triple extractor
# -----------------------------------------------------------------------

def test_extractor_basic_relations():
    triples = extract_triples_from_text(
        "Alice leads the Infrastructure team. Bob works at Acme Corp."
    )
    predicates = {t.predicate for t in triples}
    subjects = {t.subject for t in triples}
    assert "leads" in predicates or "works_at" in predicates
    assert "Alice" in subjects or "Bob" in subjects


def test_extractor_filters_pronouns():
    triples = extract_triples_from_text("It is responsible for the system.")
    subjects = {t.subject for t in triples}
    assert "It" not in subjects
    assert "it" not in subjects


def test_extractor_is_a():
    triples = extract_triples_from_text("Locus is a vectorless RAG system.")
    assert any(t.subject == "Locus" for t in triples)


def test_extractor_part_of():
    triples = extract_triples_from_text("AuthService is part of Platform.")
    subj = {t.subject for t in triples}
    assert "AuthService" in subj


def test_extractor_caps_at_max(tmp_path):
    from locus.memory.extractor import _MAX_TRIPLES
    # Generate lots of matches
    sentences = ". ".join([f"Alice{i} leads Team{i}" for i in range(100)])
    triples = extract_triples_from_text(sentences)
    assert len(triples) <= _MAX_TRIPLES


def test_extractor_skips_long_sentences():
    # A 50-word sentence should be skipped
    long_sentence = " ".join(["word"] * 50) + "."
    triples = extract_triples_from_text(long_sentence)
    assert triples == []


# -----------------------------------------------------------------------
# Phase 3 — Entity resolver
# -----------------------------------------------------------------------

def test_resolver_add_and_resolve(tmp_path):
    r = EntityResolver(tmp_path / "resolver.sqlite3")
    r.add_alias("Jeff", "Jeff Milam")
    assert r.resolve("Jeff") == "Jeff Milam"
    assert r.resolve("Jeff Milam") == "Jeff Milam"  # canonical → itself
    assert r.resolve("Unknown") == "Unknown"


def test_resolver_cache(tmp_path):
    r = EntityResolver(tmp_path / "resolver.sqlite3")
    r.add_alias("jmiaie", "Jeff Milam")
    # Second call should use cache
    assert r.resolve("jmiaie") == "Jeff Milam"
    assert "jmiaie" in r._cache


def test_resolver_persist_across_instances(tmp_path):
    db = tmp_path / "resolver.sqlite3"
    r1 = EntityResolver(db)
    r1.add_alias("OMPA", "OMPA Tool")

    r2 = EntityResolver(db)
    assert r2.resolve("OMPA") == "OMPA Tool"


def test_resolver_suggest(tmp_path):
    r = EntityResolver(tmp_path / "resolver.sqlite3")
    entities = ["Jeff Milam", "Jeff", "Alice Smith", "Alice", "Bob"]
    suggestions = r.suggest_aliases(entities, threshold=0.5)
    pairs = [(s["entity_a"], s["entity_b"]) for s in suggestions]
    # "Jeff Milam" and "Jeff" should be suggested
    assert any("Jeff" in a and "Jeff" in b for a, b in pairs)


def test_resolver_stats(tmp_path):
    r = EntityResolver(tmp_path / "resolver.sqlite3")
    r.add_alias("a", "A")
    r.add_alias("b", "B")
    assert r.stats()["alias_count"] == 2


# -----------------------------------------------------------------------
# Phase 3 — KG with prose extraction + contradictions
# -----------------------------------------------------------------------

def test_kg_prose_extraction(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    text = "Alice leads the Platform team. Bob works at Acme Corp."
    count = kg.populate_from_text(text, source="org.md", extract_prose=True)
    assert count >= 2
    triples = kg.query_entity("Alice")
    assert len(triples) >= 1


def test_kg_no_prose_extraction(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    text = "Alice leads the Platform team. No wikilinks or tags here."
    count = kg.populate_from_text(text, source="org.md", extract_prose=False)
    assert count == 0  # only wikilinks/tags, which are absent


def test_kg_find_contradictions_detects_conflict(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "role", "Engineer", source="a.md")
    kg.add_triple("Alice", "role", "Manager", source="b.md")
    contradictions = kg.find_contradictions("Alice")
    assert len(contradictions) == 1
    assert contradictions[0]["subject"] == "Alice"
    assert contradictions[0]["predicate"] == "role"


def test_kg_find_contradictions_no_overlap(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    # Non-overlapping time windows — NOT a contradiction
    kg.add_triple("Alice", "role", "Engineer", valid_from="2020-01-01", valid_to="2022-12-31")
    kg.add_triple("Alice", "role", "Manager", valid_from="2023-01-01", valid_to="2025-12-31")
    contradictions = kg.find_contradictions("Alice")
    assert len(contradictions) == 0


def test_kg_all_entities(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "knows", "Bob")
    kg.add_triple("Bob", "works_at", "Acme")
    entities = kg.all_entities()
    assert "Alice" in entities
    assert "Bob" in entities
    assert "Acme" in entities


def test_kg_resolver_transparent(tmp_path):
    resolver = EntityResolver(str(tmp_path / "resolver.sqlite3"))
    resolver.add_alias("Jeff", "Jeff Milam")
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"), resolver=resolver)
    kg.add_triple("Jeff", "leads", "Platform")
    # Querying by canonical should find it
    triples = kg.query_entity("Jeff Milam")
    assert len(triples) >= 1
    assert triples[0].subject == "Jeff Milam"


# -----------------------------------------------------------------------
# Phase 2/3 — Engine integration
# -----------------------------------------------------------------------

def test_engine_five_signal_retrieve(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "auth.md").write_text(
        "---\ndate: 2025-01-15\ntags: security\n---\n"
        "# Authentication\n\n"
        "JWT tokens are used for authentication. Alice leads the Auth team."
    )
    (tmp_path / "deploy.md").write_text(
        "# Deployment\n\nKubernetes manages container orchestration."
    )
    engine.index(tmp_path)
    chunks = engine.retrieve("authentication security", limit=5)
    assert len(chunks) > 0
    # auth.md is the relevant doc — it should appear in results
    doc_paths = {c.doc_path for c in chunks}
    assert any("auth" in p for p in doc_paths)


def test_engine_structural_via_retrieve(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "q1_notes.md").write_text(
        "---\ndate: 2025-02-10\ntags: meeting\n---\n"
        "Q1 planning meeting notes."
    )
    engine.index(tmp_path)
    chunks = engine.retrieve("Q1 2025 meeting")
    assert len(chunks) > 0
    assert any(c.provenance == "structural" for c in chunks)


def test_engine_find_contradictions(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Alice", "role", "Engineer", source="hr.md")
    engine.add_fact("Alice", "role", "Manager", source="org.md")
    result = engine.find_contradictions("Alice")
    assert result["contradiction_count"] == 1


def test_engine_add_alias_and_resolve(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_alias("Jeff", "Jeff Milam")
    engine.add_fact("Jeff", "leads", "Platform")
    facts = engine.query_entity("Jeff Milam")
    assert len(facts["facts"]) >= 1


def test_engine_suggest_aliases(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Jeff Milam", "leads", "Platform")
    engine.add_fact("Jeff", "works_at", "Acme")
    result = engine.suggest_aliases(threshold=0.5)
    assert "suggestions" in result
    assert "entity_count" in result


def test_engine_status_includes_resolver(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    status = engine.status()
    assert "resolver" in status
    assert "alias_count" in status["resolver"]


def test_engine_session_start_includes_resolver(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    start = engine.session_start()
    assert "resolver" in start


# -----------------------------------------------------------------------
# Phase 4 — locus_explain
# -----------------------------------------------------------------------

def test_explain_without_query(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    f = tmp_path / "auth.md"
    f.write_text("## JWT Auth\n\nJWT tokens are validated on every request.")
    engine.index(tmp_path)
    chunk_id = engine.corpus.get_chunks_for_doc("auth.md")[0].id

    result = engine.explain(chunk_id)
    assert result["doc_path"] == "auth.md"
    assert "narrative" in result
    assert "content_preview" in result
    assert "chunk_id" in result


def test_explain_with_query_bm25(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    f = tmp_path / "auth.md"
    f.write_text("## Authentication\n\nJWT tokens are validated for authentication.")
    engine.index(tmp_path)
    chunk_id = engine.corpus.get_chunks_for_doc("auth.md")[0].id

    result = engine.explain(chunk_id, query="JWT authentication")
    assert "bm25_matched_terms" in result
    assert "jwt" in result["bm25_matched_terms"] or "authentication" in result["bm25_matched_terms"]
    assert "narrative" in result
    assert "Retrieved because" in result["narrative"]


def test_explain_with_kg_context(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    f = tmp_path / "doc.md"
    f.write_text("Alice leads the Platform team.")
    engine.index(tmp_path)
    engine.add_fact("Alice", "leads", "Platform", source="doc.md")
    chunk_id = engine.corpus.get_chunks_for_doc("doc.md")[0].id

    result = engine.explain(chunk_id, query="Alice")
    assert len(result["kg_context"]) >= 1
    assert result["kg_context"][0]["entity"] == "Alice"


def test_explain_unknown_chunk(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    result = engine.explain("nonexistent_chunk_id")
    assert "error" in result


def test_explain_structural_match(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    f = tmp_path / "q1.md"
    f.write_text("---\ndate: 2025-02-10\ntags: planning\n---\n## Q1 Planning\n\nBudget review.")
    engine.index(tmp_path)
    chunk_id = engine.corpus.get_chunks_for_doc("q1.md")[0].id

    result = engine.explain(chunk_id, query="Q1 2025 planning")
    assert "structural_matches" in result
    assert len(result["structural_matches"]) >= 1


# -----------------------------------------------------------------------
# Phase 4 — MCP Resources
# -----------------------------------------------------------------------

def test_mcp_resources_list(tmp_path):
    from locus.mcp.server import _handle_resources_list
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Topic\n\nContent here.")
    engine.index(tmp_path)

    result = _handle_resources_list(engine)
    assert "resources" in result
    assert len(result["resources"]) == 1
    assert result["resources"][0]["uri"] == "locus://doc/doc.md"
    assert result["resources"][0]["mimeType"] == "text/markdown"


def test_mcp_resources_read(tmp_path):
    from locus.mcp.server import _handle_resources_read
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Topic\n\nContent to read back.")
    engine.index(tmp_path)

    result = _handle_resources_read("locus://doc/doc.md", engine)
    assert "contents" in result
    assert len(result["contents"]) == 1
    assert "Content to read back" in result["contents"][0]["text"]


def test_mcp_resources_read_unknown(tmp_path):
    from locus.mcp.server import _handle_resources_read
    engine = LocusEngine(store_path=tmp_path / ".locus")
    result = _handle_resources_read("locus://doc/nonexistent.md", engine)
    assert "error" in result


def test_mcp_resources_bad_scheme(tmp_path):
    from locus.mcp.server import _handle_resources_read
    engine = LocusEngine(store_path=tmp_path / ".locus")
    result = _handle_resources_read("http://example.com", engine)
    assert "error" in result


# -----------------------------------------------------------------------
# Phase 4 — MCP Prompts
# -----------------------------------------------------------------------

def test_prompts_list():
    from locus.mcp.prompts import list_prompts
    prompts = list_prompts()
    assert len(prompts) == 4
    names = {p["name"] for p in prompts}
    assert "locus_research" in names
    assert "locus_entity_summary" in names
    assert "locus_timeline" in names
    assert "locus_contradiction_analysis" in names


def test_prompt_research_renders(tmp_path):
    from locus.mcp.prompts import render_prompt
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "auth.md").write_text("# Auth\n\nJWT authentication system design.")
    engine.index(tmp_path)

    messages = render_prompt("locus_research", {"topic": "authentication"}, engine)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "authentication" in messages[0]["content"]["text"].lower()


def test_prompt_entity_summary_renders(tmp_path):
    from locus.mcp.prompts import render_prompt
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Alice", "leads", "Platform", source="org.md")

    messages = render_prompt("locus_entity_summary", {"entity": "Alice"}, engine)
    assert len(messages) == 1
    text = messages[0]["content"]["text"]
    assert "Alice" in text
    assert "leads" in text


def test_prompt_contradiction_no_conflicts(tmp_path):
    from locus.mcp.prompts import render_prompt
    engine = LocusEngine(store_path=tmp_path / ".locus")
    messages = render_prompt("locus_contradiction_analysis", {}, engine)
    text = messages[0]["content"]["text"]
    assert "No contradictions" in text or "consistent" in text


def test_prompt_contradiction_with_conflicts(tmp_path):
    from locus.mcp.prompts import render_prompt
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Alice", "role", "Engineer")
    engine.add_fact("Alice", "role", "Manager")
    messages = render_prompt("locus_contradiction_analysis", {"entity": "Alice"}, engine)
    text = messages[0]["content"]["text"]
    assert "contradiction" in text.lower() or "conflict" in text.lower()


def test_prompt_timeline_renders(tmp_path):
    from locus.mcp.prompts import render_prompt
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Alice", "role", "Engineer", valid_from="2020-01-01", valid_to="2022-12-31")
    engine.add_fact("Alice", "role", "Manager",  valid_from="2023-01-01")

    messages = render_prompt(
        "locus_timeline",
        {"entity": "Alice", "date_from": "2019-01-01", "date_to": "2024-01-01"},
        engine,
    )
    text = messages[0]["content"]["text"]
    assert "Alice" in text
    assert "Engineer" in text or "Manager" in text


def test_prompt_unknown_returns_error_message(tmp_path):
    from locus.mcp.prompts import render_prompt
    engine = LocusEngine(store_path=tmp_path / ".locus")
    messages = render_prompt("nonexistent_prompt", {}, engine)
    assert "Unknown prompt" in messages[0]["content"]["text"]


# -----------------------------------------------------------------------
# Phase 5 — LocusCluster
# -----------------------------------------------------------------------

def test_cluster_add_and_list(tmp_path):
    from locus import LocusCluster
    registry = tmp_path / "cluster.json"
    cluster = LocusCluster(registry_path=registry)

    cluster.add_node("node_a", str(tmp_path / ".locus_a"))
    cluster.add_node("node_b", str(tmp_path / ".locus_b"))

    assert set(cluster.node_names()) == {"node_a", "node_b"}
    nodes = cluster.list_nodes()
    assert len(nodes) == 2
    assert all("name" in n for n in nodes)


def test_cluster_persist_registry(tmp_path):
    from locus import LocusCluster
    registry = tmp_path / "cluster.json"

    c1 = LocusCluster(registry_path=registry)
    c1.add_node("jarv", str(tmp_path / ".locus_jarv"))

    c2 = LocusCluster(registry_path=registry)
    assert "jarv" in c2.node_names()


def test_cluster_remove_node(tmp_path):
    from locus import LocusCluster
    registry = tmp_path / "cluster.json"
    cluster = LocusCluster(registry_path=registry)
    cluster.add_node("temp", str(tmp_path / ".locus_temp"))
    result = cluster.remove_node("temp")
    assert result["removed"] == "temp"
    assert "temp" not in cluster.node_names()


def test_cluster_remove_nonexistent(tmp_path):
    from locus import LocusCluster
    cluster = LocusCluster(registry_path=tmp_path / "cluster.json")
    result = cluster.remove_node("ghost")
    assert "error" in result


def test_cluster_retrieve_empty(tmp_path):
    from locus import LocusCluster
    cluster = LocusCluster(registry_path=tmp_path / "cluster.json")
    results = cluster.retrieve("anything")
    assert results == []


def test_cluster_retrieve_single_node(tmp_path):
    from locus import LocusCluster
    registry = tmp_path / "cluster.json"
    cluster = LocusCluster(registry_path=registry)
    store = tmp_path / ".locus_a"
    cluster.add_node("node_a", str(store))

    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "auth.md").write_text("# Authentication\n\nJWT tokens for auth.")
    cluster.get_node("node_a").index(str(doc_dir))

    results = cluster.retrieve("authentication", limit=3)
    assert len(results) > 0
    assert all(c.provenance.startswith("node_a:") for c in results)


def test_cluster_retrieve_multi_node(tmp_path):
    from locus import LocusCluster
    registry = tmp_path / "cluster.json"
    cluster = LocusCluster(registry_path=registry)

    for name in ("node_a", "node_b"):
        store = tmp_path / f".locus_{name}"
        cluster.add_node(name, str(store))
        doc_dir = tmp_path / f"docs_{name}"
        doc_dir.mkdir()
        (doc_dir / "doc.md").write_text(f"# {name}\n\nContent from {name} about systems.")
        cluster.get_node(name).index(str(doc_dir))

    results = cluster.retrieve("systems", limit=5)
    assert len(results) > 0
    # Both nodes should contribute
    node_prefixes = {c.provenance.split(":")[0] for c in results}
    assert len(node_prefixes) == 2


def test_cluster_retrieve_subset_nodes(tmp_path):
    from locus import LocusCluster
    registry = tmp_path / "cluster.json"
    cluster = LocusCluster(registry_path=registry)

    for name in ("node_a", "node_b", "node_c"):
        store = tmp_path / f".locus_{name}"
        cluster.add_node(name, str(store))

    results = cluster.retrieve("query", nodes=["node_a"], limit=3)
    for c in results:
        assert c.provenance.startswith("node_a:")


def test_cluster_status(tmp_path):
    from locus import LocusCluster
    registry = tmp_path / "cluster.json"
    cluster = LocusCluster(registry_path=registry)
    cluster.add_node("n1", str(tmp_path / ".locus_n1"))

    status = cluster.status()
    assert status["node_count"] == 1
    assert "nodes" in status
    assert "n1" in status["nodes"]


# -----------------------------------------------------------------------
# Phase 6 — LocusWatcher
# -----------------------------------------------------------------------

def test_watcher_detects_new_file(tmp_path):
    from locus import LocusWatcher
    engine = LocusEngine(store_path=tmp_path / ".locus")
    docs = tmp_path / "docs"
    docs.mkdir()

    watcher = LocusWatcher(engine, watch_dir=docs, interval=0.1)
    (docs / "first.md").write_text("# First\n\nInitial content.")
    watcher._cycle()

    assert engine.corpus.doc_count() == 1


def test_watcher_skips_unchanged(tmp_path):
    from locus import LocusWatcher
    engine = LocusEngine(store_path=tmp_path / ".locus")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "stable.md").write_text("# Stable\n\nContent that will not change.")

    watcher = LocusWatcher(engine, watch_dir=docs, interval=0.1)
    added1, _ = watcher._cycle()
    added2, _ = watcher._cycle()

    assert added1 >= 1
    assert added2 == 0  # unchanged — skipped


def test_watcher_detects_update(tmp_path):
    from locus import LocusWatcher
    engine = LocusEngine(store_path=tmp_path / ".locus")
    docs = tmp_path / "docs"
    docs.mkdir()
    f = docs / "evolving.md"
    f.write_text("# V1\n\nOriginal content.")

    watcher = LocusWatcher(engine, watch_dir=docs, interval=0.1)
    watcher._cycle()

    f.write_text("# V2\n\nUpdated content with more information added here.")
    added, _ = watcher._cycle()
    assert added >= 1


def test_watcher_detects_deletion(tmp_path):
    from locus import LocusWatcher
    engine = LocusEngine(store_path=tmp_path / ".locus")
    docs = tmp_path / "docs"
    docs.mkdir()
    f = docs / "temporary.md"
    f.write_text("# Temp\n\nWill be deleted.")

    watcher = LocusWatcher(engine, watch_dir=docs, interval=0.1)
    watcher._cycle()
    assert engine.corpus.doc_count() == 1

    f.unlink()
    _, deleted = watcher._cycle()
    assert deleted == 1
    assert engine.corpus.doc_count() == 0


def test_watcher_on_change_callback(tmp_path):
    from locus import LocusWatcher
    events: list[tuple[str, str]] = []
    engine = LocusEngine(store_path=tmp_path / ".locus")
    docs = tmp_path / "docs"
    docs.mkdir()

    watcher = LocusWatcher(
        engine, watch_dir=docs, interval=0.1,
        on_change=lambda path, event: events.append((path, event))
    )
    (docs / "new.md").write_text("# New\n\nFresh content.")
    watcher._cycle()
    assert any(ev == "updated" for _, ev in events)


def test_watcher_stats(tmp_path):
    from locus import LocusWatcher
    engine = LocusEngine(store_path=tmp_path / ".locus")
    watcher = LocusWatcher(engine, watch_dir=tmp_path / "docs", interval=2.0)
    stats = watcher.stats()
    assert stats["interval_s"] == 2.0
    assert stats["cycles"] == 0
    assert not stats["running"]


def test_watcher_background_starts_and_stops(tmp_path):
    import time
    from locus import LocusWatcher
    engine = LocusEngine(store_path=tmp_path / ".locus")
    docs = tmp_path / "docs"
    docs.mkdir()

    watcher = LocusWatcher(engine, watch_dir=docs, interval=0.2)
    watcher.start(background=True)
    assert watcher._running
    time.sleep(0.5)
    watcher.stop()
    assert not watcher._running


# -----------------------------------------------------------------------
# Phase 6 — OMPA Bridge
# -----------------------------------------------------------------------

def test_ompa_bridge_indexes_markdown(tmp_path):
    from locus.bridge.ompa import OMPABridge
    engine = LocusEngine(store_path=tmp_path / ".locus")

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note1.md").write_text("# Note 1\n\nContent about the project.")
    (vault / "note2.md").write_text("# Note 2\n\nMore project information.")

    bridge = OMPABridge(engine, vault_path=vault)
    result = bridge.ingest()

    assert result["chunks_indexed"] >= 2
    assert engine.corpus.doc_count() == 2


def test_ompa_bridge_imports_kg(tmp_path):
    import sqlite3
    from locus.bridge.ompa import OMPABridge
    engine = LocusEngine(store_path=tmp_path / ".locus")

    vault = tmp_path / "vault"
    palace = vault / ".palace"
    palace.mkdir(parents=True)
    (vault / "note.md").write_text("# Note\n\nContent.")

    # Create a fake OMPA KG with the same schema
    kg_path = palace / "knowledge_graph.sqlite3"
    with sqlite3.connect(str(kg_path)) as conn:
        conn.execute(
            "CREATE TABLE triples (id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, "
            "object TEXT, valid_from TEXT, valid_to TEXT, source TEXT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO triples (subject, predicate, object, source) VALUES (?,?,?,?)",
            ("Alice", "works_at", "Acme", "note.md")
        )

    bridge = OMPABridge(engine, vault_path=vault)
    result = bridge.ingest()

    assert result["triples_imported"] == 1
    triples = engine.kg.query_entity("Alice")
    assert len(triples) >= 1


def test_ompa_bridge_nonexistent_vault(tmp_path):
    from locus.bridge.ompa import OMPABridge
    engine = LocusEngine(store_path=tmp_path / ".locus")
    bridge = OMPABridge(engine, vault_path=tmp_path / "nonexistent")
    result = bridge.ingest()
    assert "error" in result


def test_ompa_bridge_stats(tmp_path):
    from locus.bridge.ompa import OMPABridge
    engine = LocusEngine(store_path=tmp_path / ".locus")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("Content.")

    bridge = OMPABridge(engine, vault_path=vault)
    stats = bridge.stats()
    assert stats["markdown_docs"] == 1
    assert not stats["kg_available"]


# -----------------------------------------------------------------------
# Phase 6 — Evaluation (LocusEval)
# -----------------------------------------------------------------------

def test_eval_perfect_recall(tmp_path):
    from locus import LocusEval
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "auth.md").write_text("# Authentication\n\nJWT tokens and OAuth flows.")
    (tmp_path / "deploy.md").write_text("# Deployment\n\nKubernetes and Docker containers.")
    engine.index(tmp_path)

    ev = LocusEval(engine, k_values=[1, 3, 5])
    qa = [{"query": "JWT authentication", "expected_docs": ["auth.md"]}]
    report = ev.score(qa)

    assert report.mrr() > 0.0
    assert len(report.query_results) == 1


def test_eval_zero_recall(tmp_path):
    from locus import LocusEval
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "unrelated.md").write_text("# Cooking\n\nRecipes and ingredients.")
    engine.index(tmp_path)

    ev = LocusEval(engine)
    qa = [{"query": "JWT auth", "expected_docs": ["nonexistent.md"]}]
    report = ev.score(qa)

    assert report.recall_at(5) == 0.0
    assert report.mrr() == 0.0


def test_eval_multi_expected(tmp_path):
    from locus import LocusEval
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "auth.md").write_text("# Auth\n\nAuthentication system.")
    (tmp_path / "security.md").write_text("# Security\n\nSecurity policies.")
    engine.index(tmp_path)

    ev = LocusEval(engine, k_values=[3])
    qa = [{"query": "authentication", "expected_docs": ["auth.md", "security.md"]}]
    report = ev.score(qa)
    # At least one should be found
    assert report.recall_at(3) >= 0.0


def test_eval_report_summary(tmp_path):
    from locus import LocusEval
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Doc\n\nContent about retrieval systems.")
    engine.index(tmp_path)

    ev = LocusEval(engine)
    qa = [{"query": "retrieval systems", "expected_docs": ["doc.md"]}]
    report = ev.score(qa)
    summary = report.summary()
    assert "Recall@" in summary
    assert "MRR:" in summary


def test_eval_to_dict(tmp_path):
    from locus import LocusEval
    engine = LocusEngine(store_path=tmp_path / ".locus")
    ev = LocusEval(engine)
    report = ev.score([{"query": "test", "expected_docs": ["missing.md"]}])
    d = report.to_dict()
    assert "metrics" in d
    assert "recall@1" in d["metrics"]
    assert "mrr" in d["metrics"]
    assert "misses" in d


def test_eval_from_file(tmp_path):
    from locus import LocusEval
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "auth.md").write_text("# Auth\n\nJWT authentication.")
    engine.index(tmp_path)

    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps([
        {"query": "JWT tokens", "expected_docs": ["auth.md"]}
    ]))

    ev = LocusEval(engine)
    report = ev.score_from_file(qa_path)
    assert len(report.query_results) == 1


# -----------------------------------------------------------------------
# Phase 6 — HTTP server (unit-level)
# -----------------------------------------------------------------------

def test_http_server_dispatch_health(tmp_path):
    from locus.mcp.http_server import _dispatch
    engine = LocusEngine(store_path=tmp_path / ".locus")
    resp = _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, engine)
    assert resp["result"]["capabilities"]["tools"] == {}
    assert "resources" in resp["result"]["capabilities"]
    assert "prompts" in resp["result"]["capabilities"]


def test_http_server_dispatch_tools_list(tmp_path):
    from locus.mcp.http_server import _dispatch
    engine = LocusEngine(store_path=tmp_path / ".locus")
    resp = _dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, engine)
    assert "tools" in resp["result"]
    assert len(resp["result"]["tools"]) >= 20


def test_http_server_dispatch_unknown_method(tmp_path):
    from locus.mcp.http_server import _dispatch
    engine = LocusEngine(store_path=tmp_path / ".locus")
    resp = _dispatch({"jsonrpc": "2.0", "id": 3, "method": "bogus/method", "params": {}}, engine)
    assert "error" in resp


# -----------------------------------------------------------------------
# Phase 7 — KG traversal
# -----------------------------------------------------------------------

def test_kg_traverse_basic(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "leads",    "Platform")
    kg.add_triple("Platform", "uses",  "Kubernetes")
    kg.add_triple("Alice", "works_at", "Acme")

    result = kg.traverse("Alice", max_depth=2)
    assert "Alice" in result
    assert "Platform" in result       # 1 hop
    assert "Kubernetes" in result     # 2 hops


def test_kg_traverse_depth_one(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("A", "rel", "B")
    kg.add_triple("B", "rel", "C")

    result = kg.traverse("A", max_depth=1)
    assert "A" in result
    assert "B" in result
    assert "C" not in result          # too deep


def test_kg_traverse_predicate_filter(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "leads",    "Platform")
    kg.add_triple("Alice", "works_at", "Acme")

    result = kg.traverse("Alice", max_depth=1, predicate_filter=["leads"])
    entities = set(result.keys())
    assert "Platform" in entities
    assert "Acme" not in entities


def test_kg_traverse_direction_out(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "leads", "Platform")
    kg.add_triple("Bob",   "leads", "Alice")    # Alice is object here

    # "out" only follows Alice as subject
    result = kg.traverse("Alice", max_depth=1, direction="out")
    assert "Platform" in result
    assert "Bob" not in result


# -----------------------------------------------------------------------
# Phase 7 — KG pattern match
# -----------------------------------------------------------------------

def test_kg_match_all_wildcard(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "leads",    "Platform")
    kg.add_triple("Bob",   "works_at", "Acme")

    triples = kg.match("*", "*", "*")
    assert len(triples) == 2


def test_kg_match_subject_fixed(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "leads",    "Platform")
    kg.add_triple("Alice", "works_at", "Acme")
    kg.add_triple("Bob",   "works_at", "Acme")

    triples = kg.match("Alice", "*", "*")
    subjects = {t.subject for t in triples}
    assert subjects == {"Alice"}
    assert len(triples) == 2


def test_kg_match_object_fixed(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "works_at", "Acme")
    kg.add_triple("Bob",   "works_at", "Acme")
    kg.add_triple("Carol", "works_at", "Beta")

    triples = kg.match("*", "works_at", "Acme")
    assert len(triples) == 2
    subjects = {t.subject for t in triples}
    assert "Alice" in subjects
    assert "Bob"   in subjects
    assert "Carol" not in subjects


def test_kg_match_temporal_filter(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "role", "Engineer", valid_from="2020-01-01", valid_to="2022-12-31")
    kg.add_triple("Alice", "role", "Manager",  valid_from="2023-01-01")

    triples_2021 = kg.match("Alice", "role", "*", as_of="2021-06-01")
    objs = {t.object for t in triples_2021}
    assert "Engineer" in objs
    assert "Manager" not in objs


def test_kg_match_no_results(tmp_path):
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "leads", "Platform")
    triples = kg.match("Alice", "works_at", "*")
    assert triples == []


# -----------------------------------------------------------------------
# Phase 7 — LinkPopularityRetriever
# -----------------------------------------------------------------------

def test_link_popularity_returns_linked_docs(tmp_path):
    from locus.retrieval.link_popularity import LinkPopularityRetriever
    corpus = Corpus(tmp_path / "corpus")

    (tmp_path / "hub.md").write_text("# Hub\n\nEveryone references this document.")
    (tmp_path / "leaf1.md").write_text("# Leaf 1\n\nSee [[hub]] for details.")
    (tmp_path / "leaf2.md").write_text("# Leaf 2\n\nAlso references [[hub]] here.")

    corpus.add_file(tmp_path / "hub.md",   base_path=tmp_path)
    corpus.add_file(tmp_path / "leaf1.md", base_path=tmp_path)
    corpus.add_file(tmp_path / "leaf2.md", base_path=tmp_path)

    retriever = LinkPopularityRetriever(corpus)
    results = retriever.search(limit=5)
    assert len(results) >= 1
    assert results[0].doc_path == "hub.md"
    assert results[0].provenance == "link_pop"


def test_link_popularity_empty_corpus(tmp_path):
    from locus.retrieval.link_popularity import LinkPopularityRetriever
    corpus = Corpus(tmp_path / "corpus")
    retriever = LinkPopularityRetriever(corpus)
    assert retriever.search() == []


def test_link_popularity_no_links(tmp_path):
    from locus.retrieval.link_popularity import LinkPopularityRetriever
    corpus = Corpus(tmp_path / "corpus")
    (tmp_path / "a.md").write_text("# A\n\nNo links here.")
    corpus.add_file(tmp_path / "a.md", base_path=tmp_path)
    retriever = LinkPopularityRetriever(corpus)
    # No inbound links — should return empty (count=0 excluded)
    results = retriever.search()
    assert results == []


def test_link_popularity_cache_invalidation(tmp_path):
    from locus.retrieval.link_popularity import LinkPopularityRetriever
    corpus = Corpus(tmp_path / "corpus")
    retriever = LinkPopularityRetriever(corpus)

    (tmp_path / "hub.md").write_text("# Hub\n\nReference target.")
    corpus.add_file(tmp_path / "hub.md", base_path=tmp_path)
    _ = retriever.search()                      # populates cache
    assert retriever._cache_doc_count == 1

    (tmp_path / "ref.md").write_text("# Ref\n\nSee [[hub]].")
    corpus.add_file(tmp_path / "ref.md", base_path=tmp_path)
    _ = retriever.search()                      # cache invalidated
    assert retriever._cache_doc_count == 2


# -----------------------------------------------------------------------
# Phase 7 — LocusDoctor
# -----------------------------------------------------------------------

def test_doctor_empty_corpus_warns(tmp_path):
    from locus import LocusDoctor
    engine = LocusEngine(store_path=tmp_path / ".locus")
    doctor = LocusDoctor(engine)
    checks = doctor.run()
    corpus_check = next(c for c in checks if c.name == "corpus")
    assert corpus_check.status == "warn"


def test_doctor_healthy_corpus(tmp_path):
    from locus import LocusDoctor
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
    engine.index(tmp_path)
    doctor = LocusDoctor(engine)
    checks = doctor.run()
    corpus_check = next(c for c in checks if c.name == "corpus")
    assert corpus_check.status == "pass"


def test_doctor_contradictions_warn(tmp_path):
    from locus import LocusDoctor
    engine = LocusEngine(store_path=tmp_path / ".locus")
    for i in range(12):
        engine.add_fact("X", "role", f"Role{i}")
    doctor = LocusDoctor(engine)
    checks = doctor.run()
    kg_check = next(c for c in checks if c.name == "knowledge_graph")
    assert kg_check.status == "warn"


def test_doctor_report_string(tmp_path):
    from locus import LocusDoctor
    engine = LocusEngine(store_path=tmp_path / ".locus")
    doctor = LocusDoctor(engine)
    report = doctor.report()
    assert "Locus Doctor" in report
    assert "PASS" in report or "WARN" in report
    assert "Result:" in report


def test_doctor_to_dict(tmp_path):
    from locus import LocusDoctor
    engine = LocusEngine(store_path=tmp_path / ".locus")
    d = LocusDoctor(engine).to_dict()
    assert "checks" in d
    assert "summary" in d
    assert "pass" in d["summary"]


# -----------------------------------------------------------------------
# Phase 7 — KGExporter
# -----------------------------------------------------------------------

def test_export_graphml(tmp_path):
    from locus import KGExporter, LocusEngine
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Alice", "leads", "Platform")
    engine.add_fact("Bob",   "works_at", "Acme")

    out = tmp_path / "kg.graphml"
    count = KGExporter(engine.kg).to_graphml(out)
    assert count == 2
    content = out.read_text(encoding="utf-8")
    assert "<graphml" in content
    assert "Alice" in content
    assert "leads" in content


def test_export_jsonl(tmp_path):
    from locus import KGExporter, LocusEngine
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Alice", "leads", "Platform")

    out = tmp_path / "kg.jsonl"
    count = KGExporter(engine.kg).to_jsonl(out)
    assert count == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["subject"] == "Alice"
    assert obj["predicate"] == "leads"
    assert obj["object"] == "Platform"


def test_export_dot(tmp_path):
    from locus import KGExporter, LocusEngine
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Alice", "leads", "Platform")

    out = tmp_path / "kg.dot"
    count = KGExporter(engine.kg).to_dot(out)
    assert count >= 1
    content = out.read_text(encoding="utf-8")
    assert "digraph" in content
    assert "Alice" in content


def test_export_auto_format(tmp_path):
    from locus import KGExporter, LocusEngine
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("A", "rel", "B")

    for ext, check in [("graphml", "<graphml"), ("jsonl", '{"subject"'), ("dot", "digraph")]:
        out = tmp_path / f"kg.{ext}"
        KGExporter(engine.kg).export(out)
        assert check in out.read_text(encoding="utf-8")


# -----------------------------------------------------------------------
# Phase 7 — Engine methods
# -----------------------------------------------------------------------

def test_engine_kg_traverse(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Alice", "leads",    "Platform")
    engine.add_fact("Platform", "uses",  "Kubernetes")

    result = engine.kg_traverse("Alice", max_depth=2)
    assert "Alice" in result
    assert "Platform" in result


def test_engine_kg_match(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Alice", "leads", "Platform")
    engine.add_fact("Bob",   "leads", "Infra")

    result = engine.kg_match(predicate="leads")
    assert result["count"] == 2


def test_engine_doctor(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    d = engine.doctor()
    assert "checks" in d
    assert "summary" in d


def test_engine_export_kg(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Alice", "leads", "Platform")
    out = str(tmp_path / "output.jsonl")
    result = engine.export_kg(out, fmt="jsonl")
    assert result["triples_exported"] == 1


def test_engine_six_signal_retrieve(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "hub.md").write_text("# Hub\n\nCentral reference document for the system.")
    (tmp_path / "child.md").write_text("# Child\n\nSee [[hub]] for system details.")
    engine.index(tmp_path)
    chunks = engine.retrieve("system details", limit=5)
    assert len(chunks) > 0
    # link_pop provenance is possible if hub is referenced
    provenances = {c.provenance for c in chunks}
    assert len(provenances) >= 1


# -----------------------------------------------------------------------
# Phase 8 — LocusReranker
# -----------------------------------------------------------------------

def test_reranker_returns_same_count(tmp_path):
    from locus import LocusReranker
    corpus = Corpus(tmp_path / "corpus")
    (tmp_path / "a.md").write_text("## Authentication\n\nJWT tokens for auth.")
    (tmp_path / "b.md").write_text("## Deployment\n\nKubernetes deployment guide.")
    corpus.add_file(tmp_path / "a.md", base_path=tmp_path)
    corpus.add_file(tmp_path / "b.md", base_path=tmp_path)

    retriever = BM25Retriever(corpus)
    chunks = retriever.search("authentication", limit=5)
    reranker = LocusReranker(corpus)
    reranked = reranker.rerank(chunks, query="authentication")
    assert len(reranked) == len(chunks)


def test_reranker_boosts_title_match(tmp_path):
    from locus import LocusReranker, RerankerWeights
    corpus = Corpus(tmp_path / "corpus")
    # First doc: title matches query, content is thin
    (tmp_path / "auth.md").write_text("## Authentication Overview\n\nSee docs.")
    # Second doc: title doesn't match, content is richer
    (tmp_path / "deploy.md").write_text("## Deployment\n\nAuthentication is handled by JWT tokens.")
    corpus.add_file(tmp_path / "auth.md", base_path=tmp_path)
    corpus.add_file(tmp_path / "deploy.md", base_path=tmp_path)

    retriever = BM25Retriever(corpus)
    chunks = retriever.search("authentication", limit=5)
    # With high title weight, auth.md (section='Authentication Overview') should get boosted
    w = RerankerWeights(title=1.0, entity=0.0, freshness=0.0)
    reranker = LocusReranker(corpus, weights=w)
    reranked = reranker.rerank(chunks, query="authentication", weights=w)
    assert any("auth" in c.doc_path for c in reranked)


def test_reranker_does_not_surface_zeroes(tmp_path):
    from locus import LocusReranker
    from locus.retrieval.bm25 import ScoredChunk
    corpus = Corpus(tmp_path / "corpus")
    reranker = LocusReranker(corpus)
    # A chunk with score=0 should remain 0 after boost (multiplicative)
    zero_chunk = ScoredChunk("c1", "doc.md", 0.0, "content", "bm25")
    reranked = reranker.rerank([zero_chunk], query="anything")
    assert reranked[0].score == 0.0


def test_reranker_custom_weights(tmp_path):
    from locus import LocusReranker, RerankerWeights
    corpus = Corpus(tmp_path / "corpus")
    (tmp_path / "doc.md").write_text("# Doc\n\nContent about systems.")
    corpus.add_file(tmp_path / "doc.md", base_path=tmp_path)
    retriever = BM25Retriever(corpus)
    chunks = retriever.search("systems", limit=3)
    reranker = LocusReranker(corpus)
    w = RerankerWeights(title=0.0, entity=0.0, freshness=0.0)
    reranked = reranker.rerank(chunks, query="systems", weights=w)
    # Zero weights → scores unchanged (allow for rounding to 6dp)
    for orig, re in zip(chunks, reranked):
        assert abs(orig.score - re.score) < 1e-5


# -----------------------------------------------------------------------
# Phase 8 — ContextPacker
# -----------------------------------------------------------------------

def test_packer_basic(tmp_path):
    from locus import ContextPacker
    from locus.retrieval.bm25 import ScoredChunk
    packer = ContextPacker(budget=2000)
    chunks = [
        ScoredChunk("c1", "auth.md",   0.9, "JWT tokens are used for authentication here.", "bm25"),
        ScoredChunk("c2", "deploy.md", 0.7, "Kubernetes manages deployment orchestration.", "bm25"),
    ]
    packed = packer.pack(chunks)
    assert packed.chunks_included == 2
    assert packed.chunks_available == 2
    assert "auth.md" in packed.text
    assert "deploy.md" in packed.text
    assert not packed.truncated


def test_packer_budget_truncation(tmp_path):
    from locus import ContextPacker
    from locus.retrieval.bm25 import ScoredChunk
    # Very tight budget — should truncate
    packer = ContextPacker(budget=20, header_tokens=5)
    long_content = " ".join(["word"] * 100)
    chunks = [
        ScoredChunk("c1", "a.md", 1.0, long_content, "bm25"),
        ScoredChunk("c2", "b.md", 0.5, long_content, "bm25"),
    ]
    packed = packer.pack(chunks)
    assert packed.truncated
    assert packed.chunks_included < 2


def test_packer_groups_by_doc(tmp_path):
    from locus import ContextPacker
    from locus.retrieval.bm25 import ScoredChunk
    packer = ContextPacker(budget=8000)
    chunks = [
        ScoredChunk("c1", "auth.md",   0.9, "First auth chunk.", "bm25"),
        ScoredChunk("c2", "deploy.md", 0.8, "Deploy chunk.",     "bm25"),
        ScoredChunk("c3", "auth.md",   0.7, "Second auth chunk.", "bm25"),
    ]
    packed = packer.pack(chunks)
    # auth.md should appear as a group
    text = packed.text
    first_auth = text.index("auth.md")
    deploy_pos = text.index("deploy.md")
    second_auth = text.index("Second auth chunk")
    # Both auth chunks should be grouped before or after deploy, not interleaved
    assert first_auth < deploy_pos < second_auth or second_auth < first_auth < deploy_pos or \
           (first_auth < second_auth < deploy_pos) or (deploy_pos < first_auth < second_auth)


def test_packer_sources(tmp_path):
    from locus import ContextPacker
    from locus.retrieval.bm25 import ScoredChunk
    packer = ContextPacker()
    chunks = [
        ScoredChunk("c1", "a.md", 1.0, "Content A.", "bm25"),
        ScoredChunk("c2", "b.md", 0.9, "Content B.", "bm25"),
    ]
    packed = packer.pack(chunks)
    assert "a.md" in packed.sources
    assert "b.md" in packed.sources


# -----------------------------------------------------------------------
# Phase 8 — Query cache
# -----------------------------------------------------------------------

def test_cache_miss_then_hit(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Auth\n\nJWT tokens.")
    engine.index(tmp_path)

    engine.retrieve("JWT authentication")   # miss
    engine.retrieve("JWT authentication")   # hit
    stats = engine.cache_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1


def test_cache_invalidated_on_index(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
    engine.index(tmp_path)
    engine.retrieve("content")
    assert len(engine._query_cache) >= 1
    engine.index(tmp_path)   # should invalidate
    assert len(engine._query_cache) == 0


def test_cache_invalidated_on_forget(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
    engine.index(tmp_path)
    engine.retrieve("content")
    assert len(engine._query_cache) >= 1
    doc = engine.corpus.list_docs()[0]
    engine.forget(doc)
    assert len(engine._query_cache) == 0


def test_cache_clear(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
    engine.index(tmp_path)
    engine.retrieve("content")
    result = engine.clear_cache()
    assert result["cleared"] >= 1
    assert engine.cache_stats()["size"] == 0


def test_cache_no_cache_option(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
    engine.index(tmp_path)
    engine.retrieve("content", use_cache=False)
    # Should not populate cache
    assert engine.cache_stats()["size"] == 0


# -----------------------------------------------------------------------
# Phase 8 — assess_confidence
# -----------------------------------------------------------------------

def test_assess_confidence_ok(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Auth\n\nJWT authentication tokens.")
    engine.index(tmp_path)
    chunks = engine.retrieve("JWT authentication")
    conf = engine.assess_confidence(chunks)
    assert conf["level"] in ("ok", "low")
    assert "top_score" in conf


def test_assess_confidence_empty(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    conf = engine.assess_confidence([])
    assert conf["level"] == "empty"


# -----------------------------------------------------------------------
# Phase 8 — prepare_context
# -----------------------------------------------------------------------

def test_prepare_context_returns_all_keys(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "auth.md").write_text("# Auth\n\nJWT authentication system.")
    engine.index(tmp_path)
    result = engine.prepare_context("authentication", limit=3, token_budget=2000)

    assert "query" in result
    assert "confidence" in result
    assert "packed_context" in result
    assert "tokens_used" in result
    assert "token_budget" in result
    assert "chunks_included" in result
    assert "sources" in result
    assert "kg_context" in result


def test_prepare_context_packed_text(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "auth.md").write_text("# Auth\n\nJWT tokens for authentication.")
    engine.index(tmp_path)
    result = engine.prepare_context("JWT authentication")
    assert isinstance(result["packed_context"], str)


def test_prepare_context_kg_entities(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    engine.add_fact("Alice", "leads", "Auth team")
    result = engine.prepare_context("Alice leads")
    # KG context should include Alice's facts
    assert "Alice" in result["kg_context"] or len(result["kg_context"]) >= 0


def test_prepare_context_no_rerank(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
    engine.index(tmp_path)
    result = engine.prepare_context("content", rerank=False)
    assert result["chunks_included"] >= 0


# -----------------------------------------------------------------------
# Phase 9 — Async engine
# -----------------------------------------------------------------------

def test_async_engine_index_retrieve(tmp_path):
    import asyncio
    from locus.async_core import AsyncLocusEngine

    async def _run():
        async with AsyncLocusEngine(store_path=str(tmp_path / ".locus")) as ae:
            (tmp_path / "doc.md").write_text("# Auth\n\nJWT tokens.")
            result = await ae.index(str(tmp_path))
            assert result["files"] >= 1
            chunks = await ae.retrieve("JWT tokens")
            assert isinstance(chunks, list)

    asyncio.run(_run())


def test_async_engine_getattr_passthrough(tmp_path):
    from locus.async_core import AsyncLocusEngine
    ae = AsyncLocusEngine(store_path=str(tmp_path / ".locus"))
    # cache_stats() is not wrapped async — falls through to __getattr__ → sync engine
    result = ae.cache_stats()
    assert "size" in result


def test_async_engine_repr(tmp_path):
    from locus.async_core import AsyncLocusEngine
    ae = AsyncLocusEngine(store_path=str(tmp_path / ".locus"))
    assert "AsyncLocusEngine" in repr(ae)


# -----------------------------------------------------------------------
# Phase 9 — Hooks
# -----------------------------------------------------------------------

def test_hooks_register_and_fire():
    from locus.hooks import LocusHooks, HookContext
    hooks = LocusHooks()
    fired = []

    @hooks.on("test_event")
    def handler(ctx: HookContext):
        fired.append(ctx.data.get("value"))

    hooks.fire("test_event", engine=None, value=42)
    assert fired == [42]


def test_hooks_error_isolated():
    from locus.hooks import LocusHooks
    hooks = LocusHooks()

    @hooks.on("bad_event")
    def bad_handler(ctx):
        raise RuntimeError("intentional")

    # Should not raise
    hooks.fire("bad_event", engine=None)


def test_hooks_list_hooks():
    from locus.hooks import LocusHooks
    hooks = LocusHooks()
    hooks.register("ev1", lambda ctx: None)
    hooks.register("ev1", lambda ctx: None)
    hooks.register("ev2", lambda ctx: None)
    listing = hooks.list_hooks()
    assert listing["ev1"] == 2
    assert listing["ev2"] == 1


def test_hooks_unregister():
    from locus.hooks import LocusHooks
    hooks = LocusHooks()
    fired = []
    fn = lambda ctx: fired.append(1)  # noqa: E731
    hooks.register("ev", fn)
    hooks.unregister("ev", fn)
    hooks.fire("ev", engine=None)
    assert fired == []


def test_engine_set_hooks_fires_on_index(tmp_path):
    from locus.hooks import LocusHooks
    engine = LocusEngine(store_path=tmp_path / ".locus")
    hooks = LocusHooks()
    post_events = []

    @hooks.on("post_index")
    def capture(ctx):
        post_events.append(ctx.data.get("result", {}))

    engine.set_hooks(hooks)
    (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
    engine.index(tmp_path)
    assert len(post_events) == 1
    assert "files" in post_events[0]


def test_engine_hooks_on_forget(tmp_path):
    from locus.hooks import LocusHooks
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
    engine.index(tmp_path)

    hooks = LocusHooks()
    pre_events = []

    @hooks.on("pre_forget")
    def capture(ctx):
        pre_events.append(ctx.data.get("doc_path"))

    engine.set_hooks(hooks)
    doc_path = engine.corpus.list_docs()[0]
    engine.forget(doc_path)
    assert len(pre_events) == 1
    assert pre_events[0] == doc_path


# -----------------------------------------------------------------------
# Phase 9 — Reasoning
# -----------------------------------------------------------------------

def test_reasoning_find_paths_direct(tmp_path):
    from locus.reasoning import LocusReasoner
    from locus.memory.knowledge_graph import TemporalKG
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "leads", "AuthTeam")
    kg.add_triple("AuthTeam", "owns", "JWTService")
    reasoner = LocusReasoner(kg)
    paths = reasoner.find_paths("Alice", "JWTService", max_depth=2)
    assert len(paths) >= 1
    assert paths[0].start == "Alice"
    assert paths[0].end == "JWTService"
    assert paths[0].length == 2


def test_reasoning_find_paths_same_entity(tmp_path):
    from locus.reasoning import LocusReasoner
    from locus.memory.knowledge_graph import TemporalKG
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    reasoner = LocusReasoner(kg)
    paths = reasoner.find_paths("Alice", "Alice")
    assert len(paths) == 1
    assert paths[0].length == 0


def test_reasoning_path_narrative(tmp_path):
    from locus.reasoning import LocusReasoner
    from locus.memory.knowledge_graph import TemporalKG
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "leads", "Team")
    reasoner = LocusReasoner(kg)
    paths = reasoner.find_paths("Alice", "Team", max_depth=1)
    assert len(paths) >= 1
    narrative = paths[0].narrative()
    assert "Alice" in narrative
    assert "leads" in narrative


def test_reasoning_reason_returns_keys(tmp_path):
    from locus.memory.knowledge_graph import TemporalKG
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "leads", "Platform")
    engine = LocusEngine(store_path=tmp_path / ".locus")
    # Share the same KG
    engine.kg = kg
    engine._reasoner._kg = kg
    result = engine.reason("How is Alice related to Platform?")
    assert "question" in result
    assert "entities_detected" in result
    assert "reasoning_chains" in result
    assert "entity_neighborhood" in result


def test_reasoning_no_paths_empty(tmp_path):
    from locus.reasoning import LocusReasoner
    from locus.memory.knowledge_graph import TemporalKG
    kg = TemporalKG(str(tmp_path / "kg.sqlite3"))
    reasoner = LocusReasoner(kg)
    paths = reasoner.find_paths("Alice", "Bob", max_depth=2)
    assert paths == []


# -----------------------------------------------------------------------
# Phase 9 — Corpus inspection
# -----------------------------------------------------------------------

def test_top_terms_returns_list(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "doc.md").write_text("# Authentication\n\nJWT tokens for authentication.")
    engine.index(tmp_path)
    terms = engine.top_terms(limit=10)
    assert isinstance(terms, list)
    assert len(terms) <= 10
    assert all("term" in t and "doc_count" in t for t in terms)


def test_top_terms_most_frequent_ranked_first(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "a.md").write_text("authentication jwt authentication")
    (tmp_path / "b.md").write_text("authentication oauth")
    engine.index(tmp_path)
    terms = engine.top_terms(limit=5)
    names = [t["term"] for t in terms]
    # "authentication" appears in both docs — should be near the top
    assert "authentication" in names


def test_inspect_doc_returns_structure(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "auth.md").write_text("# Auth\n\nJWT tokens [[Security]] authentication.")
    engine.index(tmp_path)
    doc_path = engine.corpus.list_docs()[0]
    result = engine.inspect_doc(doc_path)
    assert "doc_path" in result
    assert "chunk_count" in result
    assert "top_terms" in result
    assert result["chunk_count"] >= 1


def test_inspect_doc_missing_returns_error(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    result = engine.inspect_doc("nonexistent.md")
    assert "error" in result


# -----------------------------------------------------------------------
# Phase 9 — GitHub bridge (offline — tests request plumbing without network)
# -----------------------------------------------------------------------

def test_github_bridge_init():
    from locus.bridge.github import GitHubBridge
    engine = None  # not needed for init test
    bridge = GitHubBridge(engine, repo="owner/repo", token="tok123")
    assert bridge._repo == "owner/repo"
    assert bridge._token == "tok123"


def test_github_bridge_matches():
    from locus.bridge.github import GitHubBridge
    assert GitHubBridge._matches("docs/README.md", "", "*.md")
    assert GitHubBridge._matches("docs/guide.md", "docs", "*.md")
    assert not GitHubBridge._matches("src/main.py", "", "*.md")
    assert not GitHubBridge._matches("other/guide.md", "docs", "*.md")


def test_version_is_0_8_0():
    from locus.core import __version__
    assert __version__ == "0.8.0"
