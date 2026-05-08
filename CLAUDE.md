# Locus — Claude Code Context

Locus is a **vectorless RAG engine** with MCP integration. No embedding models, no vector databases, no GPU required. Six retrieval signals fused via Reciprocal Rank Fusion. Every result includes a provenance tag explaining why it was returned.

## Quick orientation

```bash
pip install -e ".[dev]"
pytest tests/ -v         # 167 tests, all should pass
locus --help             # CLI entry point
```

## Key design decisions

**Zero external dependencies** — everything uses Python stdlib. `sqlite3`, `re`, `pathlib`, `threading`, `http.server`, `hashlib`, `json`. This is a hard constraint — do not add runtime deps.

**Six retrieval signals, all composable** — BM25, KGRetriever, LinkWalker, StructuralRetriever, RecencyRetriever, LinkPopularityRetriever. All return `list[ScoredChunk]`. All are fused in `LocusEngine.retrieve()` via `rrf_fuse()`. Adding a 7th signal means creating a new file in `locus/retrieval/`, adding it to `LocusEngine.__init__`, and adding it to the `rrf_fuse()` call.

**SQLite everywhere** — corpus index, KG triples, entity aliases, bulletin entries. All in the `.locus/` store directory. No external databases.

**Section-aware chunking by default** — `Chunker(section_aware=True)` splits at markdown heading boundaries. Documents without headings fall back to word-count sliding window. Each chunk stores `metadata.section`.

**Temporal KG** — every triple has `valid_from` / `valid_to`. `query_entity(entity, as_of="2024-01-01")` returns only facts valid on that date. `find_contradictions()` surfaces same-predicate triples with overlapping windows.

**Prose extraction is automatic** — during `index()`, `TemporalKG.populate_from_file()` runs the prose extractor on every file. 14 relation patterns extract S-P-O triples from natural language. No annotation required.

**Entity resolution is transparent** — `EntityResolver` sits inside `TemporalKG`. `add_triple("Jeff", ...)` resolves "Jeff" → "Jeff Milam" before writing if the alias is registered. `query_entity("Jeff")` also resolves first.

**Bulletin is write-through** — `ContextBulletin(db_path=...)` persists all tier0/tier1 entries to `bulletin.sqlite3` on every mutation. On init it reloads from DB. The hot tier survives restarts.

**MCP server handles three protocols** — stdio (default), HTTP JSON-RPC (`locus serve`), and it announces `tools + resources + prompts` capabilities on `initialize`.

## Architecture layers

```
Interface:  CLI (__main__.py) · MCP stdio (mcp/server.py) · HTTP (mcp/http_server.py)
Retrieval:  BM25 · KGRetriever · LinkWalker · Structural · Recency · LinkPopularity
            → RRF Fusion (retrieval/fusion.py)
Memory:     Corpus (corpus.py) · TemporalKG (knowledge_graph.py)
            EntityResolver (entity_resolver.py) · Chunker (chunker.py)
Context:    ContextBulletin (bulletin.py) · ContextBudget (budget.py)
Cluster:    LocusCluster (cluster.py) — multi-node, persistent JSON registry
Extras:     LocusWatcher · LocusEval · OMPABridge · LocusDoctor · KGExporter
```

## Common tasks

**Adding a test** — append to `tests/test_locus.py`. Use `tmp_path` fixture for all file I/O. Run with `pytest tests/test_locus.py::test_name -v`.

**Changing the retrieval pipeline** — edit `LocusEngine.retrieve()` in `core.py`. The `all_weights` list corresponds positionally to the lists passed to `rrf_fuse()`.

**Adding a CLI command** — add `sub.add_parser(...)` in `locus/__main__.py`, add the `elif args.cmd == "..."` handler block.

**Adding an MCP tool** — add to `TOOLS` dict in `mcp/tools.py`, add handler branch in `_call_tool()` in `mcp/server.py`.

**Version** — in `locus/core.py` → `__version__`. Also update `CHANGELOG.md` and `pyproject.toml`.

## File map (key files only)

| File | Purpose |
|---|---|
| `locus/core.py` | `LocusEngine` — the main API surface |
| `locus/memory/knowledge_graph.py` | `TemporalKG` — triples, traversal, match, contradictions |
| `locus/memory/corpus.py` | `Corpus` — inverted index, checksum dedup, batch fetch |
| `locus/memory/chunker.py` | `Chunker` — section-aware + word-count splitting |
| `locus/memory/extractor.py` | `extract_triples_from_text()` — prose triple extraction |
| `locus/retrieval/bm25.py` | `BM25Retriever` — IDF-cached, batch chunk fetch |
| `locus/retrieval/fusion.py` | `rrf_fuse()` — weighted Reciprocal Rank Fusion |
| `locus/retrieval/classifier.py` | `classify_query()` → `QueryIntent` |
| `locus/context/bulletin.py` | `ContextBulletin` — tiered, SQLite-persistent |
| `locus/mcp/server.py` | MCP JSON-RPC loop + `_call_tool()` dispatch |
| `locus/mcp/prompts.py` | 4 live-context MCP prompt templates |
| `locus/cluster.py` | `LocusCluster` — multi-node, JSON registry |
| `locus/bridge/ompa.py` | `OMPABridge` — OMPA vault importer |
| `locus/eval.py` | `LocusEval` — Recall@K, MRR benchmark |
| `locus/doctor.py` | `LocusDoctor` — health diagnostics |
| `locus/export.py` | `KGExporter` — GraphML, JSONL, DOT |
| `locus/watcher.py` | `LocusWatcher` — polling file watcher |
| `tests/test_locus.py` | 167 tests covering all modules |
