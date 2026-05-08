# Changelog

All notable changes to Locus are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.6.0] — 2026-05-07

### Added
- **KG graph traversal** — `kg.traverse(start, max_depth, predicate_filter, direction)`: BFS walk returning `{entity: [triples]}` for the full reachable subgraph
- **KG pattern matching** — `kg.match(subject, predicate, obj, as_of)`: wildcard query (`*` matches any value); temporal filter supported
- **`LinkPopularityRetriever`** — 6th retrieval signal; ranks documents by inbound wikilink count; lazy-computed and cached until corpus changes; weight 0.2 in RRF
- **`LocusDoctor`** — structured health diagnostics: corpus, KG, bulletin, entity resolver, store size, version; PASS/WARN/FAIL per check
- **`KGExporter`** — export KG to three formats: GraphML (Gephi/Cytoscape/yEd), JSONL (scripting), DOT (Graphviz); auto-detects format from file extension
- New CLI commands: `doctor`, `export-kg`, `kg-traverse`, `kg-match`
- New MCP tools: `locus_doctor`, `locus_export_kg`, `locus_kg_traverse`, `locus_kg_match` (total: 26)
- `LocusDoctor` and `KGExporter` exported from top-level `locus` package

### Changed
- Six-signal retrieval pipeline in `LocusEngine.retrieve()` (was five signals)
- `__version__` bumped to `0.6.0`

---

## [0.5.0] — 2026-05-07

### Added
- **`LocusWatcher`** — poll-based file watcher; detects adds, updates, and deletions via checksum dedup; runs blocking or as a daemon thread; optional `on_change` callback
- **HTTP JSON-RPC transport** — `locus/mcp/http_server.py`; `POST /rpc`, `GET /health`, `GET /tools`; optional Bearer-token auth; CORS headers; zero external deps
- **OMPA Bridge** — `locus/bridge/ompa.py`: indexes markdown + copies KG triples directly from `.palace/knowledge_graph.sqlite3` (identical schema, zero transformation); safe to re-run
- **Evaluation framework** — `LocusEval.score()`: Recall@K + MRR; `EvalReport.summary()` with miss analysis; `score_from_file()` loads QA JSON
- New CLI commands: `watch`, `serve`, `ingest-ompa`, `benchmark`
- New MCP tools: `locus_ingest_ompa`, `locus_benchmark`
- `LocusWatcher`, `LocusEval`, `OMPABridge` exported from top-level package

### Changed
- `__version__` bumped to `0.5.0`

---

## [0.4.0] — 2026-05-07

### Added
- **`locus_explain`** — explains *why* a chunk was retrieved: BM25 matched terms, KG entity links, structural signals, plain-English narrative; unique to graph-based RAG
- **MCP Resources** — all indexed documents exposed as `locus://doc/{path}` via `resources/list` and `resources/read`
- **MCP Prompts** — four live-context templates: `locus_research`, `locus_entity_summary`, `locus_timeline`, `locus_contradiction_analysis`
- **`LocusCluster`** — multi-node cluster; persistent JSON registry; cross-node RRF with node-prefixed provenance (`node_name:signal`)
- New MCP tools: `locus_cluster_retrieve`, `locus_add_node`, `locus_remove_node`, `locus_list_nodes`
- `LocusCluster` exported from top-level package
- `initialize` response now announces `tools`, `resources`, and `prompts` capabilities

### Changed
- `__version__` bumped to `0.4.0`

---

## [0.3.0] — 2026-05-06

### Added — Phase 2 (Retrieval Signal Expansion)
- **`StructuralRetriever`** — scores documents by frontmatter `date`, `tags`, `type`; activates only when structural signals are detected in the query (no noise otherwise)
- **`RecencyRetriever`** — exponential freshness prior; half-life 30 days; weight 0.3 in RRF
- **Section-aware chunking** — `Chunker(section_aware=True)` (default): splits at `#`/`##`/`###` heading boundaries; long sections fall back to word-count; `metadata.section` preserved
- Five-signal RRF pipeline: `[BM25, KG, LinkWalk, Structural, Recency]`

### Added — Phase 3 (KG Intelligence)
- **Prose triple extractor** — 14 relation patterns (`leads`, `works_at`, `is_a`, `part_of`, `replaced`, `depends_on`, `reports_to`, ...); auto-applied during `index()`; no models required
- **`EntityResolver`** — SQLite alias table; transparent resolution inside `TemporalKG.add_triple()` and `query_entity()`
- **Contradiction detection** — `find_contradictions()`: same-predicate triples with different objects and overlapping validity windows
- **`kg.all_entities()`** — all distinct entities for alias suggestion
- New MCP tools: `locus_contradictions`, `locus_add_alias`, `locus_suggest_aliases` (total: 15)

### Changed
- `LocusEngine` now creates `EntityResolver` at init and passes it to `TemporalKG`
- `status()` and `session_start()` include `resolver` stats
- `__version__` bumped to `0.3.0`

---

## [0.2.0] — 2026-05-06 — Phase 1 Hardening

### Added
- **Bulletin persistence** — `ContextBulletin(db_path=...)`: all mutations write-through to SQLite; hot tier survives restarts; tier0/tier1 reconstructed on load
- **Corpus checksum dedup** — `add_file()` hashes file content; skips unchanged files; `force=True` escape hatch; migration-safe via `ALTER TABLE`
- **Stats cache** — `Corpus.doc_count()` and `avg_doc_length()` cached in-process; invalidated by `add_file()` / `remove_file()`
- **Batch chunk fetch** — `get_chunks_batch()`: single `WHERE id IN (...)` query instead of N individual fetches
- **BM25 IDF cache** — term IDF values cached; invalidated when `doc_count` changes
- **Query intent classifier** — `classify_query()`: `KG_FIRST` / `BM25_FIRST` / `BALANCED`; pattern-matched; used to set per-query RRF weights
- **Weighted RRF** — `rrf_fuse(..., weights=[...])`: per-list multipliers
- **`full_content` flag** on `locus_retrieve` MCP tool
- **`intent` override** on `locus_retrieve` MCP tool
- New file: `locus/retrieval/classifier.py`

### Changed
- `LocusEngine.retrieve()` auto-classifies intent and applies weights
- `__version__` bumped to `0.2.0`

---

## [0.1.0] — 2026-05-06 — Initial Release

### Added
- **BM25+ retriever** — Robertson BM25 with delta floor; pure Python; zero external dependencies
- **`TemporalKG`** — SQLite triple store with `valid_from` / `valid_to` validity windows; populated from wikilinks and tags during indexing
- **`LinkWalker`** — wikilink/citation graph traversal from top BM25 hits; score decay per hop depth
- **RRF Fusion** — `rrf_fuse()`: Reciprocal Rank Fusion across all retrieval signals
- **`ContextBulletin`** — tiered context board (Tier 0 Pinned / Tier 1 Hot / Tier 2 Archive); hit-boost + age-decay scoring
- **`ContextBudget`** — soft token monitoring; WARNING / TREND / CRITICAL alerts
- **`LocusEngine`** orchestrator — `index()`, `retrieve()`, `add_fact()`, `query_entity()`, `session_start()`, `wrap_up()`, `status()`
- **MCP server** — 12 tools via JSON-RPC over stdio; compatible with Claude Desktop, Cursor, Windsurf
- **CLI** — `locus index`, `retrieve`, `status`, `session-start`, `wrap-up`, `add-fact`, `query-entity`, `forget`, `sync`, `mcp`
- **Section-aware chunker** — overlap-aware; frontmatter extraction; wikilink extraction
- **`Corpus`** — SQLite-backed inverted index; `add_file()`, `add_directory()`, `remove_file()`
- 40/40 tests passing
