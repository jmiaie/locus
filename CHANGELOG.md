# Changelog

All notable changes to Locus are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [1.1.0] — 2026-10-24

First release since 1.0.0. Everything listed below ships in this version. Upgrading is API-compatible, but two
defaults change results: the SQLite FTS5 backend with Porter stemming + English stop-words, and BM25 as the default
retrieval mode. Existing stores migrate on first open (one-way; see "Stores migrate automatically").

### Added — LLM query-transform benchmark (2026-10-23)
- **Measured `locus.llm` against a real endpoint** (`openai/gpt-4o-mini` via OpenRouter, temperature 0, all BEIR test queries, paired bootstrap B=10,000). NFCorpus nDCG@10: rewrite 0.347 (+0.018), multi-query 0.350 (+0.021), HyDE 0.359 (+0.030), all significant. SciFact: none significant (+0.003 to +0.008). Added latency about 0.8–2 s per query; $0.015–0.076 per 1,000 queries. Artefacts, including the exact LLM outputs, are in `benchmarks/published/2026-10-23-llm/`. The plugin stays opt-in.
- **Packaging metadata:** `Documentation` and `Benchmarks` project URLs, `Operating System :: OS Independent` and `Topic :: Text Processing :: Indexing` classifiers.

### Added — optional cross-encoder reranking
- **`locus[rerank]` extra** (`sentence-transformers`). `locus.rerank.CrossEncoderReranker` reranks the BM25 top-k with a local cross-encoder (default `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU).
  - Use it as `LocusEngine(reranker=...)` or `locus retrieve --rerank [--rerank-depth N --rerank-model M]`.
  - It is opt-in: NFCorpus improves by +0.018 nDCG@10 (significant), other benchmarks are not significantly changed, and it costs about 400 ms per query on CPU.
  - The core never imports it.

### Added — chunking options, parent-child retrieval, RM3 (all opt-in; see docs/COMPARISON.md)
- **Chunking options.** `LocusEngine(chunking={...})` / `locus index --chunk-words --overlap-words --heading-context --boundaries` / `LOCUS_CHUNKING`. The settings are stored with the index, and changing them re-chunks on the next `index()`. With no options set, chunks are byte-identical to before.
- **Parent sections.** Sections are stored as parents. `retrieve(..., parent="section"|"doc")` returns the enclosing section or document, with cited character `spans`.
- **RM3 expansion.** `retrieve(..., expansion="rm3", rm3={...})` (`--expand rm3`, MCP `expansion`) adds RM3 pseudo-relevance feedback. It is pure standard library and deterministic, and gives the same rankings on the fts5 and legacy backends. `Corpus.score_terms()` is new.
- **Reranker hook.** `LocusEngine(reranker=...)` adds a generic second-stage reranker hook, with a `retrieve(..., rerank=)` toggle.

### Fixed
- **Symlinks.** Directory indexing (and `watch`/`diff`) no longer follows symlinks by default, so files outside the indexed tree are never read. `follow_symlinks="inside"` follows only links that resolve inside the root, and `True` restores the old behaviour. The setting is also available as `--follow-symlinks no|inside|all` and `LOCUS_FOLLOW_SYMLINKS`.

### Added — metadata filters and optional LLM query transformations
- **`retrieve(..., filters=...)`.** Filters by path glob (include/exclude), frontmatter `type`, tags (all/any), `date` range, point-in-time `as_of` (honours `supersedes`) and custom frontmatter fields (`eq ne in nin gt gte lt lte contains exists`). Available in the API, the CLI (`--path --exclude-path --type --tag --any-tag --date-from --date-to --in-force --meta --filters`) and the `locus_retrieve` MCP tool (`filters`). Filtering happens after ranking with a deepening candidate pool, so scores are unchanged and up to `limit` matches are returned whenever they exist. `LocusEngine.filter_matches()` and `Corpus.doc_frontmatter()` / `supersedes_index()` are new helpers.
- **`locus.llm`.** Provider-agnostic `rewrite`, `multi` (RRF) and `hyde` query transformations over any OpenAI-compatible endpoint (`OPENROUTER_API_KEY`, or `OPENAI_API_KEY` + `OPENAI_BASE_URL`). Standard library only, with usage accounting, a token cap and memoisation. Available as `locus retrieve --llm`. It is opt-in and never imported by the core. Benchmarked on 2026-10-23 (see above).

### Added — on-prem packaging and a construction demo
- **Container image.** `Dockerfile` builds a slim image from the local wheel with `pip --no-index`. The base `python:3.12-slim` is pinned by digest. The image runs as non-root uid 10001, works with a read-only rootfs, has a `/health` HEALTHCHECK and stops cleanly on SIGTERM. `deploy/docker-compose.yml` publishes on host loopback only and uses a token secret, `read_only`, `cap_drop: ALL`, `no-new-privileges` and limits.
- **Offline bundle.** `make offline-bundle` produces a wheel, sdist, `install.sh` and SHA256SUMS in a reproducible tarball. It installs with no network. `make image` / `make image-save` never push anywhere.
- **`locus serve` options.** `--token-file` / `LOCUS_HTTP_TOKEN_FILE` (Docker secrets), plus `LOCUS_HTTP_HOST`, `LOCUS_HTTP_PORT`, `LOCUS_HTTP_TOKEN` and `LOCUS_STORE` env defaults. SIGTERM is treated as Ctrl-C.
- **Smoke test.** `scripts/container-smoke-test.sh` makes 24 checks, run with `--network=none`. All 24 passed with podman.
- **Construction demo.** `examples/construction/` is a synthetic project of 24 documents with 21 labelled questions, covering RFI trace, point-in-time spec and permit queries, KG timeline and citations. R@5 is 0.952 (1.000 on point-in-time questions with the as-of view). The README lists its failures, such as *thickened* vs *thickness*.

### Changed — default analyzer: Porter stemming + English stop-words
- **New `porter+english` default analyzer.** Index and query terms are Porter-stemmed with a new pure-stdlib, deterministic stemmer (`locus.memory.stemmer`). A larger English function-word stop list is also applied. It matches SQLite FTS5's `porter` tokenizer on 38,997 of the 38,998 distinct alphabetic words in BEIR SciFact + NFCorpus, so the `fts5` and `legacy` backends still rank identically. No new dependencies.
- **Other analyzers.** `porter` (stemming only) and `plain` (the previous behaviour) are available through `LocusEngine(analyzer=...)`, `Corpus(analyzer=...)`, `LOCUS_ANALYZER`, `locus reindex --analyzer` and `LocusEngine.reindex(analyzer=...)`.
- **Store migration.** The analyzer is stored in the index. Stores without one, i.e. those written before this change, are re-indexed from stored chunk text with the new default the first time they are opened. To keep the old behaviour, open them with `analyzer="plain"`.
- **Why.** Measured on test splits after selecting on dev splits only (SciFact train, NFCorpus dev), with a paired bootstrap over the same queries:
  - nDCG@10 changes vs `plain`: SciFact 0.668 → 0.681 (+0.013, not significant) and NFCorpus 0.306 → 0.329 (+0.022, significant).
  - LongMemEval-S R@5 vs `plain`: 0.964 → 0.977 (+0.013, significant).
  - vs LlamaIndex BM25 (bm25s): NFCorpus +0.006 (significant), SciFact −0.006 (not significant).
- **BM25 parameters.** k1/b were swept on dev (k1 0.6–2.0 × b 0.3–0.9). The best setting gained ≤ 0.002 nDCG@10 for `porter+english`, and the best parameters disagreed between datasets. FTS5's `bm25()` fixes k1=1.2, b=0.75, so these values are unchanged.
- **Latency.** Query latency is 1.23× `plain` on SciFact and 1.01× on NFCorpus (p50, interleaved in one process). Index build time is about +6–12%.
- **API behaviour.** `top_terms()` / `inspect_doc()` now report index terms (stems), and `get_posting_list()` / `term_chunk_frequency()` map a plain word to its index term.

### Changed — retrieval engine (FTS5 + honest benchmarks)
- **SQLite FTS5 index is the default backend.** BM25 (k1=1.2, b=0.75) is scored inside SQLite with `bm25()`; no new dependencies. If the interpreter's SQLite lacks FTS5, Locus falls back to the pure-Python `legacy` backend, which uses the same formula and gives identical rankings. Choose explicitly with `LocusEngine(index_backend=...)` or `LOCUS_INDEX_BACKEND=fts5|legacy`.
- **Stores migrate automatically.** A 1.0.0 store is rebuilt into FTS5 on first open (`meta.schema_version = 2`) and the old `term_index` is cleared. New `locus reindex [--backend fts5|legacy]` / `LocusEngine.reindex()` rebuilds or switches backends. Migration is one-way: to go back to 1.0.0, index into a fresh store (1.0.0 sees an empty term index in a migrated store).
- **BM25 is the default retrieval mode.** The KG, link-walk, structural, recency and link-popularity signals are now an optional re-rank of the top `rerank_depth` (default 100) BM25 hits with weight `rerank_alpha` (default 0.1): `retrieve(mode="hybrid")`, `LocusEngine(retrieval_mode="hybrid")`, `LOCUS_RETRIEVAL_MODE=hybrid`, `locus retrieve --mode hybrid`, or `mode` on the `locus_retrieve` MCP tool. `as_of=` implies hybrid. Defaults were chosen on BEIR dev splits; on the test splits hybrid was not significantly better than BM25 (paired bootstrap), so it is opt-in.
- **Reads don't write.** `retrieve()` / `stream_retrieve()` no longer post to the bulletin or update the budget unless `track=True`. The `locus_retrieve` MCP tool keeps `track=True` by default; pass `track=false` for read-only calls.
- **Deterministic results.** Ties are broken by index order, file globs are sorted, KG queries have `ORDER BY`, query-type detection no longer depends on `PYTHONHASHSEED`, and RRF ties break on chunk id. Repeated runs in fresh processes return identical rankings.
- **Incremental indexing is cheaper.** Only changed files are re-extracted into the KG; their old auto-extracted triples are replaced instead of duplicated, and manual facts are kept. `forget()` also removes the document's extracted triples (`removed_triples` in the result). Duplicate chunk ids within a file keep the last occurrence. KG triples from 1.0.0 stores are treated as manual, so re-indexing a changed file may duplicate them once; indexing into a fresh store gives a clean graph.
- **Faster at scale (shared 8-vCPU VM).** 100k synthetic docs: index build 21.7 min → 34 s, store 1,324 → 419 MB, BM25 p50 19.0 s → 93 ms. BEIR SciFact p50 878 ms → 2.7 ms.
- **Ablation configs renamed** to match the new pipeline: `full_6signal` → `hybrid_all`, `full_reranked` → `hybrid_reranked`; the baseline runner reports `locus_bm25` and `locus_hybrid`. `locus.bench.beir` scores both modes.

### Fixed — benchmark claims
- **Removed unreproduced numbers.** The README's Engineering Wiki claim (six-signal R@5 0.947 / MRR 0.633) reproduced at 0.867 / 0.594 on 1.0.0 and has been replaced with BEIR SciFact/NFCorpus nDCG@10 with 95% CIs against `rank_bm25`, scaling to 100k docs, and a 5-run determinism check. Locus BM25 is at parity with `rank_bm25` (SciFact 0.668 vs 0.664, NFCorpus 0.306 vs 0.308; differences not significant) and beats the 1.0.0 default by +0.087 / +0.022.
- **`benchmarks/README.md`** documents datasets, protocol and the exact commands; scripts are in `benchmarks/repro/`, summaries in `benchmarks/published/2026-10-20/`.
- **uvx note.** The 1.0.0 wheel has no `locus-rag` script; from 1.1.0 `uvx locus-rag mcp` (as in `server.json`) works. On 1.0.0 use `uvx --from locus-rag locus mcp`.

### Security
- **HTTP server binds `127.0.0.1` by default** (was `0.0.0.0`). Binding any non-loopback host now requires `--token`; `serve()` / `make_server()` raise `ValueError` otherwise. Token comparison is constant-time. `/tools` now requires the token when one is set.
- **DNS-rebinding guard.** Without a token, the HTTP server only serves requests whose `Host` header is loopback, and any browser `Origin` must be loopback or the `--cors-origin` value (403 otherwise). This stops a page on an attacker hostname that re-resolves to 127.0.0.1 from calling tools such as `locus_index` / `locus_export_kg`.
- **No wildcard CORS.** CORS headers are only sent for an explicit `--cors-origin`; `*` is rejected. `POST /rpc` requires `Content-Type: application/json`.
- **Safe snapshot restore.** Extraction uses `tarfile` `filter="data"` where available; archives with absolute, `..`, symlink, hardlink or device members are rejected. `overwrite=True` only wipes an empty directory or an existing Locus store, never the filesystem root or home directory. The `locus_restore` MCP tool only restores to a relative path inside the server's working directory.

### Fixed
- **MCP stdio server honours `--store`.** Tool calls, resources and prompts use the configured store unless a call passes `store_path`; `locus --store X mcp` now works too. The HTTP server uses the engine passed to `serve()`.
- **JSON-RPC:** `ping` returns an empty result; unknown methods with an `id` get a `-32601 Method not found` error (notifications still get no response). The HTTP transport answers `ping` too.
- **OMPA bridge** read a non-existent `source` column (OMPA uses `source_file`) and silently imported 0 triples. It now maps `source_file` → `source`, checks the schema before indexing, and raises `OMPASchemaError` on a mismatch (the MCP tool returns the message).
- **Docs drift:** README version, test count, tool count (47) and full tool table now match the code; tests keep the tool count, version and `server.json` in sync. Install lines show macOS/Linux and Windows forms.

### Added
- **`server.json`** — MCP registry manifest (PyPI `locus-rag`, stdio, `uvx locus-rag mcp --store PATH`) and an `mcp-name` marker in the README.
- **`locus-rag` console script** (alias of `locus`) so `uvx locus-rag ...` works; the `mcp.servers` entry point now parses `--store`.
- **`locus.bench.beir`** — scores Locus and `rank_bm25` on a local BEIR-format dataset (nDCG@10, Recall@k, MRR). No downloads; the embedding baseline is a `NotImplementedError` stub (ROADMAP N3).
- **`docs/PRD.md`, `docs/ROADMAP.md`** — Locus as the retrieval half of the OMPA + Locus MCP bundle.
- **`LocusEngine.stream_retrieve`** — runs the six existing retrieval signals one at a time and yields new chunks as each signal returns, with `StreamingRetriever` time-budget dropout. Not RRF; order can differ from `retrieve()`.
- **`examples/mcp_sales_demo.py`** — end-to-end MCP sales/consulting demo: sample notes → `locus_index` / `locus_retrieve` / `locus_explain` / KG / doctor via the same tool dispatcher MCP clients use (stdlib + locus only)
- **README Companion stack** — honest roles for Locus (retrieval), OMPA (vault memory), CognitionOS (compliance product)
- **README Benchmark Results** — published Engineering Wiki ablation + baseline numbers (from `benchmarks/README.md`); documents how to re-run; notes that a tiny vector baseline is not yet published


---
## [1.0.0] — 2026-05-08

### Added
- **`DocSimilarity`** — TF-IDF cosine similarity between corpus documents; `similar_to(doc_path, limit)` returns nearest neighbours; `similarity_matrix()` returns all pairwise scores; pure stdlib, built on the existing term index
- **`CorpusDiff`** — compare current corpus state against the filesystem without modifying anything; classifies files as `new`, `changed`, `deleted`, or `unchanged`; `has_changes()` fast boolean check
- **`AnnotationStore`** — SQLite-backed chunk annotations; attach arbitrary labels and notes to any chunk; `annotate()`, `remove()`, `get()`, `chunks_with_label()`, `all_labels()`, `clear_chunk()`, `stats()`
- **`RelevanceFeedback`** — SQLite-backed relevance signals; `mark(chunk_id, query, relevant)` persists per-(chunk, query) signals; `score_adjustment()` returns `+0.25` / `-0.40`; wired into `LocusReranker` as a 4th boost
- **`LocusReranker`** updated: four-boost formula — title + entity density + freshness + **feedback**; `feedback` param on constructor
- **`engine.related_docs(doc_path, limit)`** — surface method for `DocSimilarity.similar_to()`
- **`engine.similarity_matrix()`** — surface method for `DocSimilarity.similarity_matrix()`
- **`engine.diff(path, pattern)`** — surface method for `CorpusDiff.diff()`
- **`engine.annotate()`, `get_annotations()`, `chunks_with_label()`** — annotation surface methods
- **`engine.mark_relevant()`, `engine.mark_irrelevant()`** — feedback surface methods
- `engine.status()` now includes `annotations` and `feedback` stats
- New CLI commands: `related`, `diff`, `annotate`, `annotations`, `mark-relevant`, `mark-irrelevant`
- 7 new MCP tools (47 total): `locus_related_docs`, `locus_diff`, `locus_annotate`, `locus_get_annotations`, `locus_chunks_with_label`, `locus_mark_relevant`, `locus_mark_irrelevant`
- `DocSimilarity`, `CorpusDiff`, `AnnotationStore`, `RelevanceFeedback` exported from top-level `locus` package
- 27 new tests (254 total, all passing)

### Changed
- **`__version__` bumped to `1.0.0`** — first stable release milestone
- `pyproject.toml` version updated to `1.0.0`

---

## [0.9.0] — 2026-05-08

### Added
- **`QueryExpander`** — KG-backed query enrichment; resolves alias tokens in the query to their canonical names and appends first-hop KG neighbours as additional search terms; zero-cost when KG is empty
- **`LocusSnapshot`** — portable save/restore for a full Locus store; `save()` packs the store to a `.tar.gz` archive; `load()` extracts to any target path with optional overwrite; `inspect()` shows the manifest without extracting; path-traversal-safe
- **`engine.expand_query(query, max_expansions)`** — surface method for `QueryExpander`
- **`engine.multi_retrieve(queries, limit, as_of)`** — run N query variants through the full six-signal pipeline and fuse all result lists via RRF; deduplicates by `chunk_id`
- **`engine.timeline(entity, as_of)`** — chronological KG view of any entity; facts sorted by `valid_from`; undated facts last; temporal filter supported
- **`engine.snapshot(output_path)`** — convenience wrapper for `LocusSnapshot.save()`
- New CLI commands: `expand`, `multi-retrieve`, `timeline`, `snapshot`, `restore`, `snapshot-info`
- 6 new MCP tools (40 total): `locus_expand_query`, `locus_multi_retrieve`, `locus_timeline`, `locus_snapshot`, `locus_restore`, `locus_snapshot_info`
- `QueryExpander`, `ExpansionResult`, `LocusSnapshot` exported from top-level `locus` package
- 20 new tests (227 total, all passing)

### Changed
- `__version__` bumped to `0.9.0`
- `pyproject.toml` version updated to `0.9.0`

---

## [0.8.0] — 2026-05-08

### Added
- **`AsyncLocusEngine`** — async façade over `LocusEngine` backed by a `ThreadPoolExecutor`; async context manager (`async with`); `__getattr__` pass-through for all sync methods
- **`LocusHooks`** — lightweight event hook system; `on()` decorator, `register()`, `unregister()`, `fire()`, `list_hooks()`, `clear()`; handlers are error-isolated
- **`LocusReasoner`** — multi-hop KG reasoning; `find_paths(a, b, max_depth, predicate_filter)` via BFS; `reason(question)` extracts entities and surfaces connecting chains
- **`GitHubBridge`** — ingest markdown from any GitHub repository; fetches file tree + raw content via GitHub REST API; supports private repos with token or `GITHUB_TOKEN` env var
- **`Corpus.top_terms(limit)`** — corpus-wide top terms ranked by document frequency
- **`Corpus.inspect_doc_terms(doc_path, limit)`** — per-document top terms by chunk coverage
- **`LocusEngine.reason(question, max_depth)`** — delegate to `LocusReasoner.reason()`
- **`LocusEngine.find_paths(entity_a, entity_b, ...)`** — delegate to `LocusReasoner.find_paths()`
- **`LocusEngine.inspect_doc(doc_path, limit)`** — document stats: chunk count, word count, top terms, entities, wikilinks
- **`LocusEngine.top_terms(limit)`** — corpus-wide term frequency report
- **`LocusEngine.set_hooks(hooks)` / `LocusEngine._fire(event)`** — hook integration; fires `pre/post_index`, `pre/post_retrieve`, `pre/post_forget`, `pre/post_add_fact`
- New CLI commands: `reason`, `find-paths`, `inspect`, `top-terms`, `ingest-github`
- New MCP tools: `locus_reason`, `locus_find_paths`, `locus_inspect`, `locus_top_terms`, `locus_ingest_github` (total: 34)
- All new classes exported from top-level `locus` package: `AsyncLocusEngine`, `LocusHooks`, `LocusReasoner`, `GitHubBridge`
- 21 new tests (207 total, all passing)

### Changed
- `__version__` bumped to `0.8.0`
- `pyproject.toml` version updated to `0.8.0`

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
