<!-- mcp-name: io.github.jmiaie/locus -->
<div align="center">

# Locus

### Vectorless RAG &nbsp;·&nbsp; MCP-Native &nbsp;·&nbsp; Zero Dependencies

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-301%20passing-brightgreen?logo=pytest&logoColor=white)](tests/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)](pyproject.toml)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-blueviolet)](https://modelcontextprotocol.io)
[![47 MCP Tools](https://img.shields.io/badge/MCP%20tools-47-blueviolet)](#mcp-tools)
[![CI](https://github.com/jmiaie/locus/actions/workflows/ci.yml/badge.svg)](https://github.com/jmiaie/locus/actions)

**Retrieval-Augmented Generation without vectors, without GPUs, without black boxes.**

Every result tells you *exactly* why it was returned.

[Quick Start](#quick-start) &nbsp;·&nbsp;
[MCP Demo](#mcp-sales-demo) &nbsp;·&nbsp;
[Architecture](#architecture) &nbsp;·&nbsp;
[MCP Tools](#mcp-tools) &nbsp;·&nbsp;
[Benchmark Results](#benchmark-results) &nbsp;·&nbsp;
[Companion Stack](#companion-stack) &nbsp;·&nbsp;
[CLI Reference](#cli-reference)

</div>

---

## Why Vectorless?

Standard RAG pipelines require embedding models, vector databases, and GPU compute. The retrieval signal — cosine similarity — is a single opaque float that tells you nothing about *why* a document was returned. Locus takes a different path:

| | Vector RAG | **Locus** |
|:---|:---|:---|
| **Dependencies** | sentence-transformers, faiss / pinecone / chroma | **Zero** |
| **GPU required** | Yes (or paid embedding API) | **No** |
| **Why was this returned?** | `"similarity = 0.82"` | `"BM25: [jwt, auth]; KG: Alice links to doc"` |
| **Temporal queries** | Not supported | **Yes — validity windows on every KG fact** |
| **Knowledge graph** | Not supported | **Auto-extracted from prose** |
| **Contradiction detection** | Not supported | **Built in** |
| **Multi-node** | Requires external infrastructure | **Built-in cluster** |
| **Deterministic results** | No (model-dependent) | **Yes** — ties broken by index order, no hash-order dependence, retrieval never writes |

---

## Retrieval Pipeline

Locus retrieves in two stages. Stage 1 is always BM25 over a **SQLite FTS5** inverted index
(stdlib `sqlite3`, no extra packages). Stage 2 is an **optional** re-rank of the BM25 candidates
with five graph/metadata signals, fused with weighted Reciprocal Rank Fusion.

```
  Query ──► Stage 1: BM25 (SQLite FTS5)  ──► top-k chunks          mode="bm25"  (default)
                     │
                     └─► candidate pool (rerank_depth=100)
                           ├─ KG            entity expansion + one hop (honours as_of)
                           ├─ Link Walk     wikilinks from the top BM25 hits
                           ├─ Structural    frontmatter date / tag / type
                           ├─ Recency       frontmatter date, newest first
                           └─ Link Popularity  inbound wikilinks
                         weighted RRF (intent-aware, rerank_alpha=0.1) ─► top-k   mode="hybrid"
```

**Why BM25 is the default:** on the public BEIR SciFact and NFCorpus test sets the hybrid re-rank
did **not** beat plain BM25 (paired bootstrap, see [Benchmark Results](#benchmark-results)), and every
larger re-rank weight we tried on the dev splits made nDCG@10 worse. The re-rank stays available for
link- and frontmatter-rich vaults: `engine.retrieve(q, mode="hybrid")`,
`LocusEngine(retrieval_mode="hybrid")`, `LOCUS_RETRIEVAL_MODE=hybrid`, `locus retrieve --mode hybrid`,
or `mode` on the `locus_retrieve` MCP tool. Passing `as_of` selects hybrid, since the temporal filter
acts on KG facts.

| Stage-2 signal | Weight (× `rerank_alpha`) | What it captures |
|:---|:---|:---|
| KG Entity | intent-weighted | Entities from the query linked to documents |
| Link Walk | intent-weighted | Candidates reachable via wikilinks from the top hits |
| Structural | 1.0 | Frontmatter date ranges, tags, doc types |
| Recency | 0.3 | Newer frontmatter `date` first |
| Link Popularity | 0.2 | Hub documents cited by many others |

Retrieval is **read-only**: it never writes to the store unless you pass `track=True`, which records
hits in the hot-context bulletin and token budget (the `locus_retrieve` MCP tool does this by default
so `locus_hot_context` keeps working; pass `track: false` for a pure read).

---

## Quick Start

```bash
pip install locus-rag     # macOS / Linux: python3 -m pip install locus-rag · Windows: py -3 -m pip install locus-rag
```

**Index and retrieve in three commands:**

```bash
# Index a directory of markdown files
locus index ./my-docs

# Retrieve — every result includes a provenance tag
locus retrieve "how does authentication work?"
# [1] auth/jwt.md  score=7.4121  via=bm25
#     JWT tokens are validated on every request. The token is...
# [2] auth/oauth.md  score=5.0318  via=bm25
#     OAuth 2.0 flow for third-party integrations...

# Optional graph/metadata re-rank of the BM25 candidates
locus retrieve "who owns the auth service?" --mode hybrid

# Explain why a specific result was returned
locus explain <chunk_id> --query "authentication"
# {
#   "narrative": "Chunk from 'auth/jwt.md' (section: 'JWT Validation').
#                 Retrieved because: BM25: terms [jwt, authentication]
#                 appear in this chunk; KG: entity Alice links to this
#                 document via 3 KG triples.",
#   "bm25_matched_terms": ["jwt", "authentication"],
#   "kg_matched_entities": ["Alice"],
#   ...
# }
```

**As a Python library:**

```python
from locus import LocusEngine

engine = LocusEngine(store_path=".locus")
engine.index("./my-docs")

chunks = engine.retrieve("JWT authentication flow", limit=5)
for chunk in chunks:
    print(f"{chunk.doc_path}  via={chunk.provenance}")
    print(chunk.content[:200])

# Explain a result
print(engine.explain(chunks[0].chunk_id, query="JWT")["narrative"])
```

**As an MCP server** (Claude Desktop / Claude Code / Cursor / Windsurf):

```bash
# macOS / Linux
claude mcp add locus -- python3 -m locus.mcp.server --store /path/to/.locus

# Windows (Python launcher)
claude mcp add locus -- py -3 -m locus.mcp.server --store C:\path\to\.locus

# Any OS, via the installed console script
claude mcp add locus -- locus mcp --store /path/to/.locus
```

Tool calls use the `--store` you configured; a call can still target another store by passing `store_path`.

Then ask Claude: *"What does the codebase say about the auth system?"* — Locus retrieves context, Claude answers.

### MCP sales demo

Run a deck-ready end-to-end walkthrough (sample notes → MCP `locus_index` / `locus_retrieve` / `locus_explain` / KG / doctor) with **zero extra deps**:

```bash
pip install -e .          # or: pip install locus-rag
python examples/mcp_sales_demo.py
python examples/mcp_sales_demo.py --keep   # leave the temp store for inspection
```

The script calls the same `locus.mcp.server` tool dispatcher an MCP client would hit — useful for consulting demos without wiring Claude Desktop first. See [`examples/mcp_sales_demo.py`](examples/mcp_sales_demo.py).

---

## Architecture

Locus is three layers — memory, retrieval, and interface — with no layer depending upward:

```
┌─────────────────────────────────────────────────────────────────┐
│  Interface Layer                                                  │
│  ┌────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │  CLI (locus)   │  │  MCP stdio server│  │  HTTP JSON-RPC  │  │
│  │  41 commands   │  │  47 tools        │  │  POST /rpc      │  │
│  │                │  │  4 prompts       │  │  Bearer token   │  │
│  │                │  │  resources       │  │  CORS opt-in    │  │
│  └────────────────┘  └──────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Retrieval Layer                                                  │
│  BM25Retriever  ·  KGRetriever  ·  LinkWalker                    │
│  StructuralRetriever  ·  RecencyRetriever  ·  LinkPopularity     │
│  ──────────────────────────────────────────────────────────────  │
│  RRF Fusion  ·  QueryClassifier  ·  ContextBulletin              │
│  TokenBudget  ·  LocusCluster (multi-node)                       │
├─────────────────────────────────────────────────────────────────┤
│  Memory Layer                                                     │
│  Corpus         SQLite FTS5 BM25 index, checksum dedup, SQLite   │
│  TemporalKG     Triples + validity windows, traversal, match     │
│  EntityResolver Alias table → canonical entity names            │
│  Chunker        Section-aware (heading boundaries) + word-count  │
└─────────────────────────────────────────────────────────────────┘
```

All storage is SQLite — no external databases, no servers, no configuration.

---

## Features

### Retrieval
- **BM25 on SQLite FTS5** — inverted index, so query cost tracks the matching postings, not the corpus size (100k docs: ~93 ms p50 vs ~19 s before); pure-Python fallback when FTS5 is not compiled in
- **Optional hybrid re-rank** — KG, link walk, structural, recency and link popularity over the BM25 candidates, fused via weighted RRF (`mode="hybrid"`)
- **Incremental indexing** — changed files replace their chunks, postings and auto-extracted KG triples in place; `forget` removes them; `add_fact` facts are never touched
- **Deterministic, read-only retrieval** — ties broken by index order; nothing is written unless `track=True`
- **Section-aware chunking** — splits at `#` heading boundaries; long sections fall back to word-count
- **Temporal retrieval** — `as_of="2024-01-01"` returns only facts valid on that date
- **Query intent routing** (hybrid) — auto-classifies `kg_first` / `bm25_first` / `balanced` and shifts re-rank weights
- **Checksum dedup** — unchanged files never re-indexed; safe to run `index` on every startup

### Knowledge Graph
- **Prose triple extraction** — 14 relation patterns (`leads`, `works_at`, `is_a`, `part_of`, `replaced`, `depends_on`...) auto-run during indexing — no annotation required
- **Entity resolution** — alias table maps name variants to canonical entities transparently
- **Contradiction detection** — surfaces same-predicate triples with different objects and overlapping validity windows
- **BFS traversal** — `kg.traverse("Alice", max_depth=2)` returns the full entity subgraph
- **Pattern matching** — `kg.match("*", "works_at", "Acme")` finds everyone at Acme; `*` is a wildcard
- **KG export** — GraphML (Gephi/Cytoscape), JSONL (scripting), DOT (Graphviz)

### Explainability
```python
engine.explain(chunk_id, query="authentication")
# {
#   "bm25_matched_terms": ["jwt", "authentication"],
#   "kg_matched_entities": ["Alice"],
#   "structural_matches": ["date 2025-02-10 in range Q1 2025"],
#   "kg_context": [{"entity": "Alice", "fact_count": 3, "sample_fact": "Alice --leads--> Auth team"}],
#   "narrative": "Chunk from 'auth.md'. Retrieved because: BM25: terms [jwt,
#                 authentication]; KG: Alice links to document..."
# }
```
This is structurally impossible with vector search. No cosine score has a "why."

### MCP Integration
- **47 tools** spanning retrieval, KG operations, reasoning, bulletin management, cluster, evaluation, export, snapshots, annotations and relevance feedback
- **MCP Resources** — all indexed documents browsable as `locus://doc/{path}`
- **4 MCP Prompts** — live-context templates pre-filled with retrieved knowledge before sending to the model
- **HTTP transport** — `locus serve` binds `127.0.0.1` by default; any other host requires `--token <secret>`. No CORS headers unless you pass `--cors-origin <origin>`

### Memory Management
- **Tiered bulletin board** (Tier 0 Pinned / Tier 1 Hot / Tier 2 Archive) with hit-boost + age-decay scoring
- **Persistent hot tier** — `bulletin.sqlite3` survives process restarts
- **Soft token budget** — WARNING / TREND / CRITICAL alerts; never blocks retrieval
- **File watcher** — `locus watch ./docs` auto-reindexes changes; deletions detected

### Operations
- **Health diagnostics** — `locus doctor` checks corpus, KG, bulletin, store size, version
- **Multi-node cluster** — cross-corpus RRF fusion with node-prefixed provenance (`jarv:bm25`)
- **OMPA bridge** — imports OMPA vaults (markdown + KG triples; OMPA's `source_file` maps to Locus `source`)
- **Evaluation** — `locus benchmark qa.json` measures Recall@K and MRR

---

## MCP Tools

<details>
<summary><strong>All 47 tools</strong></summary>

| Tool | Description |
|:---|:---|
| `locus_index` | Index a file or directory; skips unchanged files |
| `locus_retrieve` | BM25 retrieval (FTS5) with optional hybrid re-rank — returns chunks with provenance tags |
| `locus_explain` | Explain *why* a chunk was retrieved for a query |
| `locus_add_fact` | Add a temporal KG triple (subject–predicate–object) |
| `locus_query_entity` | Query all KG facts about an entity; supports `as_of` |
| `locus_kg_stats` | Triple count, entity count, source count |
| `locus_kg_traverse` | BFS graph walk from a starting entity |
| `locus_kg_match` | Wildcard pattern match (`*` matches any) |
| `locus_contradictions` | Find conflicting triples (same predicate, different objects) |
| `locus_add_alias` | Register entity alias → canonical mapping |
| `locus_suggest_aliases` | Suggest entity name pairs that may be duplicates |
| `locus_hot_context` | Get the hot-tier bulletin board for context injection |
| `locus_promote` | Pin a chunk to Tier 0 (never auto-archived) |
| `locus_session_start` | Warm open: corpus + KG + bulletin stats + hot context |
| `locus_wrap_up` | Session close: tick bulletin decay, return summary |
| `locus_status` | Full store status: corpus, KG, bulletin, budget, resolver |
| `locus_forget` | Remove a document from the corpus |
| `locus_sync` | Full reindex: wipe and rebuild from directory |
| `locus_doctor` | Health diagnostics — PASS / WARN / FAIL per check |
| `locus_export_kg` | Export KG to GraphML / JSONL / DOT |
| `locus_ingest_ompa` | Import an OMPA vault (markdown + KG triples) |
| `locus_benchmark` | Recall@K + MRR evaluation against a QA JSON file |
| `locus_cluster_retrieve` | Cross-node retrieval fused via RRF |
| `locus_add_node` | Register a named node in the cluster |
| `locus_remove_node` | Unregister a node |
| `locus_list_nodes` | List nodes with corpus and KG stats |
| `locus_prepare_context` | Retrieve, re-rank and pack results into a token budget |
| `locus_cache_stats` | Query cache size, hit rate, hits, misses |
| `locus_clear_cache` | Invalidate the query result cache |
| `locus_reason` | Multi-hop KG reasoning over the entities in a question |
| `locus_find_paths` | Shortest KG paths between two entities |
| `locus_inspect` | A document's chunks, top terms, entities and links |
| `locus_top_terms` | Corpus-wide top terms by document frequency |
| `locus_ingest_github` | Fetch and index markdown from a GitHub repository |
| `locus_expand_query` | Expand a query via KG aliases and first-hop neighbours |
| `locus_multi_retrieve` | Run several query variants and fuse with RRF |
| `locus_timeline` | Chronological KG facts for an entity |
| `locus_snapshot` | Archive the store to a `.tar.gz` |
| `locus_restore` | Restore a store from a snapshot (MCP: relative target inside the working dir) |
| `locus_snapshot_info` | List a snapshot's files and sizes |
| `locus_related_docs` | Similar documents by TF-IDF cosine (no embeddings) |
| `locus_diff` | Preview new / changed / deleted files before re-indexing |
| `locus_annotate` | Attach a label and note to a chunk |
| `locus_get_annotations` | Labels and notes on a chunk |
| `locus_chunks_with_label` | Chunk IDs carrying a label |
| `locus_mark_relevant` | Persist a relevance boost for a chunk and query |
| `locus_mark_irrelevant` | Persist a relevance penalty for a chunk and query |

</details>

### MCP Prompts

Four templates rendered with live context before delivery to the model:

| Prompt | What it pre-fills |
|:---|:---|
| `locus_research` | Retrieved chunks for a topic |
| `locus_entity_summary` | All KG facts + document context for an entity |
| `locus_timeline` | Temporal fact diff for an entity between two dates |
| `locus_contradiction_analysis` | Contradictions framed for human resolution |

---

## CLI Reference

```
locus [--store PATH] COMMAND [args]
```

| Command | Description |
|:---|:---|
| `index <path>` | Index files; `--pattern **/*.md` |
| `retrieve <query>` | Retrieve; `--limit 5 --mode bm25\|hybrid --as-of 2024-01-01 --track --raw` |
| `explain <chunk_id>` | Explain retrieval; `--query "..."` |
| `status` | Corpus + KG + bulletin + budget stats |
| `session-start` | Warm context open |
| `wrap-up` | Session close + bulletin tick |
| `add-fact S P O` | Add KG triple; `--valid-from --valid-to --source` |
| `query-entity <name>` | KG entity facts; `--as-of` |
| `add-alias A B` | Register alias A → canonical B |
| `contradictions [entity]` | Find conflicting KG triples |
| `kg-traverse <start>` | BFS traversal; `--depth 2 --direction out` |
| `kg-match` | Pattern match; `--subject --predicate --obj --as-of` |
| `forget <doc_path>` | Remove document from corpus |
| `sync <path>` | Full reindex |
| `reindex` | Rebuild the search index from stored chunks; `--backend fts5\|legacy` |
| `watch <path>` | Auto-reindex on changes; `--interval 5` |
| `mcp` | Start MCP stdio server |
| `serve` | Start HTTP JSON-RPC server; `--host 127.0.0.1 --port 7391 --token --cors-origin` |
| `ingest-ompa <vault>` | Import OMPA vault |
| `benchmark <qa.json>` | Recall@K + MRR; `--k 1,3,5` |
| `doctor` | Health check report |
| `export-kg <path>` | Export KG; `--format graphml\|jsonl\|dot` |

---

## Multi-node Cluster

Locus has first-class support for running multiple named nodes — each indexing a different knowledge domain — with cross-node retrieval fused via RRF.

```python
from locus import LocusCluster

cluster = LocusCluster("cluster.json")
cluster.add_node("docs",    "/vaults/docs/.locus")
cluster.add_node("code",    "/vaults/code/.locus")
cluster.add_node("logs",    "/vaults/logs/.locus")

# Query all three simultaneously
chunks = cluster.retrieve("auth system deployment", limit=5)
for c in chunks:
    print(f"{c.provenance}  {c.doc_path}")
    # docs:bm25     auth/jwt.md
    # code:kg       src/auth/middleware.py.md
    # logs:recency  incidents/2025-02-auth-outage.md
```

Each result is tagged `node_name:signal` so you always know which knowledge base contributed which chunk. The registry persists across restarts — add a node once, always available.

---

## Companion stack

Locus is the **explainable retrieval** layer. Pair it with OMPA (memory) and CognitionOS (compliance product):

| Layer | Project | Role |
|---|---|---|
| Explainable retrieval | **Locus** ([`locus-rag`](https://pypi.org/project/locus-rag/)) | Vectorless RAG — BM25 + temporal KG + link walking, zero GPU, MCP-native |
| Vault memory | **[OMPA](https://github.com/jmiaie/ompa)** ([PyPI](https://pypi.org/project/ompa/)) | Vault · palace · temporal KG · agent lifecycle hooks |
| Compliance product | **[CognitionOS](https://github.com/jmiaie/cognition-os)** | Audit · RBAC · retention · PHI redaction on top of Locus + OMPA |

CognitionOS consumes Locus via git submodule / optional pip; keep the public `locus` API stable when changing this package.

## OMPA Bridge

If you already use [OMPA](https://github.com/jmiaie/ompa) as your agent memory layer, Locus can import your vault directly. OMPA's `triples` table (`source_file` column) is mapped onto the Locus KG (`source`). If the table is missing required columns the import raises `OMPASchemaError` rather than silently importing nothing.

```bash
locus ingest-ompa ~/path/to/ompa-vault --store .locus
```

```python
from locus.bridge.ompa import OMPABridge
bridge = OMPABridge(engine, vault_path="~/ompa-vault")
result = bridge.ingest()
# {"vault": "...", "chunks_indexed": ..., "triples_imported": ..., "kg_source": ".../.palace/knowledge_graph.sqlite3"}
```

Markdown files are indexed via the normal pipeline. KG triples from `.palace/knowledge_graph.sqlite3` are copied row by row after a schema check. Safe to re-run — checksum dedup prevents duplicate indexing.

---

## Health Diagnostics

```bash
locus doctor
```

```
Locus Doctor
============================================
[PASS]  corpus               47 docs | 203 chunks | avg 4.3 chunks/doc
[PASS]  knowledge_graph      1847 triples | 312 entities | 2 contradictions
[WARN]  bulletin             Tier 0 (pinned) is full — oldest entries will be demoted
[PASS]  entity_resolver      14 aliases registered
[PASS]  store_size           3.21 MB at .locus
[PASS]  version              Locus v1.0.0

Result: 5 pass  1 warn  0 fail
```

---

## Benchmark Results

Every number below is reproducible with the scripts in [`benchmarks/repro/`](benchmarks/repro/);
methodology, data sources and caveats are in [`benchmarks/README.md`](benchmarks/README.md).
Measured 2026-10-20 on a shared 8-vCPU / 16 GB Linux VM (Python 3.13, SQLite 3.46.1) — treat latencies as
indicative. 95% CIs are seeded bootstrap (B = 10,000) over queries.

### Public datasets — BEIR test splits, nDCG@10

| System | SciFact (300 q) | NFCorpus (323 q) | p50 latency (SciFact / NFCorpus) |
|---|---|---|---|
| **Locus default (`mode="bm25"`, FTS5)** | **0.668** [0.623, 0.713] | **0.306** [0.272, 0.341] | 2.7 ms / 1.2 ms |
| Locus hybrid (`mode="hybrid"`) | 0.669 [0.624, 0.714] | 0.306 [0.272, 0.341] | 7.7 ms / 4.6 ms |
| `rank_bm25` BM25Okapi (same tokenizer) | 0.664 [0.618, 0.709] | 0.308 [0.274, 0.343] | 7.2 ms / 1.6 ms |
| *Locus 1.0.0 six-signal pipeline (before)* | *0.581* [0.538, 0.624] | *0.284* [0.251, 0.318] | *878 ms / 404 ms* |

- **Parity with `rank_bm25`:** paired difference +0.0045 [−0.0015, +0.0111] on SciFact and −0.0017 [−0.0043, +0.0009] on NFCorpus — not significant either way.
- **Hybrid vs BM25:** +0.0007 [−0.0056, +0.0080] (SciFact) and −0.0001 [−0.0012, +0.0009] (NFCorpus) — no significant gain, so BM25 is the default.
  The re-rank weight and depth were chosen on the dev splits only (SciFact train, NFCorpus dev); no larger weight helped there.
- **vs Locus 1.0.0:** +0.0872 [+0.0636, +0.1111] (SciFact) and +0.0220 [+0.0138, +0.0305] (NFCorpus) nDCG@10 over the old six-signal default; +0.0278 / +0.0124 over the old BM25-only path (both significant).
- **Monte Carlo, 10 random 2,000-doc SciFact sub-corpora:** mean nDCG@10 BM25 0.742 · `rank_bm25` 0.742 · hybrid 0.736 (hybrid lower in 8 of 10 seeds).

### Scaling — synthetic corpus (Zipf vocabulary, known-item queries, 100 queries)

| Docs | Index build | Store size | Query p50 / p95 (default) | Before (1.0.0): build · size · p50 |
|---|---|---|---|---|
| 1,000 | 0.24 s | 4.4 MB | 1.5 ms / 2.2 ms | 6.0 s · 13 MB · 207 ms |
| 10,000 | 2.5 s | 43 MB | 10.5 ms / 16.3 ms | 85 s · 130 MB · 1.36 s |
| 100,000 | 34 s | 419 MB | 93 ms / 137 ms | 21.7 min · 1,324 MB · 19.0 s |

Query cost grows with the size of the matching posting lists: the synthetic queries include very common
words, so FTS5 still scores most of the corpus. p95 at 100k is above 100 ms on this machine.

### Determinism and the old wiki claims

The 1.0.0 README reported R@5 0.947 / MRR 0.633 for the six-signal pipeline on the bundled Engineering
Wiki benchmark. That did not reproduce (0.867 / 0.594), and the simple-100 benchmark gave three different
answers across five runs. With this release both benchmarks give identical results in 5 of 5 fresh
processes with different `PYTHONHASHSEED` values:

| Benchmark (`benchmarks/run.py`) | System | R@5 | MRR |
|---|---|---|---|
| Engineering Wiki (113 queries, synthetic) | Locus default (BM25) | 1.000 | 0.727 |
| | Locus hybrid | 1.000 | 0.743 |
| | `rank_bm25` | 1.000 | 0.760 |
| simple-100 (synthetic) | Locus default / hybrid | 0.600 | 0.274 |
| | `rank_bm25` | 0.590 | 0.271 |

These two corpora are generated by Locus's own code and are small, so treat them as smoke tests rather than evidence;
the BEIR numbers above are the ones to quote.

## Benchmarking

Create a QA JSON file and run evaluation against your indexed corpus:

```json
[
  {"query": "how does JWT authentication work?",    "expected_docs": ["auth/jwt.md"]},
  {"query": "incident response runbook",            "expected_docs": ["ops/ir.md"]},
  {"query": "Q1 2025 architecture decisions",       "expected_docs": ["adrs/q1-2025.md"]}
]
```

```bash
locus benchmark qa.json --k 1,3,5
```

```
Locus Benchmark
========================================
Queries:  20

Recall@1: 0.650  (13/20)
Recall@3: 0.800  (16/20)
Recall@5: 0.850  (17/20)
MRR:      0.723

Misses (top 5):
  [MISS] "incident response process" expected=['ir.md'] got=['deploy.md', ...]
```

---

## KG Export & Visualization

```bash
locus export-kg knowledge.graphml   # Gephi / Cytoscape / yEd
locus export-kg knowledge.dot       # Graphviz → PNG/PDF
locus export-kg knowledge.jsonl     # scripting / grep
```

```bash
# Render a quick PNG with Graphviz
dot -Tpng knowledge.dot -o knowledge.png
```

---

## Installation

```bash
pip install locus-rag          # macOS / Linux: python3 -m pip install locus-rag · Windows: py -3 -m pip install locus-rag
```

Locus is also described for the MCP registry in [`server.json`](server.json) (`uvx locus-rag mcp --store PATH`).
The `locus-rag` console script that command needs is not in the 1.0.0 wheel on PyPI; until the next release,
run `uvx --from locus-rag locus mcp` instead (verified against 1.0.0).

**Zero runtime dependencies.** Locus uses only Python stdlib: `sqlite3`, `re`, `pathlib`, `threading`, `http.server`, `hashlib`, `math`, `json`.

Optional extras:

```bash
pip install "locus-rag[ompa]"   # OMPA vault bridge
pip install "locus-rag[dev]"    # pytest, coverage
pip install "locus-rag[bench]"  # rank_bm25 baseline for benchmarks
```

**Requirements:** Python 3.11+

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome — bug reports, feature requests, PRs.

```bash
git clone https://github.com/jmiaie/locus
cd locus
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

Locus is architecturally descended from [OMPA](https://github.com/jmiaie/ompa) (Obsidian-MemPalace-Agnostic), which pioneered the three-layer vault → palace → KG design for agent memory. The tiered bulletin board, temporal KG schema, and session lifecycle patterns are direct adaptations. The OMPA bridge enables seamless import of existing OMPA vaults. [CognitionOS](https://github.com/jmiaie/cognition-os) productizes Locus + OMPA into a compliance SaaS (audit, RBAC, retention) — see [Companion stack](#companion-stack).
