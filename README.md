<div align="center">

# Locus

### Vectorless RAG &nbsp;·&nbsp; MCP-Native &nbsp;·&nbsp; Zero Dependencies

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.6.0-blue)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-167%20passing-brightgreen?logo=pytest&logoColor=white)](tests/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)](pyproject.toml)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-blueviolet)](https://modelcontextprotocol.io)
[![26 MCP Tools](https://img.shields.io/badge/MCP%20tools-26-blueviolet)](#mcp-tools)
[![CI](https://github.com/jmiaie/locus/actions/workflows/ci.yml/badge.svg)](https://github.com/jmiaie/locus/actions)

**Retrieval-Augmented Generation without vectors, without GPUs, without black boxes.**

Every result tells you *exactly* why it was returned.

[Quick Start](#quick-start) &nbsp;·&nbsp;
[Architecture](#architecture) &nbsp;·&nbsp;
[MCP Tools](#mcp-tools) &nbsp;·&nbsp;
[CLI Reference](#cli-reference) &nbsp;·&nbsp;
[Cluster](#multi-node-cluster) &nbsp;·&nbsp;
[OMPA Bridge](#ompa-bridge)

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
| **Deterministic results** | No (model-dependent) | **Yes** |

---

## Six Retrieval Signals

Locus fuses six independent signals via **Reciprocal Rank Fusion**. Query intent is auto-classified and adjusts signal weights per query.

```
                        ┌───────────────────────────────────────────┐
  Query ──────────────► │  1. BM25          keyword ranking (BM25+) │
                        │  2. KG            entity expansion + hops  │
  Intent classifier     │  3. Link Walk     wikilink graph traversal  │
  ┌────────────────┐    │  4. Structural    date / tag / type match   │
  │ kg_first       │    │  5. Recency       freshness decay prior     │
  │ bm25_first     │    │  6. Link Popularity hub docs (inbound links)│
  │ balanced       │    └──────────────────────┬────────────────────┘
  └────────────────┘                           │
                                   Weighted RRF Fusion
                                               │
                              ┌────────────────▼──────────────────┐
                              │  Ranked chunks + provenance tags  │
                              │  bm25 · kg · link:hop1 · structural│
                              │  recency · link_pop               │
                              └───────────────────────────────────┘
```

| Signal | Weight | What it captures |
|:---|:---|:---|
| BM25 | intent-weighted | Keyword frequency and term importance |
| KG Entity | intent-weighted | Entities from query linked to documents |
| Link Walk | intent-weighted | Documents reachable via wikilinks from top hits |
| Structural | 1.0 | Frontmatter date ranges, tags, doc types |
| Recency | 0.3 | Freshness prior (half-life: 30 days) |
| Link Popularity | 0.2 | Hub documents cited by many others |

---

## Quick Start

```bash
pip install locus-rag
```

**Index and retrieve in three commands:**

```bash
# Index a directory of markdown files
locus index ./my-docs

# Retrieve — every result includes a provenance tag
locus retrieve "how does authentication work?"
# [1] auth/jwt.md  score=0.0421  via=bm25
#     JWT tokens are validated on every request. The token is...
# [2] auth/oauth.md  score=0.0318  via=kg
#     OAuth 2.0 flow for third-party integrations...

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
claude mcp add locus -- py -3 -m locus.mcp.server --store /path/to/.locus
```

Then ask Claude: *"What does the codebase say about the auth system?"* — Locus retrieves context, Claude answers.

---

## Architecture

Locus is three layers — memory, retrieval, and interface — with no layer depending upward:

```
┌─────────────────────────────────────────────────────────────────┐
│  Interface Layer                                                  │
│  ┌────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │  CLI (locus)   │  │  MCP stdio server│  │  HTTP JSON-RPC  │  │
│  │  26 commands   │  │  26 tools        │  │  POST /rpc      │  │
│  │                │  │  4 prompts       │  │  Bearer token   │  │
│  │                │  │  resources       │  │  CORS headers   │  │
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
│  Corpus         BM25 inverted index, checksum dedup, SQLite      │
│  TemporalKG     Triples + validity windows, traversal, match     │
│  EntityResolver Alias table → canonical entity names            │
│  Chunker        Section-aware (heading boundaries) + word-count  │
└─────────────────────────────────────────────────────────────────┘
```

All storage is SQLite — no external databases, no servers, no configuration.

---

## Features

### Retrieval
- Six-signal pipeline fused via weighted Reciprocal Rank Fusion
- **Section-aware chunking** — splits at `#` heading boundaries; long sections fall back to word-count
- **Temporal retrieval** — `as_of="2024-01-01"` returns only facts valid on that date
- **Query intent routing** — auto-classifies `kg_first` / `bm25_first` / `balanced` and shifts weights
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
- **26 tools** spanning retrieval, KG operations, bulletin management, cluster, evaluation, export
- **MCP Resources** — all indexed documents browsable as `locus://doc/{path}`
- **4 MCP Prompts** — live-context templates pre-filled with retrieved knowledge before sending to the model
- **HTTP transport** — `locus serve --port 7391 --token <secret>` for remote clients

### Memory Management
- **Tiered bulletin board** (Tier 0 Pinned / Tier 1 Hot / Tier 2 Archive) with hit-boost + age-decay scoring
- **Persistent hot tier** — `bulletin.sqlite3` survives process restarts
- **Soft token budget** — WARNING / TREND / CRITICAL alerts; never blocks retrieval
- **File watcher** — `locus watch ./docs` auto-reindexes changes; deletions detected

### Operations
- **Health diagnostics** — `locus doctor` checks corpus, KG, bulletin, store size, version
- **Multi-node cluster** — cross-corpus RRF fusion with node-prefixed provenance (`jarv:bm25`)
- **OMPA bridge** — direct import from OMPA vaults (identical SQLite KG schema)
- **Evaluation** — `locus benchmark qa.json` measures Recall@K and MRR

---

## MCP Tools

<details>
<summary><strong>All 26 tools</strong></summary>

| Tool | Description |
|:---|:---|
| `locus_index` | Index a file or directory; skips unchanged files |
| `locus_retrieve` | Six-signal retrieval — returns chunks with provenance tags |
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
| `retrieve <query>` | Retrieve; `--limit 5 --as-of 2024-01-01 --raw` |
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
| `watch <path>` | Auto-reindex on changes; `--interval 5` |
| `mcp` | Start MCP stdio server |
| `serve` | Start HTTP JSON-RPC server; `--port 7391 --token` |
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

## OMPA Bridge

If you already use [OMPA](https://github.com/jmiaie/ompa) as your agent memory layer, Locus can import your vault directly. The KG schemas are identical — no transformation required.

```bash
locus ingest-ompa ~/path/to/ompa-vault --store .locus
```

```python
from locus.bridge.ompa import OMPABridge
bridge = OMPABridge(engine, vault_path="~/ompa-vault")
result = bridge.ingest()
# {"chunks_indexed": 847, "triples_imported": 3_241, ...}
```

Markdown files are indexed via the normal pipeline. KG triples from `.palace/knowledge_graph.sqlite3` are copied row-by-row. Safe to re-run — checksum dedup prevents duplicate indexing.

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
[PASS]  version              Locus v0.6.0

Result: 5 pass  1 warn  0 fail
```

---

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
pip install locus-rag
```

**Zero runtime dependencies.** Locus uses only Python stdlib: `sqlite3`, `re`, `pathlib`, `threading`, `http.server`, `hashlib`, `math`, `json`.

Optional extras:

```bash
pip install "locus-rag[ompa]"   # OMPA vault bridge
pip install "locus-rag[dev]"    # pytest, coverage
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

Locus is architecturally descended from [OMPA](https://github.com/jmiaie/ompa) (Obsidian-MemPalace-Agnostic), which pioneered the three-layer vault → palace → KG design for agent memory. The tiered bulletin board, temporal KG schema, and session lifecycle patterns are direct adaptations. The OMPA bridge enables seamless import of existing OMPA vaults.
