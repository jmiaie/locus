# Locus PRD: the retrieval half of the OMPA + Locus MCP bundle

Status: draft for the owner. Written against `locus-rag` 1.0.0 (`locus/core.py` `__version__`).
Everything under "What exists today" is checked against the code in this repository. Items marked
**Assumption** are not verified and need the owner's confirmation. No customers, usage numbers or
benchmark results are claimed here that the repository does not contain.

Related: [ROADMAP.md](ROADMAP.md) · [CHANGELOG](../CHANGELOG.md) · [benchmarks/README.md](../benchmarks/README.md) ·
[server.json](../server.json) · OMPA: [jmiaie/ompa](https://github.com/jmiaie/ompa)

---

## 1. Problem

Agents that keep memory in a markdown vault (OMPA's model) need two different things:

1. **A source of truth they can write to**: notes, a temporal knowledge graph, session lifecycle. OMPA does this.
2. **Retrieval they can defend**: "give me the passages that answer this, and say why each one was picked",
   without running an embedding model or a vector database.

Vector RAG answers (2) with a cosine score that cannot be explained, and it adds a model download, a GPU
or an API bill, and a second datastore. Plain keyword search is explainable but misses facts that live in
links and in the knowledge graph rather than in the query's words.

Locus fills (2): lexical retrieval (BM25) fused with KG, link, structural and recency signals, every result
carrying a provenance tag, and all of it in the Python standard library.

## 2. Target users

| User | Situation | What they need from Locus |
|---|---|---|
| Developer running an agent in Claude Code / Claude Desktop / Cursor | Already has notes or an OMPA vault; wants the agent to look things up | One `claude mcp add` line, no model downloads, answers with sources |
| Small team with a shared docs vault | Wants retrieval on a laptop or a small box, no infra | Local SQLite store, optional HTTP transport behind a token |
| Builder of a regulated product (CognitionOS, per README) | Must show why a passage was returned | `locus_explain` narratives and provenance on every chunk |

**Assumption:** these are the intended users based on the README "Companion stack" section. There is no usage
data in the repository.

## 3. Jobs to be done

- When my agent needs context about a topic, **retrieve the few passages that answer it** and tell me which signal found each one.
- When a fact changes over time, **answer "as of" a date** instead of mixing old and new facts.
- When my memory lives in OMPA, **search it without copying it by hand** and without Locus becoming a second source of truth.
- When I evaluate retrieval, **compare Locus with a standard BM25 and a small embedding model on public data** before I trust it.

## 4. What exists today (verified in code)

| Area | Fact | Where |
|---|---|---|
| Package | `locus-rag` 1.0.0, MIT, Python ≥ 3.11, `dependencies = []` | `pyproject.toml` |
| Retrieval | Six signals (BM25, KG, link walk, structural, recency, link popularity) fused with weighted RRF; query-intent routing | `locus/core.py`, `locus/retrieval/` |
| Explainability | `explain(chunk_id, query)` returns matched terms, entities and a narrative | `LocusEngine.explain` |
| Temporal KG | Triples with `valid_from` / `valid_to`, `as_of` queries, contradiction detection, aliases | `locus/memory/knowledge_graph.py` |
| MCP | 47 tools (`locus/mcp/tools.py`), 4 prompts, `locus://doc/` resources; stdio and HTTP JSON-RPC | `locus/mcp/` |
| OMPA bridge | Indexes vault markdown and copies `.palace/knowledge_graph.sqlite3` triples into the Locus KG | `locus/bridge/ompa.py` |
| Benchmarks | BEIR SciFact/NFCorpus (nDCG@10 with CIs vs `rank_bm25`), synthetic scaling to 100k docs, Engineering Wiki and simple synthetic corpora | `locus/bench/`, `benchmarks/` |
| Tests | 301 tests, CI on Python 3.11 to 3.13 | `tests/`, `.github/workflows/ci.yml` |

`benchmarks/README.md` (2026-10-20) reports BEIR results: the default BM25 (SQLite FTS5) mode is at parity
with `rank_bm25` (SciFact nDCG@10 0.668 vs 0.664, NFCorpus 0.306 vs 0.308), and the optional hybrid
re-rank is not significantly better than BM25, so it is opt-in. There is no embedding baseline yet.

## 5. Role in the bundle

```
agent ──MCP──► OMPA  (ao_* tools)    writes: vault notes, .palace/knowledge_graph.sqlite3   ← source of truth
      └─MCP──► Locus (locus_* tools) reads:  vault markdown + OMPA KG (read-only)            ← retrieval
                                      writes: its own .locus/ store (derived index, rebuildable)
```

Rules for the bundle:

1. **OMPA owns memory.** Locus never writes to an OMPA vault or its KG. The bridge opens the OMPA KG read-only
   (`mode=ro`) and checks `PRAGMA user_version` (0 or 1 accepted).
2. **The Locus store is a derived index.** Deleting `.locus/` and re-running `locus_ingest_ompa` must rebuild it.
3. **Provenance flows through.** OMPA's `source_file` is mapped to the Locus triple `source`, so Locus answers can cite
   the OMPA note a fact came from.
4. **Tool names do not collide** (`ao_*` vs `locus_*`). Both servers expose `session_start`, `wrap_up`, `status` and `sync`
   style tools; see open question Q3.

## 6. Goals and non-goals

**Goals (next two releases)**

- G1. Locus is safe to run by default as an MCP server: honours its configured store, binds loopback, rejects unsafe archives. *(Done in this PR; see section 9.)*
- G2. Ingesting an OMPA vault imports OMPA's real schema and fails loudly on a mismatch. *(Done in this PR.)*
- G3. Publish reproducible numbers on a public dataset (a BEIR subset and a LongMemEval subset) against `rank_bm25` and a small embedding baseline, including the cases where Locus loses.
- G4. Keep the Locus store in step with the OMPA vault without manual full re-ingests.
- G5. List Locus in the MCP registry from `server.json`.

**Non-goals**

- N1. Becoming a memory store: Locus does not accept writes that OMPA should own (notes, session logs).
- N2. Adding runtime dependencies. Embedding baselines live in benchmark extras only.
- N3. Hosting a service. HTTP transport stays a local or private-network convenience.
- N4. Beating dense retrievers on every dataset. The claim is "explainable and dependency-free, at a measured cost".

## 7. User stories and acceptance criteria

**US1. Configured store.** As a developer, when I start `locus mcp --store ~/vault/.locus`, every tool call reads and writes that store.
- AC: a `tools/call` without `store_path` uses the `--store` path (test `test_mcp_stdio_tool_calls_use_configured_store`).
- AC: `store_path` in the arguments overrides it; a path containing `..` is rejected (`test_mcp_call_tool_store_path_argument_overrides`).
- AC: `locus --store X mcp` and `locus mcp --store X` both work.

**US2. Safe HTTP.** As a user who runs `locus serve`, I am not exposed to my network or to random web pages by default.
- AC: default bind `127.0.0.1`; non-loopback bind without `--token` refuses to start (`test_http_server_requires_token_for_non_loopback`).
- AC: no `Access-Control-Allow-Origin` header unless `--cors-origin` is set; `*` is refused (`test_http_server_no_wildcard_cors_and_token_enforced`).

**US3. Safe restore.** As a user restoring a snapshot, a crafted archive cannot write outside the target or delete my files.
- AC: absolute, `..`, symlink and hardlink members are rejected; `filter="data"` used when available (`test_snapshot_restore_rejects_traversal_members`).
- AC: `overwrite=True` refuses a non-empty directory that is not a Locus store (`test_snapshot_overwrite_refuses_non_store_directory`).
- AC: over MCP, restore targets must be relative and inside the working directory (`test_mcp_restore_tool_confined_to_cwd`).

**US4. OMPA import.** As an OMPA user, `locus_ingest_ompa` imports my KG triples with their source.
- AC: a KG built from OMPA's real DDL (`tests/fixtures/ompa_kg_schema.sql`) imports with `source_file` → `source` (`test_ompa_bridge_imports_real_ompa_schema`).
- AC: a missing column or unknown `user_version` raises `OMPASchemaError` before any indexing; the MCP tool returns the message (`test_ompa_bridge_schema_mismatch_fails_loudly`, `test_ompa_bridge_rejects_unknown_kg_version`).

**US5. Honest benchmark.** As an evaluator, I can run one command on a local BEIR dataset and get Locus vs `rank_bm25` vs an embedding baseline with the same metrics.
- AC (partly done): `python -m locus.bench.beir <dir>` reports nDCG@10, Recall@{1,5,10,100}, MRR and ms/query for Locus and `rank_bm25`.
- AC (open): embedding baseline implemented behind an extra; results JSON committed with dataset name, split, query count, machine and versions.

**US6. Incremental sync.** As an OMPA user, after I edit notes, Locus reflects the change on the next ingest without duplicating triples or re-indexing unchanged files.
- AC (open): re-running ingest on an unchanged vault indexes 0 new chunks and leaves the triple count unchanged; a triple invalidated in OMPA (`valid_to` set) is invalidated in Locus.

**US7. Registry listing.** As a user browsing the MCP registry, I can install Locus with one line.
- AC: `server.json` validates with `mcp-publisher validate` (owner to run), its version matches `pyproject.toml` (`test_server_json_matches_package`), and the README carries the `mcp-name` marker.

## 8. Requirements

**Functional**

- F1. Tool calls resolve the store as: explicit `store_path` argument, else the transport's configured engine, else `.locus`.
- F2. JSON-RPC: `ping` returns `{}`; unknown methods with an `id` return error `-32601`; notifications get no reply.
- F3. OMPA bridge reads columns by name, maps `source_file` → `source`, accepts a legacy `source` column, and refuses unknown schema versions.
- F4. Benchmark harness reads datasets from disk only; it never downloads.

**Non-functional**

- NF1. Zero runtime dependencies (hard constraint from `CLAUDE.md`).
- NF2. Secure defaults: loopback bind, token for anything else, no wildcard CORS, safe archive extraction.
- NF3. Documentation counts (tools, version) are checked by tests so they cannot drift silently.
- NF4. Latency: report p50/p95 per query on the benchmark machine. **No target is set yet**; see Q5.

## 9. Benchmark plan

No results are reported in this document. The plan below is what the roadmap will produce.

**Datasets** (all run from local copies; download is a manual, documented step outside the harness)

| Dataset | Why | Size used |
|---|---|---|
| BEIR SciFact | Small, standard, lexical-friendly; good sanity check | full `test` split (300 queries, 5,183 docs) |
| BEIR NFCorpus | Domain vocabulary gap; where lexical retrieval is known to struggle | full `test` split (323 queries, 3,633 docs) |
| LongMemEval (`_s` variant) | Long-horizon agent memory: the bundle's actual use case | a fixed, seeded subset of questions (size decided in N4, starting with 100); score session-level retrieval recall, not answer accuracy |

**Systems**

1. `rank_bm25` BM25Okapi (already an optional extra `bench`), same tokenizer as the existing baseline.
2. Locus full pipeline (`LocusEngine.retrieve`), and Locus BM25-only for an ablation.
3. Small embedding baseline: `sentence-transformers/all-MiniLM-L6-v2`, exact cosine search with NumPy, installed only through a new benchmark extra. Never a runtime dependency.

**Metrics:** nDCG@10 (BEIR's headline metric), Recall@10 and Recall@100, MRR, index time, query p50 and p95.
For LongMemEval: recall@k of the evidence sessions.

**Protocol**

- Pin the dataset version and record a SHA-256 of each input file in the results JSON.
- Same machine, single process, warm cache, three runs; report the median.
- Commit raw JSON to `benchmarks/results/` and a table to `benchmarks/README.md`, including every config where Locus loses.
- Harness status: `locus/bench/beir.py` (this PR) covers loading, metrics, Locus and `rank_bm25`; the embedding baseline is a
  `NotImplementedError` stub pointing at ROADMAP item N3.

## 10. Positioning and pricing

- Positioning: "explainable, dependency-free retrieval for agent memory". The differentiator is provenance plus temporal KG,
  not raw accuracy. Until public numbers exist, the README should not claim accuracy parity with vector search.
- Pricing: Locus is MIT and free. **Assumption:** any commercial offer sits in CognitionOS, not in Locus. Owner to confirm.

## 11. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Public benchmarks show Locus below `rank_bm25` (it already is on the synthetic wiki MRR) | Weakens the pitch | Publish anyway; lead with explainability; tune RRF weights per intent with the public data |
| OMPA changes its KG schema | Bridge breaks | Version check (done); OMPA's proposed contract doc `docs/kg-schema-v1.md` and a shared fixture; track OMPA releases |
| Locus KG drifts from OMPA's after edits | Stale or contradictory answers | US6 incremental sync; document that `.locus/` is rebuildable |
| 47 tools overwhelm an agent's tool list | Worse tool choice, larger prompts | Q3: consider a "bundle profile" exposing a smaller tool set |
| HTTP transport misused on a network | Data exposure | Secure defaults (done); document reverse-proxy + TLS for remote use |
| `server.json` needs a release that contains the `mcp-name` marker and `locus-rag` script | Registry rejects it until then | Ship in the next patch release |

## 12. Open questions for the owner

- Q1. Should the bundle keep **copying** OMPA triples into the Locus KG, or should Locus query OMPA's KG in place (read-only) at retrieval time? Copying is what exists; in-place avoids drift.
- Q2. Release number for this work: 1.0.1 (fixes) or 1.1.0 (new `locus-rag` script, benchmark module, behaviour change in `serve` defaults)?
- Q3. Should there be a reduced "bundle" tool profile (for example retrieve, explain, prepare_context, query_entity, ingest_ompa, status) to avoid duplicating OMPA's `ao_session_start` / `ao_wrap_up` / `ao_status` / `ao_sync`?
- Q4. Is a `locus-rag` console script (needed for `uvx locus-rag ...` in `server.json`) acceptable, or would you rather list a different runtime hint?
- Q5. What latency budget per query should the bundle target on a laptop?
- Q6. Which LongMemEval subset size and which split are acceptable to publish, given its licence and size?
- Q7. Restore over the CLI still accepts any absolute `--target` (operator's choice) but refuses `/` and `$HOME`. Is that the right line, or should the CLI also be confined to the working directory?
