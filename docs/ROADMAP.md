# Locus roadmap (OMPA + Locus bundle)

Companion to [PRD.md](PRD.md). History lives in [CHANGELOG.md](../CHANGELOG.md); benchmark method and the existing
synthetic results live in [benchmarks/README.md](../benchmarks/README.md). This file does not replace either.
OMPA's side of the KG contract is OMPA's to define (OMPA is the source of truth); items here that depend on it say so.

Phases are relative to the merge of this PR. Dates are not commitments.

---

## Done in this PR (hardening, prerequisite for everything below)

| Item | Exit criterion (met) |
|---|---|
| H1. MCP stdio and HTTP honour the configured store | `test_mcp_stdio_tool_calls_use_configured_store`, `test_http_dispatch_uses_engine_passed_to_serve` |
| H2. HTTP secure defaults (loopback, token for non-loopback, CORS opt-in, JSON content type) | `test_http_server_*` tests |
| H3. Safe snapshot restore | `test_snapshot_restore_*`, `test_mcp_restore_tool_confined_to_cwd` |
| H4. OMPA bridge reads OMPA's real schema, read-only, version-checked, fails loudly | `test_ompa_bridge_imports_real_ompa_schema`, `test_ompa_bridge_schema_mismatch_fails_loudly`, `test_ompa_bridge_rejects_unknown_kg_version` |
| H5. Docs match code; tests guard tool count, version and `server.json` | `test_readme_*`, `test_server_json_matches_package` |
| H6. JSON-RPC `ping` and `-32601` for unknown methods | `test_mcp_stdio_ping_and_unknown_method`, `test_http_dispatch_ping` |

## Now (0 to 30 days): public benchmark harness and registry listing

| ID | Deliverable | Exit criteria | Depends on |
|---|---|---|---|
| N1 | BEIR loader and metrics (`locus/bench/beir.py`) | **Scaffolded in this PR**: loads local `corpus.jsonl` / `queries.jsonl` / `qrels/<split>.tsv`, scores nDCG@10, Recall@k, MRR; tests on a fixture | — |
| N2 | Locus and `rank_bm25` runs on SciFact and NFCorpus | **Done (2026-10-20):** tables with CIs in `benchmarks/README.md`, summaries in `benchmarks/published/2026-10-20/`, scripts in `benchmarks/repro/`; open: dataset SHA-256s | N1, dataset copied locally by the owner |
| N3 | Small embedding baseline (`all-MiniLM-L6-v2`, exact cosine) | `run_embedding_baseline` implemented behind a new `bench-embed` extra; no runtime dependency; same metrics and table | N1 |
| N4 | LongMemEval subset adapter | Loader turns a seeded subset of LongMemEval `_s` sessions into documents and scores evidence-session recall@k for the three systems | N1, Q6 in PRD |
| N5 | Release with `server.json` | Next release contains the `mcp-name` README marker and the `locus-rag` script; `mcp-publisher validate` passes; owner publishes | PRD Q2, Q4 |

## Next (30 to 60 days): stay in step with OMPA

| ID | Deliverable | Exit criteria | Depends on |
|---|---|---|---|
| X1 | Idempotent re-ingest | Re-running `locus_ingest_ompa` on an unchanged vault adds 0 chunks and 0 triples (test) | — |
| X2 | Invalidation follows OMPA | A triple with `valid_to` set in OMPA is closed in Locus on the next ingest (test with OMPA DDL fixture) | X1 |
| X3 | Vault-relative provenance | `source` shown as a vault-relative path when OMPA's `source_file` starts with the vault root (OMPA contract v1 section 5) | OMPA contract |
| X4 | Shared contract fixture | Replace the copied DDL in `tests/fixtures/ompa_kg_schema.sql` with OMPA's published contract module or fixture once OMPA releases it | OMPA release |
| X5 | Bundle quick start | One page: install both, two `claude mcp add` lines, ingest, ask. Tested by a script | N5 |

## Later (60 to 90 days): decide from data

| ID | Deliverable | Exit criteria | Depends on |
|---|---|---|---|
| L1 | RRF weight tuning from public data | Per-intent weights chosen on a dev split, reported on test; no regression on the synthetic suites | N2, N3 |
| L2 | Bundle tool profile | Optional `--profile bundle` exposing a reduced tool set (PRD Q3) | Owner decision |
| L3 | Query OMPA KG in place | Prototype read-only KG retrieval against OMPA's database, compared with copy-based import on drift and latency | PRD Q1, X2 |
| L4 | HTTP hardening for remote use | Host-header check against DNS rebinding, documented reverse proxy + TLS recipe | — |

## Not planned

- Runtime embedding or vector search inside Locus (PRD non-goal N2).
- Writing to OMPA vaults from Locus (PRD non-goal N1).
