# 2026-10-24 (Graphiti completed 2026-10-26): LLM-memory head-to-head on LongMemEval-S (Locus, OMPA, mem0, LangMem, Graphiti/Zep OSS, Letta)

**Status (updated 2026-10-26):** every system is complete at **n=30**.
- Locus, OMPA, mem0, LangMem and Letta were finished on 2026-10-24.
- **Graphiti (Zep OSS)** was resumed on 2026-10-26 from question 12, with exactly the same configuration as the first 11: `gpt-4o-mini` via OpenRouter, default `add_episode`, local MiniLM embeddings, the Kuzu backend and the same workarounds. The resume cost $7.20 of OpenRouter usage. That includes about $0.48 lost to a local-gateway budget stop; the questions that failed with 402 were discarded and re-run.
- In one Graphiti question (c9f37c46), 1 of 49 episodes failed because `gpt-4o-mini` returned truncated JSON. It is counted in `add_errors` and not dropped, which is the harness's original rule.

Total OpenRouter usage for this head-to-head is $15.02.

**Update 2026-10-30 (scorecard v7):** the Graphiti run on Gemini (`gemini-3.5-flash-lite`, free tier, `reasoning_effort=minimal`) is complete at n=30. It is reported in its own section below ("Graphiti on Gemini"), and is **different LLM, not directly comparable to the gpt-4o-mini rows**. The main table below is unchanged.

**Update 2026-11-06 (scorecard v8):** a separate Graphiti re-ingest on the same questions with `reasoning_effort=high` is complete at n=30 unique final good rows. It is reported in its own section below and does not replace the v7 rows or the main table. Label: gemini-3.5-flash-lite, reasoning high; different LLM setting, not directly comparable. Full text: [`SCORECARD_v8_20261106.md`](SCORECARD_v8_20261106.md). The v7 section inside that file was reconstructed from the table already published here, because a local v7 file was not in the publication bundle. This README remains the authority for the main table and for the minimal-reasoning Gemini rows.

## Design
- **Data:** LongMemEval-S (cleaned), a fixed **30-question** subset with 5 per question type, seed 20261020 (`scripts/lme_subset30.json`). Every system gets the same questions.
- **Per question:** each system gets a fresh store. It ingests the ~48 haystack sessions (one call per session), then answers one query. Retrieved memories are mapped back to their source session, and the first 10 distinct sessions are scored.
- **Metrics:** session *retrieval* only, not LLM-judged QA accuracy.
  - R@5: any answer session in the top 5.
  - R_all@5: all answer sessions in the top 5.
  - MRR@10.
- **Significance:** paired bootstrap on MRR@10 over the same questions (B=10,000, seed 20261020). R@5 saturates for Locus and OMPA. **n=30 is low-power:** only large gaps reach significance, and there is no multiple-comparison correction.
- **Models:** the LLM for every LLM-based system in the main table is `openai/gpt-4o-mini` via OpenRouter, temperature 0. Embeddings for every system are local `all-MiniLM-L6-v2`. The Gemini sections below are separate LLM settings and are not rows of that main table.
- **Telemetry:** disabled for all libraries (mem0, Graphiti, LangChain, HF).
- **Versions:** Locus main `74ffb7f`, OMPA main `34d533e`, mem0ai 2.2.1, langmem 0.0.30, graphiti-core 0.31.2 (Kuzu backend), letta 0.16.8 server (Postgres 17 + pgvector).

## Results
| System | n | R@5 [95% CI] | R_all@5 | MRR@10 | ΔMRR vs Locus, same q [95% CI] | ΔMRR vs OMPA, same q | Query p50 | Ingest p50 / haystack | LLM $ / question |
|---|---|---|---|---|---|---|---|---|---|
| Locus (FTS5 BM25, default) | 30 | 1.000 [1.00, 1.00] | 0.933 | 0.917 | – | -0.050 [-0.142, +0.025] ns | 3 ms | 0.3 s | $0 |
| OMPA (MiniLM + keyword) | 30 | 1.000 [1.00, 1.00] | 0.933 | 0.967 | +0.050 [-0.025, +0.142] ns | – | 25 ms | 9.6 s | $0 |
| mem0 2.2.1 (LLM extraction, infer=True) | 30 | 0.933 [0.83, 1.00] | 0.867 | 0.722 | -0.194 [-0.300, -0.100] **sig** | -0.244 [-0.356, -0.144] **sig** | 100 ms | 228.2 s | $0.062 |
| LangMem 0.0.30 (memory manager) | 30 | 0.767 [0.60, 0.90] | 0.567 | 0.606 | -0.311 [-0.461, -0.156] **sig** | -0.361 [-0.511, -0.211] **sig** | 57 ms | 238.2 s | $0.043 |
| Graphiti 0.31.2 / Zep OSS (default `search`) | 30 | 0.467 [0.30, 0.63] | 0.267 | 0.428 | -0.489 [-0.664, -0.317] **sig** | -0.539 [-0.717, -0.356] **sig** | 68 ms | 437.8 s | $0.390 |
| Graphiti, same graph, hybrid RRF with cosine floor removed | 30 | 0.733 [0.57, 0.87] | 0.467 | 0.547 | -0.369 [-0.533, -0.214] **sig** | -0.419 [-0.578, -0.264] **sig** | 73 ms | 437.8 s | incl. above |
| Letta 0.16.8 agent (archival_memory_search tool calls) | 30 | 0.800 [0.63, 0.93] | 0.733 | 0.800 | -0.117 [-0.292, +0.042] ns | -0.167 [-0.317, -0.033] **sig** | 3,432 ms | 120.1 s | $0.001 |
| Letta 0.16.8 archival search API (no LLM at query) | 30 | 1.000 [1.00, 1.00] | 1.000 | 0.957 | +0.040 [-0.063, +0.142] ns | -0.010 [-0.060, +0.040] ns | 270 ms | 120.1 s | incl. above |
LLM $ per question = system spend ÷ scored questions. For Graphiti this includes the pilot and interrupted attempts: $11.68 total, 6,544 calls and 80.0M prompt tokens.

## Graphiti on Gemini (different LLM, not directly comparable to the gpt-4o-mini rows)
Scorecard v7, added 2026-10-30. Same 30 questions, same harness (`scripts/p2_run.py` with `P2_PROVIDER=gemini`), same graphiti-core 0.31.2 + Kuzu workarounds, same local MiniLM embeddings, same scoring and paired bootstrap (B=10,000, seed 20261020). The only change is the LLM: `gemini-3.5-flash-lite` (`reasoning_effort=minimal`) on the Gemini API free tier, through `scripts/gateway_gemini.py`. Because the LLM builds the graph, these rows measure a different system and must **not** be read as a like-for-like replacement for the `gpt-4o-mini` Graphiti rows above.

| System (different LLM, not directly comparable to the gpt-4o-mini rows) | n | R@5 [95% CI] | R_all@5 | MRR@10 | Query p50 | Ingest p50 / haystack | LLM $ |
|---|---|---|---|---|---|---|---|
| Graphiti 0.31.2, `gemini-3.5-flash-lite` (default `search`) | 30 | 0.167 [0.03, 0.30] | 0.100 | 0.167 | 46 ms | 5,247.5 s* | $0 (free tier) |
| Graphiti, `gemini-3.5-flash-lite`, same graph, cosine floor removed | 30 | 0.367 [0.20, 0.53] | 0.233 | 0.295 | 68 ms | 5,247.5 s* | $0 (free tier) |

| Gemini row | vs (same 30 q) | ref R@5 / MRR | ΔMRR@10 [95% CI], bootstrap p | ΔR@5 [95% CI], bootstrap p | better / worse (MRR) |
|---|---|---|---|---|---|
| default | Graphiti gpt-4o-mini (default) | 0.467 / 0.428 | -0.261 [-0.467, -0.039] **sig**, p=0.0198 | -0.300 [-0.500, -0.100] **sig**, p=0.008 | 3 / 11 |
| default | Locus | 1.000 / 0.917 | -0.750 [-0.883, -0.608] **sig**, p<0.0001 | -0.833 [-0.967, -0.700] **sig**, p<0.0001 | 0 / 25 |
| default | OMPA | 1.000 / 0.967 | -0.800 [-0.933, -0.667] **sig**, p<0.0001 | -0.833 [-0.967, -0.700] **sig**, p<0.0001 | 0 / 25 |
| floor removed | Graphiti gpt-4o-mini (floor removed) | 0.733 / 0.547 | -0.252 [-0.433, -0.071] **sig**, p=0.0064 | -0.367 [-0.567, -0.167] **sig**, p=0.0026 | 6 / 17 |
| floor removed | Graphiti gpt-4o-mini (default) | 0.467 / 0.428 | -0.133 [-0.349, +0.084] ns, p=0.2208 | -0.100 [-0.333, +0.133] ns, p=0.4776 | 7 / 10 |
| floor removed | Locus | 1.000 / 0.917 | -0.622 [-0.782, -0.450] **sig**, p<0.0001 | -0.633 [-0.800, -0.467] **sig**, p<0.0001 | 1 / 23 |
| floor removed | OMPA | 1.000 / 0.967 | -0.672 [-0.817, -0.522] **sig**, p<0.0001 | -0.633 [-0.800, -0.467] **sig**, p<0.0001 | 0 / 23 |

\* Ingest time includes free-tier rate-limit waits, daily-quota holds and upstream 503 retries, so it is not a throughput measure. The two Gemini rows search the same graph. The ΔR@5 column and the two-sided bootstrap p-value, 2·min(P(Δ*≤0), P(Δ*≥0)), are extra outputs of `scripts/p2_gemini_analyze.py`; `p2_analyze.py` reports only the MRR CI. p<0.0001 means no bootstrap resample crossed 0 (B=10,000). No multiple-comparison correction.

- **Run integrity:** 30 rows, one per subset question (ids match `scripts/lme_subset30.json`), 0 error rows, 0 `add_episode` errors in total (mean 48.5 episodes per haystack; gpt-4o-mini run 48.47). Every row and every gateway call used `gemini-3.5-flash-lite`: 4,936 successful calls (337 replayed from the gateway cache after restarts), 86.3M prompt tokens and 594k completion tokens. **LLM spend: $0.** The gateway's spend guard (stop on any non-free-tier quota or billing signal) never fired.
- **What differs:** with Gemini, the default `search()` returned no edges for 19 of 30 questions (gpt-4o-mini graph: 5 of 30). The median number of edges hit was 0 (gpt-4o-mini: 2). With the cosine floor removed, the median number of edges returned was 27.5 (gpt-4o-mini: 69.5).
- **R@5 by question type:** default search: knowledge-update 0.0, multi-session 0.4, single-session-assistant 0.4, single-session-preference 0.0, single-session-user 0.2, temporal-reasoning 0.0. Floor removed: knowledge-update 0.6, multi-session 0.6, single-session-assistant 0.4, single-session-preference 0.2, single-session-user 0.4, temporal-reasoning 0.0.
- **Reading:** on this subset, Graphiti built with `gemini-3.5-flash-lite` retrieves worse than Graphiti built with `gpt-4o-mini` (ΔMRR -0.261, sig.; floor removed -0.252, sig.), and both stay far below Locus and OMPA. The Gemini run does not change any number in the main table.
- Per-question rows: `results/lme30__graphiti_gemini.jsonl`. Summary: `summary_graphiti_gemini.json` (kept separate from `summary.json`). Supervisor: `scripts/run_gemini_graphiti.sh`. API keys are read from the environment by variable name only and never appear in any file.

## Graphiti on Gemini, reasoning high (gemini-3.5-flash-lite, reasoning high; different LLM setting, not directly comparable)

Scorecard v8, added 2026-11-06. This is a separate re-ingest of the same 30 LongMemEval-S questions. It leaves the minimal-reasoning Gemini rows above (scorecard v7) and the `gpt-4o-mini` rows in the main table as they are. Graphiti 0.31.2, default `add_episode`, Kuzu, and local `all-MiniLM-L6-v2` embeddings match the earlier Graphiti runs. The LLM setting differs: `gemini-3.5-flash-lite` with `reasoning_effort=high` on the Gemini free tier (`GM_REASONING_EFFORT=high`, `P2_VARIANT=high`). Label: gemini-3.5-flash-lite, reasoning high; different LLM setting, not directly comparable.

The published JSONL holds the 30 unique final good rows, one per id in `scripts/lme_subset30.json`. The analysis summary records 35 rows before cleanup: 5 earlier incomplete-ingest rows were retried and replaced, and those error rows are not in the published file. On the 30 final rows, `add_errors` sum to 0, every row has `llm_model=gemini-3.5-flash-lite`, `reasoning_effort=high`, `variant=high`, and `cum_usd_system` 0. LLM spend on those rows is $0 (free tier). The summary's `gateway_models_seen` also lists `gemini-3.1-flash-lite` beside `gemini-3.5-flash-lite` for the gateway sqlite as a whole. That database is not in this bundle, and those calls are not attributed to this variant row by row. The scored rows themselves record only `gemini-3.5-flash-lite`. Paired bootstrap on the same questions uses B=10,000 and seed 20261020, with no multiple-comparison correction. Point metrics and the paired tests in the table were recomputed locally from this JSONL and the existing `gpt-4o-mini` and minimal-Gemini rows, using the same hit, MRR@10, and bootstrap definitions as `scripts/p2_gemini_analyze.py`. They match `summary_graphiti_gemini_high.json`. That check called no model API.

| System (gemini-3.5-flash-lite, reasoning high; different LLM setting, not directly comparable) | n | R@5 [95% CI] | R_all@5 | MRR@10 | Default search empty | Query p50 | LLM $ |
|---|---|---|---|---|---|---|---|
| Graphiti, gemini-3.5-flash-lite, reasoning high (default `search`) | 30 | 0.500 [0.333, 0.667] | 0.200 | 0.467 | 1/30 | 50 ms | $0 (free tier) |
| Same graph, cosine floor removed | 30 | 0.633 [0.467, 0.800] | 0.367 | 0.539 | n/a (search list non-empty on these 30) | 72 ms | $0 (free tier) |

| High row | vs (same 30 q) | ref R@5 / MRR | ΔMRR@10 [95% CI], p | ΔR@5 [95% CI], p |
|---|---|---|---|---|
| default | Graphiti gpt-4o-mini (default) | 0.467 / 0.428 | +0.039 [-0.200, +0.278] ns, p=0.7622 | +0.033 [-0.200, +0.267] ns, p=0.8826 |
| default | Gemini minimal (default, v7) | 0.167 / 0.167 | +0.300 [+0.106, +0.500] **sig**, p=0.0026 | +0.333 [+0.133, +0.533] **sig**, p=0.0006 |
| floor removed | Graphiti gpt-4o-mini (floor removed) | 0.733 / 0.547 | -0.009 [-0.202, +0.190] ns, p=0.9324 | -0.100 [-0.300, +0.100] ns, p=0.4366 |
| floor removed | Gemini minimal (floor removed, v7) | 0.367 / 0.295 | +0.244 [+0.068, +0.426] **sig**, p=0.0054 | +0.267 [+0.100, +0.467] **sig**, p=0.0050 |

**How to read the tests.** On MRR@10 there is no statistically significant difference versus Graphiti built with `gpt-4o-mini`: p=0.7622 for default search and p=0.9324 with the cosine floor removed. Absence of a significant difference is not evidence of equivalence. n=30 is low power, and the intervals are wide (about ±0.2 MRR). Versus the minimal-reasoning Gemini run, the MRR difference is significant: p=0.0026 (default) and p=0.0054 (floor removed). No new paired test against Locus or OMPA was run for this variant. The published point estimates on this subset remain MRR 0.917 for Locus and 0.967 for OMPA. This variant's MRR is 0.467 (default) and 0.539 (floor removed), which sits well below those point estimates.

**A nonempty search list is not a correct answer.** Default `search()` returned no sessions for 1 of 30 questions (`157a136e`). The minimal-reasoning rows returned no sessions for 19 of 30, and the `gpt-4o-mini` rows for 5 of 30. The other 29 high-reasoning questions returned a session list. That is only a nonempty search. Default R@5 is 0.500, which is 15 of 30, so a nonempty list usually still misses the answer sessions in the top 5. With the cosine floor removed, all 30 searches returned at least one session and R@5 is 0.633 (19 of 30). Query p50 rounds to 50 ms (median 49.5 ms) and 72 ms (median 71.8 ms). MRR@10 rounds to 0.467 from 0.4666… and, with the floor removed, to 0.539 from 0.5386….

**Extraction diagnosis (prior notes, not recomputed here).** Earlier diagnostic notes on the minimal-reasoning run reported that `extract_edges` returned no facts for 1070 of 1455 episodes (74%), against 16 of 1454 episodes (1%) for `gpt-4o-mini`. Those counts are not re-derived from the files in this bundle, the raw gateway logs are not included, and the high-reasoning run was not measured per episode. They stay observations from those notes, not a new confirmed recount. The quantity this publication checks from the rows is search-list emptiness: 1/30 versus 19/30. The scorecard's spot checks, which are not a full 30-graph recount, say two validation graphs had more extracted edges than the matching minimal graphs (147 and 255 versus 32 and 46), and that two questions which had been empty under minimal reasoning then returned hits (5 and 13) while the answer session was still outside the top 5. Those same notes say the harness had been ruled out for the minimal run (no parse errors, no truncation, no null embeddings, no group_id mismatch, no stale full-text index). `reasoning_effort=high` is a generic model setting, not a change tuned to the answer keys.

Ingest p50 is 2,987.2 s per haystack. That figure includes free-tier rate-limit waits, daily-quota holds, and 503 retries, so it is not a throughput measure. The two high-reasoning rows search the same graph.

**R@5 by question type** (5 questions each). Default search: knowledge-update 0.8, multi-session 0.6, single-session-assistant 0.6, single-session-preference 0.2, single-session-user 0.4, temporal-reasoning 0.4. Floor removed: 0.8 / 0.8 / 0.6 / 0.4 / 0.6 / 0.6.

Per-question rows: `results/lme30__graphiti_gemini_high.jsonl`. Summary: `summary_graphiti_gemini_high.json` (kept separate from `summary.json` and from `summary_graphiti_gemini.json`). The summary is the analysis output copied from the publication bundle, including its analysis-time note that asked not to publish before approval. Jeff approved this publication; that note is retained as history. Scorecard: `SCORECARD_v8_20261106.md`. Historical sections inside the scorecard file do not replace this README or `docs/COMPARISON.md`. Supervisor: `scripts/run_gemini_graphiti_high.sh`. `scripts/p2_run.py` and `scripts/gateway_gemini.py` take the variant through `P2_VARIANT` and `GM_REASONING_EFFORT`. Unset, both keep the previous minimal-reasoning behavior. `scripts/p2_gemini_analyze.py` is unchanged and still reads the minimal-reasoning JSONL. API keys are read from the environment by variable name only and do not appear in any file. The supervisor expects `lme_subset30_order_high.json` in the run directory; that order file is not in this bundle. Scoring uses `scripts/lme_subset30.json`.

## Notes
- **mem0** (`infer=True`, local Qdrant, spaCy and fastembed extras): 2 R@5 misses, both on preference questions. About 329 memories per haystack.
- **LangMem:** the LangGraph `recursion_limit` was raised from 25 to 100. With the default, 11 of 48 sessions errored. 7 R@5 misses. About 56 memories per haystack. A memory is attributed to the session that created it.
- **Graphiti**
  - Zep Cloud needs a separate Zep key, so Graphiti OSS was used instead.
  - The Kuzu backend in 0.31.2 needed two workarounds: the missing `_database` attribute, and creating the FTS indexes.
  - The default `search()` cosine floor (0.6) is tuned for OpenAI embeddings and returns few edges with MiniLM. The "floor removed" row searches the same graph without it.
  - Graphiti R@5 by question type (default search, n=30): knowledge-update 0.6, multi-session 0.4, single-session-assistant 0.6, single-session-preference 0.4, single-session-user 0.0, temporal-reasoning 0.8. With the floor removed: 0.6 / 1.0 / 0.6 / 1.0 / 0.2 / 1.0.
- **Letta**
  - The archival-search tool was attached explicitly, because 0.16.8's default tool set lacks it.
  - Each session turn is one archival passage tagged with its session id.
  - "Agent" = the passages the agent retrieved through its own tool call. "Archival search API" = Letta's direct search endpoint.
  - The agent made 0.97 archival-search calls per question on average. In 5 of 30 questions its tool call returned nothing usable, and those are its misses. The direct archival search API, with no LLM at query time, scores R@5 1.000 / MRR 0.957. That is statistically tied with Locus (+0.040 [-0.063, +0.142]) and OMPA (-0.010 [-0.060, +0.040]), at about 270 ms per query and about 2 minutes of ingest per haystack.
- No API key appears anywhere here. The gateway reads it from the environment. Per-question rows are in `results/`.
