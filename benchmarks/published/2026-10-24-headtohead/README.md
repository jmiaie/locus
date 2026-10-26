# 2026-10-24 (Graphiti completed 2026-10-26): LLM-memory head-to-head on LongMemEval-S (Locus, OMPA, mem0, LangMem, Graphiti/Zep OSS, Letta)

**Status (updated 2026-10-26):** every system is complete at **n=30**.
- Locus, OMPA, mem0, LangMem and Letta were finished on 2026-10-24.
- **Graphiti (Zep OSS)** was resumed on 2026-10-26 from question 12, with exactly the same configuration as the first 11: `gpt-4o-mini` via OpenRouter, default `add_episode`, local MiniLM embeddings, the Kuzu backend and the same workarounds. The resume cost $7.20 of OpenRouter usage. That includes about $0.48 lost to a local-gateway budget stop; the questions that failed with 402 were discarded and re-run.
- In one Graphiti question (c9f37c46), 1 of 49 episodes failed because `gpt-4o-mini` returned truncated JSON. It is counted in `add_errors` and not dropped, which is the harness's original rule.

Total OpenRouter usage for this head-to-head is $15.02.

A Graphiti run on Gemini (`gemini-3.5-flash-lite`, free tier) is in progress. It will be reported separately and labeled "different LLM, not directly comparable to the gpt-4o-mini rows".

## Design
- **Data:** LongMemEval-S (cleaned), a fixed **30-question** subset with 5 per question type, seed 20261020 (`scripts/lme_subset30.json`). Every system gets the same questions.
- **Per question:** each system gets a fresh store. It ingests the ~48 haystack sessions (one call per session), then answers one query. Retrieved memories are mapped back to their source session, and the first 10 distinct sessions are scored.
- **Metrics:** session *retrieval* only, not LLM-judged QA accuracy.
  - R@5: any answer session in the top 5.
  - R_all@5: all answer sessions in the top 5.
  - MRR@10.
- **Significance:** paired bootstrap on MRR@10 over the same questions (B=10,000, seed 20261020). R@5 saturates for Locus and OMPA. **n=30 is low-power:** only large gaps reach significance, and there is no multiple-comparison correction.
- **Models:** the LLM for every LLM-based system is `openai/gpt-4o-mini` via OpenRouter, temperature 0. Embeddings for every system are local `all-MiniLM-L6-v2`.
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
