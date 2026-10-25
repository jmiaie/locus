# 2026-10-24: LLM-memory head-to-head on LongMemEval-S (Locus, OMPA, mem0, LangMem, Graphiti/Zep OSS, Letta)

**Status:** Locus, OMPA, mem0 and LangMem are complete (n=30). **Graphiti and Letta are in progress (partial n).** This folder will be updated when they finish.

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
| Graphiti 0.31.2 / Zep OSS (default `search`) | 4 | 0.500 [0.00, 1.00] | 0.250 | 0.333 | -0.479 [-0.833, -0.125] **sig** | -0.667 [-1.000, -0.250] **sig** | 81 ms | 352.2 s | $0.436 |
| Graphiti, same graph, hybrid RRF with cosine floor removed | 4 | 0.750 [0.25, 1.00] | 0.500 | 0.542 | -0.271 [-0.667, +0.125] ns | -0.458 [-0.750, -0.125] **sig** | 70 ms | 352.2 s | incl. above |
| Letta 0.16.8 agent (archival_memory_search tool calls) | 4 | 0.750 [0.25, 1.00] | 0.750 | 0.750 | -0.062 [-0.750, +0.562] ns | -0.250 [-0.750, +0.000] ns | 3,401 ms | 113.3 s | $0.001 |
| Letta 0.16.8 archival search API (no LLM at query) | 4 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | +0.188 [+0.000, +0.562] ns | +0.000 [+0.000, +0.000] ns | 267 ms | 113.3 s | incl. above |

LLM $ per question = system spend ÷ scored questions. For Graphiti this includes the pilot and interrupted attempts. A clean run measured about $0.38 per question.

## Notes
- **mem0** (`infer=True`, local Qdrant, spaCy and fastembed extras): 2 R@5 misses, both on preference questions. About 329 memories per haystack.
- **LangMem:** the LangGraph `recursion_limit` was raised from 25 to 100. With the default, 11 of 48 sessions errored. 7 R@5 misses. About 56 memories per haystack. A memory is attributed to the session that created it.
- **Graphiti**
  - Zep Cloud needs a separate Zep key, so Graphiti OSS was used instead.
  - The Kuzu backend in 0.31.2 needed two workarounds: the missing `_database` attribute, and creating the FTS indexes.
  - The default `search()` cosine floor (0.6) is tuned for OpenAI embeddings and returns few edges with MiniLM. The "floor removed" row searches the same graph without it.
- **Letta**
  - The archival-search tool was attached explicitly, because 0.16.8's default tool set lacks it.
  - Each session turn is one archival passage tagged with its session id.
  - "Agent" = the passages the agent retrieved through its own tool call. "Archival search API" = Letta's direct search endpoint.
- No API key appears anywhere here. The gateway reads it from the environment. Per-question rows are in `results/`.
