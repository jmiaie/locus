# 2026-10-23: LLM query transforms (rewrite / multi-query / HyDE) on BEIR

This page measures the optional `locus.llm` plugin from PR #10 against a real endpoint. The plugin was previously tested only against a mock.

- **Locus:** `main` @ `6801174`, default retrieval (SQLite FTS5 BM25 + Porter stemming).
- **LLM:** `openai/gpt-4o-mini` via OpenRouter (served by OpenAI and Azure), temperature 0. Prompts are unmodified from `locus/llm.py`.
- **Fusion:** each strategy keeps the original query and fuses with RRF (k=60) at depth 100 chunks, then takes the top-10 distinct docs.
- **Queries:** BEIR test splits, **all queries**: SciFact n=300, NFCorpus n=323.
- **Statistics:** paired bootstrap over queries (B=10,000, seed 20261020). SIG means the 95% CI excludes 0.

| Dataset | Variant | nDCG@10 [95% CI] | Δ vs default [95% CI] | | LLM p50 | $ / 1k queries |
|---|---|---|---|---|---|---|
| SciFact | default | 0.6810 [0.6379, 0.7235] | – | | – | 0 |
| SciFact | rewrite | 0.6886 | +0.0076 [−0.0016, +0.0170] | ns | 865 ms | $0.019 |
| SciFact | multi (3) | 0.6840 | +0.0029 [−0.0183, +0.0241] | ns | 1,355 ms | $0.048 |
| SciFact | HyDE | 0.6887 | +0.0076 [−0.0125, +0.0280] | ns | 2,058 ms | $0.076 |
| NFCorpus | default | 0.3286 [0.2941, 0.3637] | – | | – | 0 |
| NFCorpus | rewrite | 0.3469 | +0.0182 [+0.0093, +0.0277] | **SIG** | 832 ms | $0.015 |
| NFCorpus | multi (3) | 0.3497 | +0.0211 [+0.0094, +0.0333] | **SIG** | 1,118 ms | $0.032 |
| NFCorpus | HyDE | **0.3588** | +0.0302 [+0.0177, +0.0436] | **SIG** | 1,919 ms | $0.072 |

**Takeaways**
- On short, vocabulary-mismatched queries (NFCorpus), all three transforms help significantly.
- HyDE gives the largest gain measured for Locus so far. For comparison, the cross-encoder rerank gave +0.018 and RM3 gave +0.014.
- On long, precise claims (SciFact) the change is not significant.
- The cost is about 0.8–2 s of added latency per query and roughly 1.5–8 cents per 1,000 queries. Because of that, the plugin stays opt-in.

**Not included yet**
- LongMemEval results for these transforms.
- The LLM-memory head-to-head (mem0, Graphiti/Zep, Letta, LangMem).

Both were blocked because the OpenRouter account had no purchased credits (HTTP 402). They will be added to this folder when they are run. No numbers were estimated.

**Files**
- `summary.json`: metrics, CIs, latency and cost.
- `llm_outputs/`: the exact LLM-generated queries (gzip JSON), so these numbers can be reproduced without an API key.
- `scripts/`: the harness. The key is read only from the environment and is never stored.
