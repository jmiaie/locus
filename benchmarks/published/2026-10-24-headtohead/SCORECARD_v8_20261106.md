> **Locus publication note.** This file is scorecard v8 (2026-11-06). The new material is the high-reasoning section at the top. The v7 block was reconstructed in the publication bundle from the table already published in this repository (Locus PR #20), because a separate local v7 file was not in the bundle. v6 and the 2026-10-20/21 scorecard text are carried here so the file can be read on its own. They are not a new measurement. Where an earlier row, caveat, or distinction differs, the documents already in this repository stay authoritative: this folder's README (the main n=30 table and the minimal-reasoning Gemini section), [`docs/COMPARISON.md`](../../../docs/COMPARISON.md), and the other write-ups under `benchmarks/published/`. Those files are left in place.

# SCORECARD v8 (2026-11-06): Graphiti on Gemini with high reasoning (separate variant)

This is the current scorecard text for the high-reasoning variant. v7 (the minimal-reasoning Gemini rows) was published in the repos (Locus PR #20 / OMPA PR #25) and is reproduced in the v7 section below from that published table, because a separate local v7 file was not in the publication bundle. Earlier baseline sections are included in this file. They are unchanged by the new rows. The Locus repository documents remain the authority for those historical rows (see the note at the top).

**New variant label:** gemini-3.5-flash-lite, reasoning high; different LLM setting, not directly comparable to the `gpt-4o-mini` rows (nor fully to the v7 minimal-reasoning rows, which used a different reasoning setting).

Setup: same 30 LongMemEval-S questions, Graphiti 0.31.2, default `add_episode`, local MiniLM embeddings, Kuzu. Only the LLM setting differs: Gemini free tier with `reasoning_effort=high`. n=30, all 30 unique questions completed (5 earlier incomplete-ingest rows were retried and replaced), add_errors=0, LLM spend $0. Paired bootstrap on the same questions, B=10,000, seed 20261020.

| System (different LLM setting, not directly comparable) | n | R@5 [95% CI] | R_all@5 | MRR@10 | Default search empty | Query p50 | LLM $ |
|---|---|---|---|---|---|---|---|
| Graphiti, gemini-3.5-flash-lite, reasoning high (default `search`) | 30 | 0.500 [0.333, 0.667] | 0.200 | 0.467 | 1/30 | 50 ms | $0 (free tier) |
| Same graph, cosine floor removed | 30 | 0.633 [0.467, 0.800] | 0.367 | 0.539 | n/a (non-empty) | 72 ms | $0 (free tier) |

| High row | vs (same 30 q) | ref R@5 / MRR | ΔMRR@10 [95% CI], p | ΔR@5 [95% CI], p |
|---|---|---|---|---|
| default | Graphiti gpt-4o-mini (default) | 0.467 / 0.428 | +0.039 [-0.200, +0.278] ns, p=0.7622 | +0.033 [-0.200, +0.267] ns, p=0.8826 |
| default | Gemini minimal (default, v7) | 0.167 / 0.167 | +0.300 [+0.106, +0.500] **sig**, p=0.0026 | +0.333 [+0.133, +0.533] **sig**, p=0.0006 |
| floor removed | Graphiti gpt-4o-mini (floor removed) | 0.733 / 0.547 | -0.009 [-0.202, +0.190] ns, p=0.9324 | -0.100 [-0.300, +0.100] ns, p=0.4366 |
| floor removed | Gemini minimal (floor removed, v7) | 0.367 / 0.295 | +0.244 [+0.068, +0.426] **sig**, p=0.0054 | +0.267 [+0.100, +0.467] **sig**, p=0.0050 |

**How to read this:** The high-reasoning Gemini graph shows no statistically significant difference from `gpt-4o-mini` Graphiti on this n=30 subset. That does **not** show the two are equivalent: the confidence intervals are wide (about ±0.2 MRR). It is significantly better than the minimal-reasoning Gemini run, and it remains far below Locus (MRR 0.917) and OMPA (0.967). No new paired tests against Locus or OMPA were run for this variant.

**Why the minimal run returned nothing (diagnosis, prior notes).**
The episode-level `extract_edges` counts below are from earlier diagnostic notes. They were not recomputed for this publication, and the raw logs are not in this bundle. Treat them as those notes' observations, not as a new confirmed recount. The high-reasoning run was not measured per episode. What the published per-question rows support directly is default-search emptiness: 1/30 (high), 19/30 (minimal), and 5/30 (`gpt-4o-mini`). A nonempty search list is not a count of correct answers.

| | gpt-4o-mini | Gemini minimal | Gemini high |
|---|---|---|---|
| `extract_edges` returned no facts (per episode) | 16/1454 (1%) | 1070/1455 (74%) | not measured per episode* |
| Mean facts (edges) per graph | 114.2 | 48.5 | higher on validated graphs (147 and 255 vs 32 and 46 minimal) |
| Default search returned nothing | 5/30 | 19/30 | 1/30 |

*Spot checks only: 2 validation questions. Graphiti's default search uses a cosine floor of 0.6, which MiniLM rarely reaches, so in practice default search is BM25 over the extracted facts. Too few facts meant no word overlap. The harness was ruled out: no parse errors, no truncation, no null embeddings, no group_id mismatch, no stale full-text index. Raising Gemini's reasoning to `high` is a generic model setting, not tuned to the answers. Default empty results still dropped from 19/30 to 1/30, and both validation questions that had been empty now return results (5 and 13 hits), though neither has the answer in the top 5.

**Ingest time** includes free-tier rate-limit waits, daily-quota holds and 503 retries, so it is not a throughput measure.

---

# SCORECARD v7 (2026-10-29, as published in Locus PR #20 / OMPA PR #25): Graphiti on Gemini, minimal reasoning

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

\*Ingest includes free-tier waits; not a throughput measure.

---

# SCORECARD v6 (2026-10-26): Graphiti complete (n=30); every system now at n=30

Supersedes v5; v1–v5 are kept. Same design as v3–v5:
- LongMemEval-S, the same 30 questions, session retrieval.
- Paired bootstrap on MRR@10 (B=10,000, seed 20261020). n=30 is low power, and there is no multiple-comparison correction.
- `gpt-4o-mini` via OpenRouter, MiniLM embeddings for every system.

**What changed since v5:** Graphiti 0.31.2 (Zep OSS) was resumed from question 12 with exactly the same configuration as the first 11: `gpt-4o-mini`, default `add_episode`, local MiniLM, Kuzu backend, and the same two Kuzu workarounds. It is now complete at **n=30**.
- **Graphiti, default `search()`:** R@5 **0.467** [0.30, 0.63], MRR **0.428**. vs Locus: -0.489 [-0.664, -0.317], **sig** (1 question better, 18 worse). vs OMPA: -0.539 [-0.717, -0.356], **sig** (1 better, 18 worse).
- **Same graph with the cosine floor removed:** R@5 0.733 [0.57, 0.87], MRR 0.547. vs Locus: -0.369 [-0.533, -0.214], **sig**. vs OMPA: -0.419 [-0.578, -0.264], **sig**.
- **R@5 by question type (default search):** knowledge-update 0.6, multi-session 0.4, single-session-assistant 0.6, single-session-preference 0.4, single-session-user 0.0, temporal-reasoning 0.8.
- **Ingest:** 48.5 episodes per haystack on average. Ingest p50 is 438 s per haystack. In total there were 6,544 LLM calls and 80.0M prompt tokens. LLM cost was $11.68 for Graphiti overall, including the pilot and interrupted attempts, which is $0.389 per scored question.
- **One question (c9f37c46)** lost 1 of 49 episodes: `gpt-4o-mini` returned truncated JSON (JSONDecodeError). Under the harness's original rules this counts in `add_errors` and is not dropped. Mean add_errors is 0.03 per question.

**Spend.** Graphiti's resume cost $7.20 (OpenRouter `usage` went from $7.82 to $15.02). That includes about $0.48 wasted in a local-gateway budget-stop incident, whose 402-failed rows were discarded and re-run. Total OpenRouter usage for the LLM work is now $15.02.

**Gemini side-by-side (in progress, not in this table):** Graphiti is also being run on Gemini, using `gemini-3.5-flash-lite` on the free tier. That is a different LLM, so it is not directly comparable to the `gpt-4o-mini` rows. It will be reported separately when it completes.

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

---


# SCORECARD v2 (2026-10-23): adds Locus LLM query transforms (Phase 1, BEIR)

v1 is preserved as `SCORECARD_v1_20261021.md` and is reproduced unchanged below this section. Full details are in `/workspace/ompa-locus-20261020/results/llm_20261023/SUMMARY.md`.

**New rows** (Locus main 6801174; the default is now FTS5 BM25 + Porter; LLM = openai/gpt-4o-mini via OpenRouter; all test queries; paired bootstrap B=10,000, seed 20261020):

| System | SciFact nDCG@10 | NFCorpus nDCG@10 | LME R@5 | Query p50 (e2e) | $/1k queries |
|---|---|---|---|---|---|
| Locus main default (FTS5 BM25 + stemming) | 0.681 | 0.329 | 0.977 (ragpack run) | 2–4 ms | 0 |
| Locus + LLM rewrite | 0.689 (ns) | 0.347 (**+0.018 SIG**) | not run | ~0.85 s | $0.015–0.019 |
| Locus + LLM multi-query (3) | 0.684 (ns) | 0.350 (**+0.021 SIG**) | not run | ~1.1–1.4 s | $0.03–0.05 |
| Locus + LLM HyDE | 0.689 (ns) | **0.359 (+0.030 SIG)**; best NFCorpus number on this scorecard | not run | ~1.9–2.1 s | $0.07–0.08 |

**Still not run:**
- mem0 with LLM extraction, Zep/Graphiti, Letta and LangMem.
- The cause is not a missing key anymore. The OpenRouter account behind the key has never purchased credits, so it returns HTTP 402.
- The adapters, the fixed 30-question LME subset and a local Letta server are ready. See SUMMARY.md.

---

# SCORECARD: Locus and OMPA vs open-source memory and retrieval stacks (2026-10-20/21)

## Summary of wins and losses (every number below is measured on this box; see the tables)

**Where Locus (main 10ddcbf) wins**
- **Footprint.** Locus installs as 1 distribution of about 1 MB, with zero third-party runtime dependencies and no model download. That is far smaller than every competitor: rank_bm25 is 2 dists / 57 MB, LlamaIndex BM25 65 / 185, Chroma 84 / 368, and mem0 with local embeddings 86 / 5,668. OMPA with semantic search is 62 / 5,595.
- **Memory.** Peak RSS is 48 to 76 MB. Chroma peaks at 1,833 MB and OMPA at 553 MB.
- **Latency.** Locus has low single-digit-ms queries: 2.1 ms on SciFact and 1.1 ms on LongMemEval (LME), both p50. Its index build is about 3 s, while Chroma needs about 157 s to embed SciFact on this CPU.
- **Quality against some baselines.** On both BEIR sets and LME, Locus is significantly better than LangChain's default BM25Retriever: −0.150, −0.092 and −0.038 for LangChain, with all paired CIs excluding 0. On LME it is also significantly better than plain Chroma/MiniLM (Chroma −0.040).
- **Features.** Locus has point-in-time queries (`as_of`), per-chunk provenance, and a 47-tool MCP server. It is fully offline with no GPU and is MIT licensed.

**Where Locus loses (honestly)**
- **LlamaIndex BM25Retriever** (bm25s with stemming and stop-words, default settings) beats Locus on:
  - NFCorpus: +0.017 nDCG@10, significant.
  - SciFact: +0.019, with CI [−0.001, +0.039], so not significant.
  - Speed: it is also faster at query time (0.5 ms p50 on SciFact).
- **rank_bm25** beats Locus on LME R@5: 0.972 vs 0.964, a +0.0085 paired difference that is significant (4 queries better, 0 worse).
- **OMPA** has the best LME R@5 of everything run: 0.981 vs Locus 0.964, a +0.017 difference with CI [−0.002, +0.036], so not significant at 95%.
- **Locus hybrid re-rank** adds nothing over Locus BM25 on any of the three sets.
- **On BEIR**, Locus is statistically tied with rank_bm25 and Chroma.

**OMPA (main 32e9ce2)**
- **Wins:**
  - Highest LME R@5 measured here (0.981).
  - Significantly better than Chroma/MiniLM (+0.057).
  - Better on 14 queries vs 6 against Locus, though not significant.
- **Loses:**
  - Index cost is high: about 10 s per 48-session haystack at p50, and 82 minutes in total for 470 haystacks.
  - Queries are slower: 18 ms p50.
  - Footprint is heavy: 5.6 GB site-packages with default PyPI resolution (CUDA torch), 553 MB RSS.
- **Not measured:** BEIR SciFact/NFCorpus were **not run**. The run was stopped on instruction to wrap up, and the box was memory-constrained at about 4 GB free with OOM kills.

**Not run on this box (reason)**
- **mem0 OSS, local mode** (infer=False with MiniLM and local Qdrant): **not run**, for memory and time. A heavy-process slot was not available and the run was stopped on instruction. The adapter was smoke-tested on 3 LME questions, but no scored run exists, so no number is reported.
- **mem0 with LLM extraction, Zep/Graphiti, Letta, LangMem:** **not run: needs LLM API key.** Graphiti also needs a graph DB. No personal keys were used and nothing was signed up for. Their own published numbers are in section 3, marked as not reproduced and on different metrics.
- **LanceDB:** install footprint only. Chroma was the vector-DB baseline.

**Caveats**
- **Shared machine.** This is a loaded, shared 8-vCPU / 16 GB VM, so latencies are relative.
- **Chroma latency.** Chroma's absolute latency (about 185 ms p50) is dominated by CPU contention on ONNX MiniLM at 2 threads.
- **LME metric.** LME here measures session *retrieval* recall (R_any@5), not LLM-judged QA accuracy.
- **Install times.** Times are from `uv` on a datacenter network.
- **Duplicate sessions.** 12 LME haystacks contain a byte-identical duplicate session id. Chroma keeps one copy; the other systems index both, and ranking dedups by id.
- **mem0 telemetry disclosure:** during the mem0 smoke tests, mem0's default-on anonymous PostHog telemetry (`MEM0_TELEMETRY` defaults to True) was active before I disabled it. That may have sent anonymous usage events to us.i.posthog.com. Later runs set `MEM0_TELEMETRY=False` and `HF_HUB_OFFLINE=1`. Chroma 1.5.9's product telemetry is a no-op.

# Head-to-head scorecard: Locus and OMPA vs open-source memory/retrieval stacks

Generated from measured artefacts in `/workspace/scorecard-20261020`. Locus `main` @ `10ddcbf082dcb348c77da96f8b5793ca9889c3c7`, OMPA `main` @ `32e9ce2c074a8986850ad3c5f71e5665682d4e80` (the OMPA search branch another worker is changing was **not** used).

**Environment:** shared 8-vCPU / 16 GB Linux VM (load average 10–15 from other jobs during the runs), Python 3.12.14, CPU only, 2 intra-op threads per model-backed process, at most 2 heavy processes at a time. Latency is wall-clock per query on this loaded box: compare systems with each other, not with other machines.

**Data (public):** BEIR SciFact test (300 q, 5,183 docs) and NFCorpus test (323 q, 3,633 docs), nDCG@10 with graded gains; LongMemEval-S cleaned, 470 non-abstention questions, ~48 sessions per haystack, session-level retrieval, Recall_any@5 against `answer_session_ids` (retrieval only — not the LLM-judged QA accuracy reported in papers). sha256 of the inputs: `data/SHA256SUMS`.

**Statistics:** 95% CIs from a seeded bootstrap over queries (B = 10000, seed 20261020); differences are paired bootstraps over the same queries. "Significant" below means the 95% CI of the paired difference excludes 0 (no multiple-comparison correction; read borderline cases with care).

## 1. Scorecard (everything here was run on this box)

| System | SciFact nDCG@10 [95% CI] | NFCorpus nDCG@10 [95% CI] | LongMemEval-S R@5 [95% CI] | Query p50 / p95 ms (SciFact · LME) | Peak RSS MB (max of runs) | Cold install: dists · MB | Needs LLM key |
|---|---|---|---|---|---|---|---|
| Locus main (BM25, default) | 0.668 [0.623, 0.713] | 0.306 [0.272, 0.341] | 0.964 [0.947, 0.979] | 2.1 / 4.3 · 1.1 / 2.3 | 76 | 1 · 1 | no |
| Locus main (hybrid re-rank) | 0.669 [0.624, 0.714] | 0.306 [0.272, 0.341] | 0.964 [0.947, 0.979] | 6.8 / 9.8 · 6.9 / 15.4 | 76 | 1 · 1 | no |
| OMPA main (MiniLM + keyword boost) | not run | not run | 0.981 [0.968, 0.992] | – / – · 18.3 / 26.4 | 553 | 62 · 5,595 | no |
| rank_bm25 (BM25Okapi) | 0.664 [0.618, 0.709] | 0.308 [0.274, 0.343] | 0.972 [0.957, 0.985] | 8.4 / 16.3 · 0.2 / 0.5 | 124 | 2 · 57 | no |
| LlamaIndex BM25Retriever (bm25s, defaults) | 0.687 [0.644, 0.730] | 0.323 [0.288, 0.358] | 0.970 [0.953, 0.985] | 0.5 / 0.6 · 0.9 / 1.2 | 182 | 65 · 185 | no |
| LangChain BM25Retriever (defaults) | 0.518 [0.468, 0.566] | 0.214 [0.183, 0.246] | 0.925 [0.902, 0.949] | 12.6 / 25.1 · 0.3 / 0.6 | 186 | 50 · 155 | no |
| Chroma (default ONNX MiniLM, HNSW) | 0.645 [0.600, 0.690] | 0.313 [0.279, 0.346] | 0.923 [0.898, 0.947] | 184.5 / 257.7 · 184.8 / 221.1 | 1833 | 84 · 368 | no |

Install footprint = fresh Python 3.12 venv, `uv pip install --no-cache` with default PyPI resolution (so `sentence-transformers` pulls CUDA torch wheels), counting installed distributions and `site-packages` size. Run venvs used CPU-only torch. Locus figures are the git install of `main` (1 dist = Locus itself).

### Index build and per-haystack cost

| System | SciFact build s · store MB | NFCorpus build s · store MB | LME index s per haystack (p50) | LME total index min |
|---|---|---|---|---|
| Locus main (BM25, default) | 3.0 · 23.0 | 2.4 · 17.4 | 0.20 | 2.1 |
| Locus main (hybrid re-rank) | 3.0 · 23.0 | 2.4 · 17.4 | 0.20 | 2.1 |
| OMPA main (MiniLM + keyword boost) | – · – | – · – | 10.16 | 81.9 |
| rank_bm25 (BM25Okapi) | 0.5 · – | 0.4 · – | 0.02 | 0.2 |
| LlamaIndex BM25Retriever (bm25s, defaults) | 1.6 · – | 1.7 · – | 0.04 | 0.3 |
| LangChain BM25Retriever (defaults) | 0.8 · – | 0.8 · – | 0.02 | 0.2 |
| Chroma (default ONNX MiniLM, HNSW) | 156.9 · 73.2 | 101.6 · 52.9 | 1.68 | 13.3 |

Locus hybrid shares the BM25 index, so its build cost is the same index. In-memory retrievers (rank_bm25, LangChain, LlamaIndex) have no on-disk store.

### LongMemEval-S secondary metrics

| System | R_any@5 | R_all@5 | R_any@10 | nDCG@10 |
|---|---|---|---|---|
| Locus main (BM25, default) | 0.964 | 0.826 | 0.989 | 0.893 |
| Locus main (hybrid re-rank) | 0.964 | 0.826 | 0.989 | 0.892 |
| OMPA main (MiniLM + keyword boost) | 0.981 | 0.877 | 0.989 | 0.909 |
| rank_bm25 (BM25Okapi) | 0.972 | 0.832 | 0.981 | 0.902 |
| LlamaIndex BM25Retriever (bm25s, defaults) | 0.970 | 0.857 | 0.985 | 0.901 |
| LangChain BM25Retriever (defaults) | 0.925 | 0.749 | 0.962 | 0.831 |
| Chroma (default ONNX MiniLM, HNSW) | 0.923 | 0.777 | 0.968 | 0.830 |

## 2. Paired differences

**SciFact nDCG@10, system − Locus default** (better/worse/tied queries)

| System | Δ [95% CI] | Significant | better/worse/tie |
|---|---|---|---|
| Locus main (hybrid re-rank) | +0.0007 [-0.0056, +0.0080] | no | 2/5/293 |
| rank_bm25 (BM25Okapi) | -0.0045 [-0.0111, +0.0015] | no | 12/13/275 |
| LlamaIndex BM25Retriever (bm25s, defaults) | +0.0188 [-0.0011, +0.0392] | no | 42/31/227 |
| LangChain BM25Retriever (defaults) | -0.1504 [-0.1851, -0.1171] | **yes** | 17/107/176 |
| Chroma (default ONNX MiniLM, HNSW) | -0.0231 [-0.0625, +0.0161] | no | 67/77/156 |

**NFCorpus nDCG@10, system − Locus default** (better/worse/tied queries)

| System | Δ [95% CI] | Significant | better/worse/tie |
|---|---|---|---|
| Locus main (hybrid re-rank) | -0.0001 [-0.0012, +0.0009] | no | 2/5/316 |
| rank_bm25 (BM25Okapi) | +0.0017 [-0.0009, +0.0043] | no | 39/27/257 |
| LlamaIndex BM25Retriever (bm25s, defaults) | +0.0165 [+0.0069, +0.0268] | **yes** | 76/59/188 |
| LangChain BM25Retriever (defaults) | -0.0920 [-0.1134, -0.0718] | **yes** | 47/146/130 |
| Chroma (default ONNX MiniLM, HNSW) | +0.0063 [-0.0181, +0.0306] | no | 116/99/108 |

**LongMemEval-S R@5, system − Locus default** (better/worse/tied queries)

| System | Δ [95% CI] | Significant | better/worse/tie |
|---|---|---|---|
| Locus main (hybrid re-rank) | +0.0000 [+0.0000, +0.0000] | no | 0/0/470 |
| OMPA main (MiniLM + keyword boost) | +0.0170 [-0.0021, +0.0362] | no | 14/6/450 |
| rank_bm25 (BM25Okapi) | +0.0085 [+0.0021, +0.0170] | **yes** | 4/0/466 |
| LlamaIndex BM25Retriever (bm25s, defaults) | +0.0064 [-0.0064, +0.0213] | no | 7/4/459 |
| LangChain BM25Retriever (defaults) | -0.0383 [-0.0596, -0.0191] | **yes** | 3/21/446 |
| Chroma (default ONNX MiniLM, HNSW) | -0.0404 [-0.0681, -0.0128] | **yes** | 12/31/427 |

**LongMemEval-S R@5, system − OMPA** (better/worse/tied queries)

| System | Δ [95% CI] | Significant | better/worse/tie |
|---|---|---|---|
| Locus main (BM25, default) | -0.0170 [-0.0362, +0.0021] | no | 6/14/450 |
| Locus main (hybrid re-rank) | -0.0170 [-0.0362, +0.0021] | no | 6/14/450 |
| rank_bm25 (BM25Okapi) | -0.0085 [-0.0255, +0.0085] | no | 6/10/454 |
| LlamaIndex BM25Retriever (bm25s, defaults) | -0.0106 [-0.0277, +0.0064] | no | 6/11/453 |
| LangChain BM25Retriever (defaults) | -0.0553 [-0.0809, -0.0319] | **yes** | 4/30/436 |
| Chroma (default ONNX MiniLM, HNSW) | -0.0574 [-0.0787, -0.0362] | **yes** | 0/27/443 |

### LongMemEval-S R@5 by question type

| System | knowledge-update (n=72) | multi-session (n=121) | single-session-assistant (n=56) | single-session-preference (n=30) | single-session-user (n=64) | temporal-reasoning (n=127) |
|---|---|---|---|---|---|---|
| Locus main (BM25, default) | 1.000 | 0.967 | 1.000 | 0.833 | 0.984 | 0.945 |
| Locus main (hybrid re-rank) | 1.000 | 0.967 | 1.000 | 0.833 | 0.984 | 0.945 |
| OMPA main (MiniLM + keyword boost) | 0.986 | 0.984 | 1.000 | 0.933 | 0.969 | 0.984 |
| rank_bm25 (BM25Okapi) | 1.000 | 0.975 | 1.000 | 0.867 | 1.000 | 0.953 |
| LlamaIndex BM25Retriever (bm25s, defaults) | 1.000 | 0.992 | 1.000 | 0.833 | 1.000 | 0.937 |
| LangChain BM25Retriever (defaults) | 1.000 | 0.950 | 0.964 | 0.600 | 0.953 | 0.905 |
| Chroma (default ONNX MiniLM, HNSW) | 0.972 | 0.950 | 0.982 | 0.833 | 0.812 | 0.921 |

## 3. Not run here, and published numbers (not reproduced)

| System | Status here | Their published number | Benchmark / metric (not comparable to R@5 or nDCG above unless stated) | Source |
|---|---|---|---|---|
| Zep (cloud) / Graphiti | not run: needs an LLM API key (graph construction) and a graph DB (Neo4j/FalkorDB) | 71.2% (gpt-4o), 63.8% (gpt-4o-mini); their full-context baseline 60.2% (gpt-4o) | LongMemEval-S end-to-end QA accuracy, GPT-4o judge (not retrieval R@5) | https://arxiv.org/abs/2501.13956 (Table 2) |
| mem0 with LLM extraction (default infer=True) | not run: needs an LLM API key; a local LLM was not attempted (22,419 LME sessions each need extraction + update LLM calls, far beyond the CPU/time budget of this shared box). mem0 was run with infer=False instead (section 1) | 66.88% (Mem0), 68.44% (Mem0-graph); search p50 0.148 s | LoCoMo LLM-as-judge J (vendor paper; Zep disputes the Zep numbers in it) | https://arxiv.org/abs/2504.19413 (Table 2) |
| Letta | not run: the agent loop needs an LLM API key | 74.0% (gpt-4o-mini, filesystem agent) | LoCoMo QA accuracy (vendor blog) | https://www.letta.com/blog/benchmarking-ai-agent-memory/ |
| LangChain LLM memory (LangMem) | not run: needs an LLM API key (the LangChain BM25 retriever was run, section 1) | 58.10% | LoCoMo J, as measured by the mem0 authors (third party) | https://arxiv.org/abs/2504.19413 (Table 2) |
| LongMemEval authors' retrievers | reference only | Recall@5: BM25 0.634, Contriever 0.723, Stella V5 1.5B 0.720 | LongMemEval-M (~500 sessions per question), session values, user turns only — harder than the -S setting used here | https://arxiv.org/abs/2410.10813 (Table 9) |

Published reference points for the baselines that *were* run here (also not reproduced by us, listed for sanity):

- BEIR paper BM25 (Elasticsearch) nDCG@10: SciFact 0.665, NFCorpus 0.325 — https://arxiv.org/abs/2104.08663 (Table 2)
- all-MiniLM-L6-v2 MTEB nDCG@10: SciFact 0.645 (64.508), NFCorpus 0.316 (31.594) — https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/97ef89b0452be49b401bb722c776152b035909e7/README.md

## 4. Feature matrix (from docs and source; Y = yes, P = partial / conditional, N = no or not found)

| System | Point-in-time (temporal) queries | Provenance / citations | Audit trail | Offline, zero-GPU | Licence | MCP |
|---|---|---|---|---|---|---|
| Locus (main 10ddcbf) | Y | Y | P | Y | MIT | Y |
| OMPA (main 32e9ce2) | Y | P | P | Y | MIT | Y |
| mem0 OSS (mem0ai 2.2.1) | N | P | Y | P | Apache-2.0 | P |
| Zep / Graphiti (graphiti-core) | Y | Y | P | N | Apache-2.0 (Graphiti); Zep cloud is commercial | Y |
| Letta | N | P | P | N | Apache-2.0 | Y |
| LangChain BM25Retriever | N | P | N | Y | MIT | P |
| LlamaIndex BM25Retriever | N | P | N | Y | MIT (core and bm25s) | P |
| Chroma | P | P | N | Y | Apache-2.0 | Y |
| rank_bm25 | N | N | N | Y | Apache-2.0 | N |

Detail and source for every cell:

- **Locus (main 10ddcbf)**: Point-in-time (temporal) queries: Y — KG facts with valid_from/valid_to; retrieve(as_of=...), query_entity --as-of (https://github.com/jmiaie/locus/blob/10ddcbf082dcb348c77da96f8b5793ca9889c3c7/README.md); Provenance / citations: Y — every chunk carries doc_path + provenance tag (bm25/kg/link...); explain(chunk_id, query) (https://github.com/jmiaie/locus/blob/10ddcbf082dcb348c77da96f8b5793ca9889c3c7/README.md); Audit trail: P — opt-in retrieval log (track=True records hits in the bulletin); no tamper-evident audit log (README points to CognitionOS for audit) (https://github.com/jmiaie/locus/blob/10ddcbf082dcb348c77da96f8b5793ca9889c3c7/locus/core.py); Offline, zero-GPU: Y — pure Python + SQLite FTS5, no model download, zero third-party runtime deps (measured) (measured: results/install_footprint.jsonl); Licence: MIT (https://github.com/jmiaie/locus/blob/10ddcbf082dcb348c77da96f8b5793ca9889c3c7/LICENSE); MCP: Y — 47 tools, stdio + HTTP JSON-RPC server (https://github.com/jmiaie/locus/blob/10ddcbf082dcb348c77da96f8b5793ca9889c3c7/README.md)
- **OMPA (main 32e9ce2)**: Point-in-time (temporal) queries: Y — kg.query_entity(name, as_of=...) over valid_from/valid_to triples; history preserved since #16 (https://github.com/jmiaie/ompa/blob/32e9ce2c074a8986850ad3c5f71e5665682d4e80/ompa/knowledge_graph.py); Provenance / citations: P — search results carry the note path + excerpt; no per-signal provenance (https://github.com/jmiaie/ompa/blob/32e9ce2c074a8986850ad3c5f71e5665682d4e80/ompa/semantic.py); Audit trail: P — KG timeline(entity); vault notes are plain markdown (git-able); no access log (https://github.com/jmiaie/ompa/blob/32e9ce2c074a8986850ad3c5f71e5665682d4e80/ompa/knowledge_graph.py); Offline, zero-GPU: Y — CPU sentence-transformers MiniLM (one-time model download) or keyword fallback (https://github.com/jmiaie/ompa/blob/32e9ce2c074a8986850ad3c5f71e5665682d4e80/pyproject.toml); Licence: MIT (https://github.com/jmiaie/ompa/blob/32e9ce2c074a8986850ad3c5f71e5665682d4e80/LICENSE); MCP: Y — ao-mcp server, 15 tools (https://github.com/jmiaie/ompa/blob/32e9ce2c074a8986850ad3c5f71e5665682d4e80/README.md)
- **mem0 OSS (mem0ai 2.2.1)**: Point-in-time (temporal) queries: N — no as-of query in the Memory API; graph variant marks relations invalid instead of deleting (paper) (https://github.com/mem0ai/mem0/blob/main/AGENTS.md ; https://arxiv.org/abs/2504.19413); Provenance / citations: P — user/agent/run ids + free-form metadata per memory; extracted facts are LLM rewrites, not source spans (https://github.com/mem0ai/mem0/blob/main/AGENTS.md); Audit trail: Y — history(memory_id) returns the ADD/UPDATE/DELETE change history (SQLite history db) (https://github.com/mem0ai/mem0/blob/main/AGENTS.md); Offline, zero-GPU: P — default pipeline calls an LLM (OpenAI) for extraction; infer=False + local embedder runs offline on CPU (measured here) (https://docs.mem0.ai); Licence: Apache-2.0 (https://github.com/mem0ai/mem0); MCP: P — hosted MCP at mcp.mem0.ai; local OpenMemory MCP is marked as being sunset (https://github.com/mem0ai/mem0/tree/main/openmemory ; https://github.com/mem0ai/mem0/blob/main/AGENTS.md)
- **Zep / Graphiti (graphiti-core)**: Point-in-time (temporal) queries: Y — bi-temporal edges (valid_at/invalid_at + created/expired); search_memory_facts date filters (https://github.com/getzep/graphiti/blob/main/mcp_server/README.md ; https://arxiv.org/abs/2501.13956); Provenance / citations: Y — episodes are kept non-lossy; get_episode_entities traces facts back to episodes (https://github.com/getzep/graphiti/blob/main/mcp_server/README.md); Audit trail: P — transaction-time timeline (created/expired) on facts; not an access log (https://arxiv.org/abs/2501.13956); Offline, zero-GPU: N — needs an LLM API for extraction plus a graph DB (Neo4j/FalkorDB); MCP server 'requires a configured database and valid API keys' (https://github.com/getzep/graphiti/blob/main/mcp_server/src/graphiti_mcp_server.py); Licence: Apache-2.0 (Graphiti); Zep cloud is commercial (https://github.com/getzep/graphiti/blob/main/LICENSE); MCP: Y — reference MCP server (no auth; not for untrusted networks) (https://github.com/getzep/graphiti/blob/main/mcp_server/README.md)
- **Letta**: Point-in-time (temporal) queries: N — not found in the docs reviewed (https://docs.letta.com/); Provenance / citations: P — agent tools return file/passage content; no citation contract found in docs reviewed (https://www.letta.com/blog/letta-filesystem/); Audit trail: P — persisted agent state/messages on the server; no audit API found in docs reviewed (https://docs.letta.com/); Offline, zero-GPU: N — agent loop requires an LLM; published benchmark used gpt-4o-mini + OpenAI embeddings (https://www.letta.com/blog/benchmarking-ai-agent-memory/); Licence: Apache-2.0 (https://github.com/letta-ai/letta); MCP: Y — MCP client: agents consume MCP servers (stdio/HTTP/SSE) (https://docs.letta.com/agent-sdk/mcp/)
- **LangChain BM25Retriever**: Point-in-time (temporal) queries: N — stateless retriever (https://python.langchain.com/); Provenance / citations: P — returns Documents with caller-supplied metadata (https://python.langchain.com/); Audit trail: N; Offline, zero-GPU: Y — in-memory rank_bm25 (measured); Licence: MIT — installed package metadata (https://github.com/langchain-ai/langchain); MCP: P — MCP client adapters (langchain.mcp) (https://docs.langchain.com/oss/python/langchain/mcp)
- **LlamaIndex BM25Retriever**: Point-in-time (temporal) queries: N — stateless retriever (metadata filters only) (https://developers.llamaindex.ai/); Provenance / citations: P — returns NodeWithScore with node metadata (https://developers.llamaindex.ai/); Audit trail: N; Offline, zero-GPU: Y — bm25s, in-memory (persistable) (measured); Licence: MIT (core and bm25s) — installed package metadata (https://github.com/run-llama/llama_index); MCP: P — MCP client tool spec (llama-index-tools-mcp) (https://developers.llamaindex.ai/python/framework-api-reference/tools/mcp/)
- **Chroma**: Point-in-time (temporal) queries: P — only via user metadata filters (https://docs.trychroma.com/); Provenance / citations: P — ids + metadata returned with each hit (https://docs.trychroma.com/); Audit trail: N; Offline, zero-GPU: Y — local ONNX MiniLM on CPU (one-time model download, measured) (measured); Licence: Apache-2.0 — installed package metadata (https://github.com/chroma-core/chroma); MCP: Y — official chroma-mcp server (https://github.com/chroma-core/chroma-mcp)
- **rank_bm25**: Point-in-time (temporal) queries: N; Provenance / citations: N — returns scores by position only (https://github.com/dorianbrown/rank_bm25); Audit trail: N; Offline, zero-GPU: Y — numpy only (measured); Licence: Apache-2.0 — installed package metadata (https://github.com/dorianbrown/rank_bm25); MCP: N

## 5. Install footprint (all candidates; fresh py3.12 venv, `uv pip install --no-cache`, default PyPI resolution)

| Package (spec) | Installed dists | site-packages MB | Install s | torch/nvidia dists |
|---|---|---|---|---|
| Locus main (git @10ddcbf) | 1 | 1 | 1.6 | 0 |
| locus-rag 1.0.0 (PyPI) | 1 | 1 | 0.2 | 0 |
| rank_bm25 | 2 | 57 | 0.4 | 0 |
| OMPA main core | 10 | 12 | 1.9 | 0 |
| OMPA main [semantic] | 62 | 5,595 | 17.3 | 16 |
| chromadb | 84 | 368 | 1.8 | 0 |
| lancedb | 16 | 411 | 1.4 | 0 |
| mem0ai | 34 | 134 | 1.1 | 0 |
| mem0ai + sentence-transformers (local embed) | 86 | 5,668 | 17.7 | 16 |
| graphiti-core (needs neo4j + LLM key at runtime) | 27 | 95 | 0.9 | 0 |
| letta 0.34.12 | 1 | 254 | 1.8 | 0 |
| langchain-community BM25 (+rank_bm25) | 50 | 155 | 1.2 | 0 |
| llama-index BM25 (bm25s) | 65 | 185 | 1.4 | 0 |

Source: `results/install_footprint.jsonl`, with freezes and logs in `installs/`. The CPU-only torch wheel would shrink the semantic installs, but the default `pip install` pulls CUDA wheels.

## 6. Artefacts (all under /workspace/scorecard-20261020/)

- `SCORECARD.md` (this file); `scorecard.csv` and `scorecard.json` (one row per system); `SCORECARD_metrics.json` (all metrics, CIs, paired tests); `scorecard_quality_latency.csv`.
- `charts/scorecard.png`: the three quality panels with 95% CIs, plus quality vs p50 latency.
- `features.json` (feature matrix with URLs); `published.json` (published, not reproduced numbers with URLs).
- `results/*.jsonl` (per-query rankings and latencies) and `*.meta.json` (peak RSS, index time, store size).
- `harness/` (common.py, run.py, analyze.py, report.py, chain scripts); `data/SHA256SUMS`.
- `repos/` (fresh clones: locus @10ddcbf and ompa @32e9ce2; the OMPA remote was not touched); `venvs/` (run environments); `logs/`.
