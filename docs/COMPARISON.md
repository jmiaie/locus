<!-- Measured 2026-10-20/21 on a shared CPU VM. Raw per-query results, harness and artefacts were kept outside this repo; numbers are copied verbatim from that run. Locus main @10ddcbf, OMPA main @32e9ce2. -->
# Comparison: Locus and OMPA vs open-source memory and retrieval stacks (2026-10-20/21)

> **Update, 2026-10-21: Locus default analyzer is now `porter+english`** (Porter stemming plus English stop-words; see
> [`benchmarks/README.md`](../benchmarks/README.md#analyzer-porter-stemming--english-stop-words-2026-10-21)).
> The tables below were measured on 2026-10-20 at main `10ddcbf`, which used the `plain` analyzer. They are kept unchanged.
> With the new default, measured on the same queries (BM25 mode; paired bootstrap, 95% CI):
>
> | | SciFact nDCG@10 | NFCorpus nDCG@10 | LongMemEval-S R@5 |
> |---|---|---|---|
> | Locus, new default | 0.681 [0.638, 0.724] | 0.329 [0.294, 0.364] | 0.977 [0.962, 0.989] |
> | Δ vs Locus `plain` (10ddcbf) | +0.013 (ns) | **+0.022** (sig.) | **+0.013** (sig.) |
> | Δ vs LlamaIndex BM25 | −0.006 (ns) | **+0.006** (sig.) | +0.006 (ns) |
> | Δ vs rank_bm25 | +0.017 (ns) | **+0.021** (sig.) | +0.004 (ns) |
> | Δ vs OMPA main 32e9ce2 | not run | not run | −0.004 (ns) |
>
> What changed in the comparison:
> - LlamaIndex BM25 no longer beats Locus significantly anywhere. Locus now wins significantly on NFCorpus.
> - LlamaIndex is still ahead on SciFact (not significant) and is still faster per query: 0.5 ms vs 2.3 ms p50 on SciFact.
> - rank_bm25's significant LongMemEval-S lead is gone.
> - OMPA's LongMemEval-S score is now within noise of Locus.
> - Locus query latency rose 1.23× on SciFact and 1.01× on NFCorpus, both p50.
> - The win/loss bullets below describe the 10ddcbf (`plain`) measurement.

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

# Head-to-head scorecard: Locus and OMPA vs open-source memory/retrieval stacks

Locus `main` @ `10ddcbf082dcb348c77da96f8b5793ca9889c3c7`, OMPA `main` @ `32e9ce2c074a8986850ad3c5f71e5665682d4e80` (the OMPA search branch another worker is changing was **not** used).

**Environment:** shared 8-vCPU / 16 GB Linux VM (load average 10–15 from other jobs during the runs), Python 3.12.14, CPU only, 2 intra-op threads per model-backed process, at most 2 heavy processes at a time. Latency is wall-clock per query on this loaded box: compare systems with each other, not with other machines.

**Data (public):** BEIR SciFact test (300 q, 5,183 docs) and NFCorpus test (323 q, 3,633 docs), nDCG@10 with graded gains; LongMemEval-S cleaned, 470 non-abstention questions, ~48 sessions per haystack, session-level retrieval, Recall_any@5 against `answer_session_ids` (retrieval only — not the LLM-judged QA accuracy reported in papers). 

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

The CPU-only torch wheel would shrink the semantic installs, but the default `pip install` pulls CUDA wheels.

## 6. How this was measured

- **Systems.** Each system is run through one adapter harness over the same public data:
  - BEIR SciFact and NFCorpus test splits.
  - LongMemEval-S cleaned, 470 non-abstention questions.
- **Locus and OMPA input.** Locus and OMPA index each document as markdown (`# title` + text). All other systems get `title\n\ntext`.
- **Library settings.** Every library is used with its documented defaults unless noted:
  - rank_bm25 uses the Locus tokenizer.
  - Chroma uses its default ONNX all-MiniLM-L6-v2 embedder.
  - OMPA uses `enable_semantic=True` with all-MiniLM-L6-v2.
- **Threads.** Model-backed processes were capped at 2 intra-op threads.
- **Not reproduced.** Vendor numbers in section 3 come from their own papers and blogs and were **not** reproduced. Their metrics (QA accuracy, LoCoMo J) are not comparable with retrieval recall.
- **Interpretation.** Read the paired-difference tables, not the leaderboard order. Several gaps are inside the noise.
