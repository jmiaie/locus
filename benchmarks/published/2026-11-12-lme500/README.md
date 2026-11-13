# LongMemEval-S, all 500 questions: Locus and OMPA (2026-11-12)

Session-level retrieval on `longmemeval_s_cleaned.json` (500 questions, about 48 sessions per history). The primary
set is the **470 non-abstention questions**; the 30 `_abs` questions have no answer session. No LLM calls, $0.
Local `all-MiniLM-L6-v2` embeddings (OMPA hybrid), CPU only, shared 8-vCPU VM, so latencies are indicative.

- Locus: FTS5 BM25 (`mode="bm25"`, `limit=100` chunks, deduplicated to the top 10 sessions).
- OMPA: hybrid MiniLM + keyword, `search(limit=10)`.
- Metrics: R@5 (any answer session in the top 5), R_all@5 (all answer sessions in the top 5), MRR@10, NDCG@10.
- Stats: percentile bootstrap, B=10,000, seed 20261020; OMPA − Locus is paired.

| System (470 q) | R@5 [95% CI] | R_all@5 | MRR@10 [95% CI] | query p50 / p95 | ingest p50 per history |
|---|---|---|---|---|---|
| Locus | 0.977 [0.962, 0.989] | 0.860 | 0.911 [0.889, 0.931] | 2.3 / 3.9 ms | 0.22 s |
| OMPA | 0.981 [0.968, 0.992] | 0.877 | 0.933 [0.913, 0.950] | 26.0 / 57.7 ms | 10.1 s |

OMPA − Locus: ΔMRR +0.022 [+0.005, +0.039], significant, almost all of it on single-session-preference questions;
ΔR@5 +0.004 [−0.009, +0.017], not significant.

Files:
- `summary500__locus_ompa.json`: all metrics, per-type breakdown with CIs, paired tests, and the all-500 numbers.
- `run500.py`, `analyze500.py`, `lme500_qids.json`: the harness and analysis as run. They reference local paths of the
  original run (dataset and checkouts), which are not included. Per-question top-10 rows for the same configuration
  are in [`../2026-11-12-ranking/`](../2026-11-12-ranking/) (`rows__*.jsonl`, variant `baseline`).
- `published_competitors.md`: vendors' own published numbers, with sources. They are LLM-judged answer accuracy and
  are **not comparable** to the retrieval metrics above.
