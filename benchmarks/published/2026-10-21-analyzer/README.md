# Analyzer (Porter stemming + English stop-words), 2026-10-21

As-run artefacts for the change that made `porter+english` the default analyzer.

Setup:
- Shared 8-vCPU / 16 GB VM, Python 3.12.
- One light process at a time.
- Seeded bootstrap, B = 10,000, seed 20261020.

The scripts are kept exactly as they were run. They use absolute paths on the benchmark box: BEIR data under
`/workspace/ompa-locus-20261020/data` and the scorecard harness under `/workspace/scorecard-20261020`.
Change those paths to re-run them.

| File | What |
|---|---|
| `sweep_scifact_train.json`, `sweep_nfcorpus_dev.json` | **Dev only.** Doc-level BM25 simulation (FTS5 IDF): analyzer × k1 ∈ {0.6, 0.9, 1.2, 1.5, 2.0} × b ∈ {0.3, 0.5, 0.75, 0.9} (`scripts/sweep.py`). |
| `dev_engine.json` | **Dev only.** The real engine, BM25 mode, three analyzers (`scripts/dev_engine.py`). |
| `compare.json` | **Test.** BM25 mode, nDCG@10 (BEIR) and R_any@5 (LongMemEval-S, 470 q). Includes per-system CIs and paired bootstraps against Locus main @9e62d95, rank_bm25, LlamaIndex BM25 and the branch `plain` analyzer, plus OMPA on LME (`scripts/compare.py`). |
| `compare_hybrid.json` | The same comparison for `mode="hybrid"`. |
| `latency.json` | In-process interleaved latency: 5 rounds alternating analyzers, median per query, then p50/p95 (`scripts/latency.py`). |

Test runs used the scorecard harness (`harness/run.py locus <bench>` with `LOCUS_ANALYZER=plain|porter|porter+english`).
The baselines are the 2026-10-20 scorecard runs (see `docs/COMPARISON.md`).
The `plain` analyzer on this branch reproduced Locus main's rankings exactly on all 1,093 test queries.
