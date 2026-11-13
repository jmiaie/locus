# Opt-in ranking features on LongMemEval-S (2026-11-12)

Re-measurement of the **implemented** opt-in features (`locus.dense.DenseReranker`, `retrieve(..., temporal=True)` in
Locus; `Ompa.search(..., temporal=True)` in OMPA) on LongMemEval-S, 470 non-abstention questions, session-level
retrieval. No LLM calls; local `all-MiniLM-L6-v2` embeddings on CPU; $0.

- Harness: `remeasure.py` (same data and settings as the 2026-11-12 LME-500 runs: Locus FTS5 BM25, `limit=100` chunks
  deduplicated to the top 10 sessions; OMPA hybrid MiniLM + keyword, `search(limit=10)`). Each session document carries
  its session date (Locus: YAML frontmatter `date:`; OMPA: `write(date=...)`); `reference_date` = the question date.
- Stats: `analyze_rm.py` — paired bootstrap vs the in-harness baseline, B=10,000, seed 20261020, percentile 95% CIs.
- `compare_proto.py` — per-question agreement with the earlier prototypes (`prototype__*.out`; the prototype re-ranked a
  cached candidate set and was not a repo change). These scripts reference local paths of the original run
  (dataset and prototype cache are not included here).
- Rows: `rows__locus.jsonl`, `rows__ompa.jsonl` (top-10 sessions per variant). Summaries: `summary__*.json`.

Baseline check: the Locus in-harness baseline is identical to the original LME-500 run on 470/470 questions (the date
frontmatter does not change BM25). OMPA's is identical on 314/470: the frontmatter date value slightly changes OMPA's
index, so OMPA deltas are against its own dated baseline (MRR 0.9323 vs 0.9325 in the original run).

## Locus (n=470, baseline MRR@10 0.9107)

| Variant | MRR@10 | ΔMRR [95% CI] | ΔR_all@5 | per-type highlights | prototype ΔMRR |
|---|---|---|---|---|---|
| `temporal=True` | 0.9116 | +0.0009 [−0.0081, +0.0100] ns | −0.0043 ns | temporal-reasoning +0.018 ns; multi-session −0.007 ns; single-session-user −0.016 ns; 22 q changed | +0.0011 ns |
| `DenseReranker()` | 0.9343 | **+0.0236 [+0.0078, +0.0396] sig** | **+0.0277 [+0.0064, +0.0511] sig** | single-session-preference 0.572 → 0.793 (+0.221 sig) | +0.0132 ns (ΔR_all +0.0298 sig) |
| `DenseReranker()` + `temporal=True` | 0.9374 | **+0.0267 [+0.0087, +0.0449] sig** | +0.0170 ns | preference +0.221 sig; temporal-reasoning 0.872 → 0.926 (+0.054 sig); single-session-user −0.025 ns | +0.0160 ns |
| `DenseReranker(route="advice")` (**post-hoc**) | 0.9239 | **+0.0132 [+0.0059, +0.0217] sig** (12 better / 0 worse) | 0.000 | preference +0.206 sig; every other type unchanged | +0.0121 sig (12/1) |

Agreement with the prototype (same top-1 session): temporal 468/470, dense 459/470, dense+temporal 457/470, advice
router 469/470. Top-10 lists differ more for dense (identical on 198/470) because the implementation scores only the
documents in the 300-chunk BM25 pool, while the prototype ranked every session.

Latency (shared 8-core VM under load, 2 torch threads; indicative): BM25 p50 3.1 ms. Dense, first query on a fresh
~48-session haystack (embeds every candidate session): p50 12.3 s, p95 18.1 s. With the embedding cache warm (second
query on the same engine, dense + temporal): p50 22 ms, p95 56 ms. Temporal alone adds no measurable latency.

## OMPA (n=470, dated baseline MRR@10 0.9323)

| Variant | MRR@10 | ΔMRR [95% CI] | ΔR_all@5 | per-type highlights | prototype ΔMRR |
|---|---|---|---|---|---|
| `search(temporal=True)` | 0.9351 | +0.0028 [−0.0063, +0.0121] ns | −0.0064 ns | temporal-reasoning 0.893 → 0.918 (**+0.025 [+0.002, +0.053] sig**); multi-session −0.008 ns; single-session-user −0.014 ns; 22 q changed | +0.0039 ns (temporal-reasoning +0.029 sig) |

Same top-1 session as the prototype on 464/470. Search latency p50 28.5 ms with or without temporal.

## Caveats

- **Post-hoc choices.** The advice router regex was written after seeing per-type results (it fires on 25 of the 30
  single-session-preference questions and on no other type: close to a question-type oracle on this dataset). The
  temporal parser was narrowed after the prototype (duration ranges such as "past few months" and windows wider than 45
  days are ignored). Narrowing did not remove the small, non-significant losses on multi-session / single-session-user;
  those come from point-in-time phrases ("last month", "last weekend"). Neither choice has been validated on other data.
- One dataset (LongMemEval-S); session-level retrieval only, not end-to-end answer accuracy.
- A general recency prior was measured in the prototype as strongly harmful (ΔMRR −0.24, sig) and is not implemented.
