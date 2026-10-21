# Locus Benchmarks

Everything here is reproducible from this repository plus two public BEIR datasets.
Numbers quoted in the top-level README come from these runs (2026-10-20, Locus branch
`feat/fts5-backend-and-honest-benchmarks`, compared against `main` at `62156b5`).

**Environment:** shared 8-vCPU / 16 GB Linux VM, Python 3.13.5, SQLite 3.46.1, no GPU.
Another benchmark job was running on the same machine, so latencies are indicative only;
quality numbers are unaffected by load.

---

## 1. Public datasets (BEIR)

| | SciFact | NFCorpus |
|---|---|---|
| Corpus | 5,183 abstracts | 3,633 docs |
| Dev split (tuning only) | `qrels/train.tsv`, 809 q | `qrels/dev.tsv`, 324 q |
| Test split (reported) | `qrels/test.tsv`, 300 q | `qrels/test.tsv`, 323 q |

Download the zips from the BEIR project (`https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`
and `.../nfcorpus.zip`) and unzip them under one directory, e.g. `data/beir/scifact/`, `data/beir/nfcorpus/`.

```bash
pip install -e ".[bench]" numpy
export LOCUS_BEIR_DIR=data/beir LOCUS_BENCH_OUT=benchmarks/results/repro
python benchmarks/repro/beir_eval.py scifact      # ~3 min: index + dev grid + test
python benchmarks/repro/beir_eval.py nfcorpus
python benchmarks/repro/analyze_beir.py           # metrics, CIs, dev-only selection, paired tests
python benchmarks/repro/mc_subcorpus.py           # 10-seed Monte Carlo over SciFact sub-corpora
```

Protocol:

- Each doc becomes `# {title}\n\n{text}` in its own markdown file; Locus indexes the folder.
  Chunk hits are mapped back to doc ids (first occurrence), top 10 kept.
- `rank_bm25` (`BM25Okapi`, k1=1.5, b=0.75) uses Locus's tokenizer on `title + text`.
- nDCG@10 uses graded gains (2^rel − 1). 95% CIs: seeded bootstrap over queries, B = 10,000, seed 20261020.
  Differences are **paired** bootstraps over the same queries.
- Hybrid re-rank grid: `rerank_alpha` ∈ {0.1, 0.25, 0.5, 1.0} × `rerank_depth` ∈ {40, 100, 200}.
  The configuration is chosen by mean **dev** nDCG@10 across both datasets; test labels are never used for tuning.

### Dev grid (nDCG@10, dev splits)

| Config | SciFact-train | NFCorpus-dev |
|---|---|---|
| **BM25 (default)** | **0.6680** | 0.2650 |
| rank_bm25 | 0.6672 | 0.2652 |
| hybrid α=0.1 (depth 40/100/200) | 0.6661 | 0.2651 |
| hybrid α=0.25, depth 100 | 0.6617 | 0.2645 |
| hybrid α=0.5, depth 40 | 0.6602 | 0.2636 |
| hybrid α=1.0, depth 200 | 0.6553 | 0.2613 |

Every hybrid configuration scored at or below BM25 on dev, and larger α was monotonically worse.
α=0.1 at depths 40, 100 and 200 scored identically; depth 100 was selected (ties broken by name)
and α=0.1 / depth 100 is the shipped hybrid default.

### Test results

| System | SciFact nDCG@10 [95% CI] | NFCorpus nDCG@10 [95% CI] | p50 / p95 ms (SciFact) | p50 / p95 ms (NFCorpus) |
|---|---|---|---|---|
| **Locus default — BM25 on FTS5** | **0.6682** [0.6225, 0.7128] | **0.3063** [0.2723, 0.3412] | 2.7 / 4.9 | 1.2 / 2.4 |
| Locus hybrid (α=0.1, depth 100) | 0.6688 [0.6238, 0.7136] | 0.3063 [0.2722, 0.3411] | 7.7 / 12.1 | 4.6 / 9.2 |
| rank_bm25 | 0.6637 [0.6184, 0.7086] | 0.3080 [0.2740, 0.3425] | 7.2 / 13.5 | 1.6 / 3.8 |
| *before: 1.0.0 six-signal default* | *0.5810* [0.5375, 0.6244] | *0.2843* [0.2510, 0.3183] | *878 / 1,511* | *404 / 817* |
| *before: 1.0.0 BM25-only path* | *0.6404* [0.5943, 0.6866] | *0.2939* [0.2597, 0.3286] | *205 / 564* | *27 / 242* |

Paired differences in nDCG@10 (mean [95% CI], bootstrap P(diff ≤ 0)):

| Comparison | SciFact | NFCorpus |
|---|---|---|
| default − rank_bm25 | +0.0045 [−0.0015, +0.0111], P=0.07 | −0.0017 [−0.0043, +0.0009], P=0.90 |
| hybrid − default | +0.0007 [−0.0056, +0.0080], P=0.44 | −0.0001 [−0.0012, +0.0009], P=0.54 |
| default − before six-signal | +0.0872 [+0.0636, +0.1111], P<0.001 | +0.0220 [+0.0138, +0.0305], P<0.001 |
| default − before BM25-only | +0.0278 [+0.0102, +0.0454], P<0.001 | +0.0124 [+0.0053, +0.0197], P<0.001 |

**Decision:** the hybrid re-rank does not significantly beat BM25 on either dataset, so
`mode="bm25"` is the default and hybrid is opt-in. Locus BM25 is at parity with `rank_bm25`
(neither difference is significant). Re-running the default on the test split gave identical
rankings for 300/300 and 323/323 queries.

### Monte Carlo sub-corpora (SciFact)

10 seeds; each samples 100 test queries and builds a 2,000-doc corpus from their relevant docs plus random distractors.

| System | mean nDCG@10 | min | max |
|---|---|---|---|
| Locus default (BM25) | 0.7418 | 0.7066 | 0.8234 |
| Locus hybrid | 0.7363 | 0.7005 | 0.8234 |
| rank_bm25 | 0.7420 | 0.7035 | 0.8265 |
| *before: six-signal / BM25-only* | *0.628 / 0.721* | | |

Hybrid was below BM25 in 8 seeds, equal in 1, above in 1. BM25 vs rank_bm25: higher in 4 seeds, lower in 6, mean −0.0002.

---

## 2. Scaling (synthetic)

```bash
python benchmarks/repro/scale.py 1000 100 1 --rank-bm25 --tag after
python benchmarks/repro/scale.py 10000 100 1 --rank-bm25 --tag after
python benchmarks/repro/scale.py 100000 100 1 --hybrid-queries 50 --tag after   # ~3 min, ~420 MB disk
```

Docs are 150–250 tokens drawn Zipf(1.1) from a 20k pseudo-word vocabulary plus a unique two-token
"needle"; each query is a needle plus 3 Zipf-sampled words (known-item search: every current system
scores MRR@10 1.0; the old six-signal pipeline scored 0.89 at 1k and 0.98 at 10k — this is a
latency/size test, not a quality test). Three warm-up queries are excluded.

| Docs | System | Index build | Store | p50 ms | p95 ms |
|---|---|---|---|---|---|
| 1,000 | **Locus default** | 0.24 s | 4.4 MB | 1.5 | 2.2 |
| | Locus hybrid | | | 5.4 | 9.3 |
| | rank_bm25 | 0.07 s | (RAM) | 0.7 | 1.0 |
| | *before: BM25-only / six-signal* | *6.0 s* | *13 MB* | *207 / 521* | *317 / 755* |
| 10,000 | **Locus default** | 2.5 s | 43 MB | 10.5 | 16.3 |
| | Locus hybrid | | | 14.5 | 19.0 |
| | rank_bm25 | 0.7 s | (RAM) | 10.5 | 12.6 |
| | *before: BM25-only / six-signal (50 q)* | *85 s* | *130 MB* | *1,361 / 3,966* | *2,688 / 6,458* |
| 100,000 | **Locus default** | 34 s | 419 MB | 93 | 137 |
| | Locus hybrid (50 q) | | | 98 | 143 |
| | *before: BM25-only / six-signal (20 q)* | *21.7 min* | *1,324 MB* | *19,033 / 54,604* | *37,112 / 91,647* |

Sizes are MiB. `rank_bm25` was not re-run at 100k to keep memory bounded on the shared machine
(the earlier run measured p50 179 ms, ~1.8 GB RSS). The "before" rows come from the same generator
and seed, run on the 1.0.0 code earlier the same day.

**Caveats:** p95 at 100k is above 100 ms. Query time still grows roughly linearly here because
every synthetic query contains very common words, and FTS5's `bm25()` scores every chunk that matches
any query term; there is no WAND/MaxScore pruning. The synthetic corpus produces no KG triples, so
hybrid latency on KG-heavy corpora at this scale is not measured.

---

## 3. Bundled synthetic benchmarks and determinism

```bash
PYTHON=python benchmarks/repro/repeat.sh . after          # 5x wiki + 5x simple-100, fresh process each,
python benchmarks/repro/analyze_repeats.py                # PYTHONHASHSEED = 7919·i
python benchmarks/run.py --no-latency                     # one wiki run
python benchmarks/run.py --simple --docs 100 --no-latency # one simple-100 run
```

### Engineering Wiki (113 typed queries, generated by `locus/bench/wiki.py`)

| Config | R@1 | R@3 | R@5 | MRR | ms/q |
|---|---|---|---|---|---|
| bm25_only (**default**) | 0.584 | 0.796 | 1.000 | 0.727 | 0.5 |
| bm25 + kg | 0.575 | 0.885 | 1.000 | 0.744 | 0.9 |
| bm25 + kg + links | 0.575 | 0.885 | 1.000 | 0.744 | 1.4 |
| bm25 + kg + links + structural + recency | 0.557 | 0.956 | 1.000 | 0.747 | 1.7 |
| hybrid_all (`mode="hybrid"`) | 0.549 | 0.956 | 1.000 | 0.743 | 1.8 |
| hybrid_all + heuristic reranker | 0.611 | 0.779 | 1.000 | 0.737 | 3.5 |
| rank_bm25 | 0.540 | 0.991 | 1.000 | 0.760 | 0.0 |
| grep_literal | 0.522 | 0.911 | 1.000 | 0.729 | 0.1 |

Hybrid helps the wiki's *entity* queries (MRR 0.316 → 0.442, n=33) and slightly hurts term, navigate,
temporal and type queries. The corpus is small and generated by Locus's own code, so this is a smoke
test, not evidence of a general gain.

**The 1.0.0 README claimed R@5 0.947 / MRR 0.633 for the six-signal pipeline here.** On `62156b5` it
reproduces at **0.867 / 0.594** (5 of 5 runs here; an earlier assessment saw it vary between 0.867
and 0.920). Those claims have been removed.

### simple-100 (`--simple --docs 100`)

| System | R@5 | MRR |
|---|---|---|
| Locus default / hybrid (all ablation configs identical) | 0.600 | 0.274 |
| rank_bm25 | 0.590 | 0.271 |
| grep_literal | 0.530 | 0.245 |

### Determinism, 5 fresh processes with different `PYTHONHASHSEED`

| | before (`62156b5`) | after |
|---|---|---|
| wiki, all configs | identical in these 5 runs (0.867 / 0.594) | identical (every config) |
| simple-100, six-signal / hybrid | **3 different R@5 values** (0.513, 0.517, 0.580) | identical (0.600 / 0.274) |

Sources of the old nondeterminism found in the code, all fixed: query doc-type detection iterated a
`frozenset` (hash-seed dependent), recency ranked documents by index wall-clock time (differs per run),
and several SQL queries and sorts had no tie-break. Retrieval also wrote to the bulletin and budget on every call; it is now
read-only unless `track=True`.

---

### Comparison with vector RAG

Locus is not trying to compete with embedding-based retrieval on semantic
similarity tasks. The tradeoff is explicit:

| Dimension | Locus | Vector RAG |
|---|---|---|
| Setup | `pip install locus-rag` + index | Embedding model + GPU/API |
| Dependencies | Zero (pure stdlib) | transformers / openai / etc. |
| Retrieval type | Keyword + graph + structure | Semantic similarity |
| Transparency | Per-chunk provenance tags | Black box scores |
| Temporal queries | Native (valid_from/valid_to) | Requires metadata filter |
| Explainability | `locus explain <chunk_id>` | Not available |
| Cost at scale | SQLite on local disk | API calls or GPU time |

The right tool depends on your corpus and constraints. Locus excels when
you want **zero infrastructure friction**, **full explainability**, and
**structured knowledge** (entities, dates, relationships) alongside
free-text retrieval.

---

## Running the original suite

```bash
python benchmarks/run.py                 # wiki ablation + baselines + latency (10–500 docs)
python benchmarks/run.py --docs 50 --fast
python benchmarks/run.py --latency-only
python -m locus.bench.beir data/beir/scifact --split test --out benchmarks/results/beir-scifact.json
```

Results land in `benchmarks/results/` (git-ignored). Published summaries of the runs above are in
[`published/2026-10-20/`](published/2026-10-20/).
