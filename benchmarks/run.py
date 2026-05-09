"""
Locus benchmark runner.

Usage:
    python benchmarks/run.py                        # full run, 100 docs
    python benchmarks/run.py --docs 50 --fast       # quick run
    python benchmarks/run.py --docs 200 --latency-only
    python benchmarks/run.py --no-latency           # skip scale tests

Results saved to benchmarks/results/latest.json and printed to stdout.
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from locus.core import LocusEngine, __version__
from locus.bench.synthetic import SyntheticCorpus
from locus.bench.ablation import AblationBench
from locus.bench.latency import LatencyBench
from locus.bench.baseline import BaselineBench, RANK_BM25_AVAILABLE
from locus.bench.report import BenchReport


def main() -> None:
    parser = argparse.ArgumentParser(description="Locus benchmark suite")
    parser.add_argument("--docs",         type=int, default=100,  help="Corpus size for ablation + baseline")
    parser.add_argument("--seed",         type=int, default=42,   help="RNG seed for reproducibility")
    parser.add_argument("--fast",         action="store_true",     help="Smaller corpus + latency scales")
    parser.add_argument("--no-latency",   action="store_true",     help="Skip latency benchmarks")
    parser.add_argument("--latency-only", action="store_true",     help="Run only latency benchmarks")
    parser.add_argument("--out",          default=None,            help="Output JSON file path")
    args = parser.parse_args()

    if args.fast:
        args.docs = max(args.docs, 30)
        latency_scales = [10, 50, 100]
    else:
        latency_scales = [10, 50, 100, 500]

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"\nLocus v{__version__} — Benchmark Suite")
    print(f"{'=' * 50}")
    print(f"Corpus size   : {args.docs} docs")
    print(f"Seed          : {args.seed}")
    print(f"rank_bm25     : {'available' if RANK_BM25_AVAILABLE else 'not installed (pip install rank_bm25)'}")
    print()

    ablation_results = []
    baseline_results = []
    latency_results  = []
    num_queries      = 0

    # ----------------------------------------------------------------
    # Ablation + Baseline
    # ----------------------------------------------------------------
    if not args.latency_only:
        print(f"Generating {args.docs}-doc synthetic corpus...")
        corpus = SyntheticCorpus(num_docs=args.docs, seed=args.seed)
        store_tmp = tempfile.mkdtemp(prefix="locus_bench_main_")

        try:
            doc_dir = corpus.generate()
            engine  = LocusEngine(store_path=store_tmp)

            print("Indexing corpus...")
            engine.index(str(doc_dir))

            qa_pairs = corpus.qa_pairs
            num_queries = len(qa_pairs)
            print(f"QA pairs generated : {num_queries}")
            print()

            # Ablation
            print("Running signal ablation...")
            ablation  = AblationBench(engine)
            ablation_results = ablation.run(qa_pairs, limit=5)
            for r in ablation_results:
                print(f"  {r.config:<22} R@5={r.recall_at_5:.3f}  MRR={r.mrr:.3f}  {r.avg_query_ms:.1f}ms/q")

            print()

            # Baseline comparison
            print("Running baseline comparison...")
            baseline  = BaselineBench(engine, doc_dir=doc_dir)
            baseline_results = baseline.run(qa_pairs, limit=5)
            for r in baseline_results:
                if r.num_queries > 0:
                    print(f"  {r.system:<22} R@5={r.recall_at_5:.3f}  MRR={r.mrr:.3f}  {r.avg_query_ms:.1f}ms/q")
                else:
                    print(f"  {r.system:<22} {r.notes}")

        finally:
            corpus.cleanup()
            import shutil
            shutil.rmtree(store_tmp, ignore_errors=True)

    print()

    # ----------------------------------------------------------------
    # Latency
    # ----------------------------------------------------------------
    if not args.no_latency:
        print(f"Running latency benchmark at scales: {latency_scales}")
        lat = LatencyBench(scales=latency_scales, num_queries=30)
        latency_results = lat.run()
        for r in latency_results:
            print(
                f"  {r.corpus_size:>5} docs — "
                f"idx {r.index_total_ms:.0f}ms ({r.index_per_doc_ms:.1f}ms/doc)  "
                f"q p50={r.query_p50_ms:.1f}ms p95={r.query_p95_ms:.1f}ms  "
                f"store {r.store_kb:.0f}KB"
            )

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    report = BenchReport(
        ablation=ablation_results,
        latency=latency_results,
        baseline=baseline_results,
        corpus_size=args.docs,
        num_queries=num_queries,
    )

    print()
    print(report.summary())

    # Save results
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else results_dir / f"bench_{ts}.json"
    report.save(out_path)

    # Also overwrite latest
    latest_path = results_dir / "latest.json"
    report.save(latest_path)

    print(f"\nResults saved → {out_path}")
    print(f"Latest link   → {latest_path}")


if __name__ == "__main__":
    main()
