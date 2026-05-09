"""
Latency benchmark — measures index time and query latency at multiple corpus scales.

Scale points: 10, 50, 100, 500, 1000 documents (configurable).
Metrics per scale:
  - index_total_ms       total time to index all documents
  - index_per_doc_ms     average per document
  - query_p50_ms         median query latency
  - query_p95_ms         95th-percentile query latency
  - query_p99_ms         99th-percentile query latency
  - store_bytes          on-disk store size after indexing
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .synthetic import QAPair


@dataclass
class LatencyResult:
    corpus_size: int
    index_total_ms: float
    index_per_doc_ms: float
    query_p50_ms: float
    query_p95_ms: float
    query_p99_ms: float
    query_count: int
    store_bytes: int

    @property
    def store_kb(self) -> float:
        return round(self.store_bytes / 1024, 1)

    def to_dict(self) -> dict:
        return {
            "corpus_size": self.corpus_size,
            "index_total_ms": round(self.index_total_ms, 1),
            "index_per_doc_ms": round(self.index_per_doc_ms, 2),
            "query_p50_ms": round(self.query_p50_ms, 2),
            "query_p95_ms": round(self.query_p95_ms, 2),
            "query_p99_ms": round(self.query_p99_ms, 2),
            "query_count": self.query_count,
            "store_kb": self.store_kb,
        }


class LatencyBench:
    """Profile index and query latency at different corpus scales."""

    DEFAULT_SCALES = [10, 50, 100, 500]
    NUM_QUERIES = 30  # queries per scale point

    def __init__(self, scales: list[int] | None = None, num_queries: int | None = None) -> None:
        self.scales = scales or self.DEFAULT_SCALES
        self.num_queries = num_queries or self.NUM_QUERIES

    def run(self) -> list[LatencyResult]:
        from ..core import LocusEngine
        from .synthetic import SyntheticCorpus

        results: list[LatencyResult] = []
        for size in self.scales:
            result = self._run_scale(size, LocusEngine, SyntheticCorpus)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_scale(self, size: int, EngineClass, CorpusClass) -> LatencyResult:
        corpus = CorpusClass(num_docs=size, seed=99)
        store_tmpdir = tempfile.mkdtemp(prefix=f"locus_lat_{size}_")

        try:
            doc_dir = corpus.generate()
            engine = EngineClass(store_path=store_tmpdir)

            # --- Index timing ---
            t0 = time.perf_counter()
            engine.index(str(doc_dir))
            index_ms = (time.perf_counter() - t0) * 1000

            # --- Query timing ---
            queries = self._sample_queries(corpus.qa_pairs, self.num_queries)
            query_times: list[float] = []
            for q in queries:
                t0 = time.perf_counter()
                engine.retrieve(q, limit=5, use_cache=False)
                query_times.append((time.perf_counter() - t0) * 1000)

            query_times.sort()
            store_bytes = self._dir_size(store_tmpdir)

        finally:
            corpus.cleanup()
            shutil.rmtree(store_tmpdir, ignore_errors=True)

        n = len(query_times)
        p50 = query_times[int(n * 0.50)] if n else 0.0
        p95 = query_times[int(n * 0.95)] if n else 0.0
        p99 = query_times[min(int(n * 0.99), n - 1)] if n else 0.0

        return LatencyResult(
            corpus_size=size,
            index_total_ms=index_ms,
            index_per_doc_ms=index_ms / size if size else 0.0,
            query_p50_ms=p50,
            query_p95_ms=p95,
            query_p99_ms=p99,
            query_count=n,
            store_bytes=store_bytes,
        )

    @staticmethod
    def _sample_queries(qa_pairs: list["QAPair"], n: int) -> list[str]:
        """Return up to n query strings, cycling if qa_pairs is smaller."""
        if not qa_pairs:
            return ["authentication", "deployment", "monitoring"] * (n // 3 + 1)
        queries = [qa.query for qa in qa_pairs]
        result: list[str] = []
        while len(result) < n:
            result.extend(queries)
        return result[:n]

    @staticmethod
    def _dir_size(path: str) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total
