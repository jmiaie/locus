"""
Locus evaluation framework — recall@K and MRR benchmarks.

Measures retrieval quality against a set of (query, expected_docs) pairs.
Works with any LocusEngine configuration.

QA file format (JSON):
    [
      {"query": "how does JWT auth work?", "expected_docs": ["auth.md"]},
      {"query": "incident response process", "expected_docs": ["ir.md", "runbook.md"]},
      ...
    ]

Usage:
    from locus import LocusEngine
    from locus.eval import LocusEval

    engine = LocusEngine(".locus")
    ev = LocusEval(engine)
    report = ev.score_from_file("benchmark.json")
    print(report.summary())

CLI:
    locus benchmark benchmark.json --k 1,3,5
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import LocusEngine

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    query: str
    expected_docs: list[str]
    retrieved_docs: list[str]      # doc_paths in rank order
    hit_rank: int | None           # rank (1-indexed) of first expected doc hit, or None

    @property
    def hit(self) -> bool:
        return self.hit_rank is not None

    def recall_at(self, k: int) -> bool:
        return self.hit_rank is not None and self.hit_rank <= k

    def reciprocal_rank(self) -> float:
        return 1.0 / self.hit_rank if self.hit_rank else 0.0


@dataclass
class EvalReport:
    query_results: list[QueryResult] = field(default_factory=list)
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5])

    # ----------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------

    def recall_at(self, k: int) -> float:
        if not self.query_results:
            return 0.0
        hits = sum(1 for r in self.query_results if r.recall_at(k))
        return hits / len(self.query_results)

    def mrr(self) -> float:
        if not self.query_results:
            return 0.0
        return sum(r.reciprocal_rank() for r in self.query_results) / len(self.query_results)

    def misses(self, k: int = 5) -> list[QueryResult]:
        return [r for r in self.query_results if not r.recall_at(k)]

    # ----------------------------------------------------------------
    # Formatting
    # ----------------------------------------------------------------

    def summary(self, max_misses: int = 5) -> str:
        lines = [
            "Locus Benchmark",
            "=" * 40,
            f"Queries:  {len(self.query_results)}",
        ]
        for k in self.k_values:
            r = self.recall_at(k)
            hits = int(r * len(self.query_results))
            lines.append(f"Recall@{k}: {r:.3f}  ({hits}/{len(self.query_results)})")
        lines.append(f"MRR:      {self.mrr():.3f}")

        missed = self.misses(max(self.k_values))
        if missed:
            lines += ["", f"Misses (top {min(max_misses, len(missed))})"]
            for qr in missed[:max_misses]:
                expected = qr.expected_docs[:2]
                retrieved = qr.retrieved_docs[:3]
                lines.append(
                    f"  [MISS] \"{qr.query[:60]}\"  "
                    f"expected={expected}  got={retrieved}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "query_count": len(self.query_results),
            "metrics": {
                **{f"recall@{k}": round(self.recall_at(k), 4) for k in self.k_values},
                "mrr": round(self.mrr(), 4),
            },
            "misses": [
                {
                    "query": r.query,
                    "expected": r.expected_docs,
                    "retrieved": r.retrieved_docs[:5],
                }
                for r in self.misses(max(self.k_values))
            ],
        }


class LocusEval:
    def __init__(self, engine: LocusEngine, k_values: list[int] = None):
        self.engine = engine
        self.k_values = k_values or [1, 3, 5]

    def score(self, qa_pairs: list[dict]) -> EvalReport:
        """
        Run retrieval for each query and measure recall.

        Each QA pair must have:
          query:         str — the retrieval query
          expected_docs: list[str] — doc paths that should appear in results
                         (or "expected_doc": str for single-doc format)
        """
        limit = max(self.k_values)
        results: list[QueryResult] = []

        for i, qa in enumerate(qa_pairs):
            query = qa.get("query", "")
            if not query:
                continue

            expected = qa.get("expected_docs") or (
                [qa["expected_doc"]] if "expected_doc" in qa else []
            )
            expected = [str(e) for e in expected]

            try:
                chunks = self.engine.retrieve(query, limit=limit)
            except Exception as e:
                logger.warning("Retrieval failed for query %d: %s", i, e)
                chunks = []

            retrieved = [c.doc_path for c in chunks]

            # Find rank of first expected doc hit
            hit_rank = None
            for rank, doc in enumerate(retrieved, start=1):
                if any(exp in doc or doc in exp for exp in expected):
                    hit_rank = rank
                    break

            results.append(QueryResult(
                query=query,
                expected_docs=expected,
                retrieved_docs=retrieved,
                hit_rank=hit_rank,
            ))
            if (i + 1) % 10 == 0:
                logger.info("Evaluated %d/%d queries", i + 1, len(qa_pairs))

        return EvalReport(query_results=results, k_values=self.k_values)

    def score_from_file(self, path: str | Path) -> EvalReport:
        """Load QA pairs from a JSON file and run evaluation."""
        path = Path(path)
        qa_pairs = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(qa_pairs, list):
            raise ValueError(f"QA file must be a JSON array: {path}")
        logger.info("Loaded %d QA pairs from %s", len(qa_pairs), path)
        return self.score(qa_pairs)
