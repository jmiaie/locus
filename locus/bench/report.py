"""
BenchReport — format benchmark results as ASCII tables and JSON.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ablation import AblationResult
    from .baseline import BaselineResult
    from .latency import LatencyResult


class BenchReport:
    def __init__(
        self,
        ablation: "list[AblationResult] | None" = None,
        latency: "list[LatencyResult] | None" = None,
        baseline: "list[BaselineResult] | None" = None,
        corpus_size: int = 0,
        num_queries: int = 0,
    ) -> None:
        self.ablation = ablation or []
        self.latency = latency or []
        self.baseline = baseline or []
        self.corpus_size = corpus_size
        self.num_queries = num_queries
        self.timestamp = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    def summary(self) -> str:
        sections: list[str] = [
            "=" * 70,
            f"  LOCUS BENCHMARK REPORT — {self.timestamp[:19]}",
            f"  Corpus: {self.corpus_size} docs   Queries: {self.num_queries}",
            "=" * 70,
        ]

        if self.ablation:
            sections.append("\n-- SIGNAL ABLATION --------------------------------------------------")
            sections.append(self._ablation_table())

        if self.baseline:
            sections.append("\n-- BASELINE COMPARISON ----------------------------------------------")
            sections.append(self._baseline_table())

        if self.latency:
            sections.append("\n-- LATENCY PROFILE --------------------------------------------------")
            sections.append(self._latency_table())

        sections.append("\n" + "=" * 70)
        return "\n".join(sections)

    def _ablation_table(self) -> str:
        header = f"{'Config':<22} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'ms/q':>7}"
        sep = "-" * len(header)
        rows = [header, sep]
        for r in self.ablation:
            rows.append(
                f"{r.config:<22} "
                f"{r.recall_at_1:>6.3f} "
                f"{r.recall_at_3:>6.3f} "
                f"{r.recall_at_5:>6.3f} "
                f"{r.mrr:>6.3f} "
                f"{r.avg_query_ms:>7.1f}"
            )
        # Delta row vs BM25-only baseline
        if len(self.ablation) >= 2:
            base = self.ablation[0]
            last = self.ablation[-1]
            dr1  = last.recall_at_1  - base.recall_at_1
            dr5  = last.recall_at_5  - base.recall_at_5
            dmrr = last.mrr          - base.mrr
            rows.append(sep)
            rows.append(f"{'Δ vs bm25_only':<22} {dr1:>+6.3f} {'':>6} {dr5:>+6.3f} {dmrr:>+6.3f}")
        return "\n".join(rows)

    def _baseline_table(self) -> str:
        header = f"{'System':<22} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'ms/q':>7}"
        sep = "-" * len(header)
        rows = [header, sep]
        for r in self.baseline:
            if r.num_queries == 0:
                rows.append(f"{r.system:<22}  {r.notes}")
                continue
            rows.append(
                f"{r.system:<22} "
                f"{r.recall_at_1:>6.3f} "
                f"{r.recall_at_3:>6.3f} "
                f"{r.recall_at_5:>6.3f} "
                f"{r.mrr:>6.3f} "
                f"{r.avg_query_ms:>7.1f}"
            )
        return "\n".join(rows)

    def _latency_table(self) -> str:
        header = f"{'Docs':>6} {'Idx(ms)':>10} {'ms/doc':>8} {'p50(ms)':>9} {'p95(ms)':>9} {'p99(ms)':>9} {'Store(KB)':>10}"
        sep = "-" * len(header)
        rows = [header, sep]
        for r in self.latency:
            rows.append(
                f"{r.corpus_size:>6} "
                f"{r.index_total_ms:>10.1f} "
                f"{r.index_per_doc_ms:>8.2f} "
                f"{r.query_p50_ms:>9.2f} "
                f"{r.query_p95_ms:>9.2f} "
                f"{r.query_p99_ms:>9.2f} "
                f"{r.store_kb:>10.1f}"
            )
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "corpus_size": self.corpus_size,
            "num_queries": self.num_queries,
            "ablation": [r.to_dict() for r in self.ablation],
            "baseline": [r.to_dict() for r in self.baseline],
            "latency": [r.to_dict() for r in self.latency],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
