"""
Signal ablation benchmark — measures the retrieval quality contribution
of each signal added to the pipeline.

Configurations tested (cumulative).  Stage 1 is always BM25; the other
signals re-rank the BM25 candidate pool (``LocusEngine.retrieve(mode="hybrid")``):
  1. bm25_only        — BM25 alone (the default retrieval mode)
  2. bm25_kg          — + KG entity expansion
  3. bm25_kg_links    — + link walking
  4. bm25_kg_links_sf — + structural + recency
  5. hybrid_all       — + link popularity (all re-rank signals)
  6. hybrid_reranked  — hybrid_all + heuristic re-ranker
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import LocusEngine
    from .synthetic import QAPair



_CONFIGURATIONS: list[tuple[str, list[str]]] = [
    ("bm25_only",        ["bm25"]),
    ("bm25_kg",          ["bm25", "kg"]),
    ("bm25_kg_links",    ["bm25", "kg", "links"]),
    ("bm25_kg_links_sf", ["bm25", "kg", "links", "structural", "recency"]),
    ("hybrid_all",       ["bm25", "kg", "links", "structural", "recency", "link_pop"]),
    ("hybrid_reranked",  ["bm25", "kg", "links", "structural", "recency", "link_pop", "rerank"]),
]


@dataclass
class AblationResult:
    config: str
    signals: list[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    avg_query_ms: float
    num_queries: int
    by_type: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "signals": self.signals,
            "recall@1": round(self.recall_at_1, 4),
            "recall@3": round(self.recall_at_3, 4),
            "recall@5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
            "avg_query_ms": round(self.avg_query_ms, 2),
            "num_queries": self.num_queries,
            "by_type": self.by_type,
        }


class AblationBench:
    """Run all signal configurations on the same engine + QA pairs."""

    def __init__(self, engine: "LocusEngine", k_values: list[int] | None = None) -> None:
        self._engine = engine
        self._k_values = k_values or [1, 3, 5]

    def run(self, qa_pairs: "list[QAPair]", limit: int = 5) -> list[AblationResult]:
        results = []
        for config_name, signals in _CONFIGURATIONS:
            result = self._run_config(config_name, signals, qa_pairs, limit)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_config(
        self,
        config_name: str,
        signals: list[str],
        qa_pairs: "list[QAPair]",
        limit: int,
    ) -> AblationResult:
        engine = self._engine
        do_rerank = "rerank" in signals
        active = [s for s in signals if s != "rerank"]

        hits_at: dict[int, int] = {k: 0 for k in self._k_values}
        rr_sum = 0.0
        query_times: list[float] = []

        # Per query-type tracking (for WikiQAPair)
        type_hits: dict[str, dict[int, int]] = {}
        type_rr: dict[str, float] = {}
        type_n: dict[str, int] = {}

        for qa in qa_pairs:
            qtype = getattr(qa, "query_type", "term")
            if qtype not in type_hits:
                type_hits[qtype] = {k: 0 for k in self._k_values}
                type_rr[qtype] = 0.0
                type_n[qtype] = 0
            type_n[qtype] += 1

            t0 = time.perf_counter()
            chunks = self._retrieve(engine, qa.query, active, limit)
            if do_rerank and chunks:
                chunks = engine.rerank(chunks, qa.query)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            query_times.append(elapsed_ms)

            retrieved_docs = [c.doc_path for c in chunks]
            expected = qa.expected_doc

            for k in self._k_values:
                if expected in retrieved_docs[:k]:
                    hits_at[k] += 1
                    type_hits[qtype][k] += 1

            for rank, doc in enumerate(retrieved_docs, 1):
                if doc == expected:
                    rr_sum += 1.0 / rank
                    type_rr[qtype] += 1.0 / rank
                    break

        n = len(qa_pairs)
        avg_ms = sum(query_times) / len(query_times) if query_times else 0.0

        by_type = {
            qt: {
                "recall@5": round(type_hits[qt].get(5, 0) / type_n[qt], 3) if type_n[qt] else 0,
                "mrr":      round(type_rr[qt] / type_n[qt], 3) if type_n[qt] else 0,
                "n":        type_n[qt],
            }
            for qt in type_hits
        }

        return AblationResult(
            config=config_name,
            signals=active + (["rerank"] if do_rerank else []),
            recall_at_1=hits_at.get(1, 0) / n if n else 0.0,
            recall_at_3=hits_at.get(3, 0) / n if n else 0.0,
            recall_at_5=hits_at.get(5, 0) / n if n else 0.0,
            mrr=rr_sum / n if n else 0.0,
            avg_query_ms=avg_ms,
            num_queries=n,
            by_type=by_type,
        )

    @staticmethod
    def _retrieve(engine: "LocusEngine", query: str, signals: list[str], limit: int):
        rerank_signals = [s for s in signals if s != "bm25"]
        if not rerank_signals:
            return engine.retrieve(query, limit=limit, mode="bm25", use_cache=False)
        return engine.retrieve(
            query, limit=limit, mode="hybrid", signals=rerank_signals, use_cache=False,
        )
