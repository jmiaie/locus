"""
Baseline comparison — Locus full pipeline vs. rank_bm25 vs. grep (literal match).

rank_bm25 is an optional dependency (pip install rank_bm25).
If not installed, that comparison is skipped and the report notes it.

All three systems index the same corpus and answer the same QA pairs.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .synthetic import QAPair

# rank_bm25 is optional
try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False


@dataclass
class BaselineResult:
    system: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    avg_query_ms: float
    num_queries: int
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "recall@1": round(self.recall_at_1, 4),
            "recall@3": round(self.recall_at_3, 4),
            "recall@5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
            "avg_query_ms": round(self.avg_query_ms, 2),
            "num_queries": self.num_queries,
            "notes": self.notes,
        }


class BaselineBench:
    """Compare Locus against rank_bm25 and grep on the same corpus."""

    def __init__(self, engine, doc_dir: Path) -> None:
        self._engine = engine
        self._doc_dir = Path(doc_dir)

    def run(self, qa_pairs: "list[QAPair]", limit: int = 5) -> list[BaselineResult]:
        results: list[BaselineResult] = []

        # --- Locus full pipeline ---
        results.append(self._run_locus(qa_pairs, limit))

        # --- rank_bm25 ---
        if RANK_BM25_AVAILABLE:
            results.append(self._run_rank_bm25(qa_pairs, limit))
        else:
            results.append(BaselineResult(
                system="rank_bm25",
                recall_at_1=0, recall_at_3=0, recall_at_5=0, mrr=0,
                avg_query_ms=0, num_queries=0,
                notes="skipped — pip install rank_bm25",
            ))

        # --- grep (literal substring) ---
        results.append(self._run_grep(qa_pairs, limit))

        return results

    # ------------------------------------------------------------------
    # Locus
    # ------------------------------------------------------------------

    def _run_locus(self, qa_pairs: "list[QAPair]", limit: int) -> BaselineResult:
        return self._score(
            system="locus_6signal",
            qa_pairs=qa_pairs,
            retrieve_fn=lambda q: [
                c.doc_path for c in
                self._engine.retrieve(q, limit=limit, use_cache=False)
            ],
            notes="BM25 + KG + LinkWalk + Structural + Recency + LinkPop + Reranker",
        )

    # ------------------------------------------------------------------
    # rank_bm25
    # ------------------------------------------------------------------

    def _run_rank_bm25(self, qa_pairs: "list[QAPair]", limit: int) -> BaselineResult:
        docs = list(self._doc_dir.glob("**/*.md"))
        tokenized = [self._tokenize(d.read_text(encoding="utf-8")) for d in docs]
        bm25 = _BM25Okapi(tokenized)

        def retrieve(query: str) -> list[str]:
            tokens = self._tokenize(query)
            scores = bm25.get_scores(tokens)
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
            return [docs[i].name for i in ranked[:limit]]

        return self._score(
            system="rank_bm25",
            qa_pairs=qa_pairs,
            retrieve_fn=retrieve,
            notes="BM25Okapi, same tokenization as Locus corpus",
        )

    # ------------------------------------------------------------------
    # Grep (literal match)
    # ------------------------------------------------------------------

    def _run_grep(self, qa_pairs: "list[QAPair]", limit: int) -> BaselineResult:
        docs = list(self._doc_dir.glob("**/*.md"))
        contents = {d.name: d.read_text(encoding="utf-8").lower() for d in docs}

        def retrieve(query: str) -> list[str]:
            terms = query.lower().split()
            scored: list[tuple[str, int]] = []
            for name, text in contents.items():
                hits = sum(text.count(t) for t in terms)
                if hits > 0:
                    scored.append((name, hits))
            scored.sort(key=lambda x: -x[1])
            return [name for name, _ in scored[:limit]]

        return self._score(
            system="grep_literal",
            qa_pairs=qa_pairs,
            retrieve_fn=retrieve,
            notes="Literal term frequency, no IDF, no signals",
        )

    # ------------------------------------------------------------------
    # Shared scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        system: str,
        qa_pairs: "list[QAPair]",
        retrieve_fn,
        notes: str = "",
    ) -> BaselineResult:
        k_vals = [1, 3, 5]
        hits_at: dict[int, int] = {k: 0 for k in k_vals}
        rr_sum = 0.0
        times: list[float] = []

        for qa in qa_pairs:
            t0 = time.perf_counter()
            retrieved = retrieve_fn(qa.query)
            times.append((time.perf_counter() - t0) * 1000)

            expected = qa.expected_doc
            for k in k_vals:
                if expected in retrieved[:k]:
                    hits_at[k] += 1
            for rank, doc in enumerate(retrieved, 1):
                if doc == expected:
                    rr_sum += 1.0 / rank
                    break

        n = len(qa_pairs)
        return BaselineResult(
            system=system,
            recall_at_1=hits_at[1] / n if n else 0.0,
            recall_at_3=hits_at[3] / n if n else 0.0,
            recall_at_5=hits_at[5] / n if n else 0.0,
            mrr=rr_sum / n if n else 0.0,
            avg_query_ms=sum(times) / len(times) if times else 0.0,
            num_queries=n,
            notes=notes,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        stopwords = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to",
            "for", "of", "with", "is", "are", "was", "be", "it", "as",
        }
        return [w for w in text.split() if w not in stopwords and len(w) > 1]
