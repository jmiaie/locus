"""
Recency retriever — soft freshness prior over the corpus.

Score = exp(-days_since_indexed * ln2 / half_life_days)
Default half-life: 30 days (a doc indexed 30 days ago scores 0.5).

Used as a low-weight (0.3) fourth signal in RRF so freshness gently
nudges results toward recent content without dominating relevance signals.
"""

import logging
import math
import sqlite3
from datetime import datetime, timezone

from ..memory.corpus import Corpus
from .bm25 import ScoredChunk

logger = logging.getLogger(__name__)

DEFAULT_HALF_LIFE = 30.0


class RecencyRetriever:
    def __init__(self, corpus: Corpus, half_life_days: float = DEFAULT_HALF_LIFE):
        self.corpus = corpus
        self._decay = math.log(2) / half_life_days

    def search(self, limit: int = 20) -> list[ScoredChunk]:
        now = datetime.now(timezone.utc)
        scored: list[ScoredChunk] = []

        for doc_path, indexed_at in self._doc_dates():
            score = self._score(indexed_at, now)
            chunks = self.corpus.get_chunks_for_doc(doc_path)
            if not chunks:
                continue
            c = chunks[0]
            scored.append(ScoredChunk(
                chunk_id=c.id,
                doc_path=c.doc_path,
                score=score,
                content=c.content,
                provenance="recency",
                entities=c.metadata.get("entities", []),
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    def _score(self, indexed_at: str | None, now: datetime) -> float:
        if not indexed_at:
            return 0.5
        try:
            dt = datetime.fromisoformat(indexed_at.replace(" ", "T"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days_ago = max(0.0, (now - dt).total_seconds() / 86400.0)
            return math.exp(-self._decay * days_ago)
        except Exception:
            return 0.5

    def _doc_dates(self) -> list[tuple[str, str | None]]:
        with sqlite3.connect(self.corpus.db_path) as conn:
            rows = conn.execute(
                "SELECT doc_path, indexed_at FROM doc_stats"
            ).fetchall()
        return [(r[0], r[1]) for r in rows]
