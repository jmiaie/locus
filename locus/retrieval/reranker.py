"""
LocusReranker — lightweight post-RRF heuristic re-ranking.

Applied *after* Reciprocal Rank Fusion; it reshuffles the already-fused
result set rather than expanding it.  Three independent boosts are
combined multiplicatively so a chunk with zero RRF score stays at zero:

    new_score = rrf_score * (1 + title_boost + entity_boost + freshness_boost)

Boosts (all normalised 0–1 before weighting):

  title     — query term overlap with the chunk's section heading
  entity    — fraction of chunk entities that have KG facts
  freshness — recency of the document's frontmatter date field

Default weights are conservative: the re-ranker nudges, it does not
override.  Pass custom RerankerWeights to tighten or loosen.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..memory.corpus import Corpus, tokenize
from .bm25 import ScoredChunk

if TYPE_CHECKING:
    from ..memory.knowledge_graph import TemporalKG

logger = logging.getLogger(__name__)

_FRESHNESS_HALF_LIFE = 90.0   # days — longer than RecencyRetriever's 30


@dataclass
class RerankerWeights:
    title: float     = 0.30   # heading term overlap
    entity: float    = 0.20   # entity KG coverage
    freshness: float = 0.10   # document date recency


class LocusReranker:
    """
    Post-RRF re-ranker using three heuristic boosts.

    Usage:
        reranker = LocusReranker(corpus, kg)
        reranked = reranker.rerank(chunks, query="JWT auth")
    """

    def __init__(
        self,
        corpus: Corpus,
        kg: TemporalKG | None = None,
        weights: RerankerWeights | None = None,
    ):
        self.corpus = corpus
        self.kg = kg
        self.weights = weights or RerankerWeights()
        self._decay = math.log(2) / _FRESHNESS_HALF_LIFE

    def rerank(
        self,
        chunks: list[ScoredChunk],
        query: str,
        weights: RerankerWeights | None = None,
    ) -> list[ScoredChunk]:
        """
        Re-rank chunks by applying heuristic boosts to RRF scores.
        Returns a new list sorted by boosted score; original list untouched.
        """
        w = weights or self.weights
        query_terms = set(tokenize(query))
        now = datetime.now(timezone.utc)

        # Pre-fetch all chunks in one batch (avoids N individual queries)
        chunk_ids = [c.chunk_id for c in chunks]
        raw_map = self.corpus.get_chunks_batch(chunk_ids)

        # Pre-fetch KG entity coverage for all entities across chunks
        all_entities: set[str] = set()
        for c in chunks:
            all_entities.update(c.entities or [])
        kg_covered = self._covered_entities(all_entities) if self.kg else set()

        boosted: list[tuple[ScoredChunk, float]] = []
        for chunk in chunks:
            raw = raw_map.get(chunk.chunk_id)
            boost = 0.0

            # --- Title boost ---
            if query_terms and raw:
                section = raw.metadata.get("section", "")
                if section:
                    section_terms = set(tokenize(section))
                    overlap = len(query_terms & section_terms) / max(len(query_terms), 1)
                    boost += overlap * w.title

            # --- Entity density boost ---
            entities = chunk.entities or []
            if entities and kg_covered:
                coverage = sum(1 for e in entities if e in kg_covered) / len(entities)
                boost += coverage * w.entity

            # --- Document freshness boost ---
            if raw and w.freshness > 0:
                doc_date = raw.metadata.get("frontmatter", {}).get("date", "")
                if doc_date:
                    fresh = self._freshness(doc_date, now)
                    boost += fresh * w.freshness

            boosted.append((chunk, chunk.score * (1.0 + boost)))

        boosted.sort(key=lambda x: x[1], reverse=True)

        return [
            ScoredChunk(
                chunk_id=c.chunk_id,
                doc_path=c.doc_path,
                score=round(new_score, 6),
                content=c.content,
                provenance=c.provenance + ("+r" if new_score != c.score else ""),
                entities=c.entities,
            )
            for c, new_score in boosted
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _covered_entities(self, entities: set[str]) -> set[str]:
        """Return entities that have at least one KG fact."""
        covered: set[str] = set()
        for entity in entities:
            if self.kg.query_entity(entity):
                covered.add(entity)
        return covered

    def _freshness(self, date_str: str, now: datetime) -> float:
        try:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days_ago = max(0.0, (now - dt).total_seconds() / 86400.0)
            return math.exp(-self._decay * days_ago)
        except Exception:
            return 0.0
