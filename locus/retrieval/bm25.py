"""
BM25+ retriever — Robertson BM25 with delta floor.
Pure Python, zero external dependencies.

Optimisations over v0.1.0:
- IDF values cached per-term, invalidated when corpus doc_count changes
- Top-k chunks fetched in a single batch query instead of N individual fetches
- doc_count and avg_doc_length read from corpus cache (not re-queried per search)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..memory.corpus import Corpus, tokenize

logger = logging.getLogger(__name__)

K1 = 1.5
B = 0.75
DELTA = 0.5


@dataclass
class ScoredChunk:
    chunk_id: str
    doc_path: str
    score: float
    content: str
    provenance: str
    entities: list[str] = field(default_factory=list)


class BM25Retriever:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self._idf_cache: dict[str, float] = {}
        self._cache_doc_count: int = -1

    def _get_idf(self, term: str, N: int, df: int) -> float:
        """Return cached IDF, invalidating the cache if corpus size changed."""
        if N != self._cache_doc_count:
            self._idf_cache.clear()
            self._cache_doc_count = N
        if term not in self._idf_cache:
            self._idf_cache[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return self._idf_cache[term]

    def search(self, query: str, limit: int = 10) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms:
            return []

        N = self.corpus.doc_count()
        avgdl = self.corpus.avg_doc_length()
        if N == 0 or avgdl == 0:
            return []

        # Pass 1: accumulate partial scores (tf contribution only; dl normalisation in pass 2)
        partial: dict[str, float] = {}   # chunk_id -> sum of idf * tf (unnormalised)
        chunk_tf: dict[str, dict[str, float]] = {}  # chunk_id -> {term: tf}

        for term in terms:
            posting = self.corpus.get_posting_list(term)
            if not posting:
                continue
            idf = self._get_idf(term, N, len(posting))
            for chunk_id, tf in posting.items():
                partial[chunk_id] = partial.get(chunk_id, 0.0) + idf
                chunk_tf.setdefault(chunk_id, {})[term] = tf

        if not partial:
            return []

        # Pass 2: fetch top candidate chunks in one batch, compute full BM25 scores
        candidate_ids = sorted(partial, key=lambda x: partial[x], reverse=True)[: limit * 3]
        chunks_map = self.corpus.get_chunks_batch(candidate_ids)

        scores: dict[str, float] = {}
        for chunk_id in candidate_ids:
            chunk = chunks_map.get(chunk_id)
            if not chunk:
                continue
            dl = len(chunk.content.split())
            score = 0.0
            for term, tf in chunk_tf.get(chunk_id, {}).items():
                posting = self.corpus.get_posting_list(term)
                idf = self._get_idf(term, N, len(posting))
                norm_tf = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avgdl)) + DELTA
                score += idf * norm_tf
            scores[chunk_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        results: list[ScoredChunk] = []
        for chunk_id, score in ranked:
            chunk = chunks_map.get(chunk_id)
            if chunk:
                results.append(
                    ScoredChunk(
                        chunk_id=chunk_id,
                        doc_path=chunk.doc_path,
                        score=score,
                        content=chunk.content,
                        provenance="bm25",
                        entities=chunk.metadata.get("entities", []),
                    )
                )
        return results
