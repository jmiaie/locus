"""
BM25+ retriever — Robertson BM25 with delta floor.
Pure Python, zero external dependencies.
"""

import math
import logging
from dataclasses import dataclass, field

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

    def search(self, query: str, limit: int = 10) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms:
            return []

        N = self.corpus.doc_count()
        avgdl = self.corpus.avg_doc_length()
        if N == 0 or avgdl == 0:
            return []

        scores: dict[str, float] = {}

        for term in terms:
            posting = self.corpus.get_posting_list(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for chunk_id, tf in posting.items():
                chunk = self.corpus.get_chunk(chunk_id)
                if not chunk:
                    continue
                dl = len(chunk.content.split())
                norm_tf = (
                    tf * (K1 + 1)
                ) / (
                    tf + K1 * (1 - B + B * dl / avgdl)
                ) + DELTA
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * norm_tf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        results: list[ScoredChunk] = []
        for chunk_id, score in ranked:
            chunk = self.corpus.get_chunk(chunk_id)
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
