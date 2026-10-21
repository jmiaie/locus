"""
BM25 retriever — first-stage lexical retrieval.

Scoring is delegated to the corpus index backend:

* ``fts5``   — SQLite FTS5 ``bm25()`` (C implementation, inverted index)
* ``legacy`` — pure-Python BM25 over the ``term_index`` table

Both use Okapi BM25 with k1=1.2, b=0.75 and FTS5's IDF
(``log((N - n + 0.5) / (n + 0.5))``, floored at 1e-6), computed per chunk.
Results are ordered by score descending, then index order (chunk rowid), so
the output is fully deterministic.  Searching never writes to the store.
"""

import logging
from dataclasses import dataclass, field

from ..memory.corpus import BM25_B, BM25_K1, Corpus, tokenize  # noqa: F401  (re-export)

logger = logging.getLogger(__name__)

# Kept for backwards compatibility with code that imported these names.
K1 = BM25_K1
B = BM25_B
DELTA = 0.0


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
        terms = self.corpus.analyze(query)
        if not terms or limit <= 0:
            return []
        ranked = self.corpus.search_bm25(terms, limit=limit)
        if not ranked:
            return []
        chunks_map = self.corpus.get_chunks_batch([cid for cid, _ in ranked])
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
