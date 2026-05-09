"""
DocSimilarity — sparse TF-IDF cosine similarity between corpus documents.

No embeddings, no GPU.  Builds TF-IDF vectors from the existing term_index
table and computes cosine distance in-process.  Useful for "find related
documents" without a query string.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory.corpus import Corpus


class DocSimilarity:
    """Cosine-similarity ranking over indexed documents."""

    def __init__(self, corpus: "Corpus") -> None:
        self._corpus = corpus

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def similar_to(self, doc_path: str, limit: int = 5) -> list[dict]:
        """Return the *limit* most similar documents to *doc_path*.

        Each result: ``{"doc_path": str, "similarity": float}``.
        Similarity is cosine of TF-IDF vectors, range [0, 1].
        Returns empty list if *doc_path* is not in the corpus.
        """
        query_vec = self._doc_vector(doc_path)
        if not query_vec:
            return []

        results: list[dict] = []
        for other in self._corpus.list_docs():
            if other == doc_path:
                continue
            other_vec = self._doc_vector(other)
            score = self._cosine(query_vec, other_vec)
            if score > 0.0:
                results.append({"doc_path": other, "similarity": round(score, 4)})

        results.sort(key=lambda x: -x["similarity"])
        return results[:limit]

    def similarity_matrix(self) -> list[dict]:
        """Return pairwise similarities for all indexed documents.

        Each entry: ``{"doc_a": str, "doc_b": str, "similarity": float}``.
        Only pairs with similarity > 0 are included.
        """
        docs = self._corpus.list_docs()
        vectors = {d: self._doc_vector(d) for d in docs}
        pairs: list[dict] = []
        for i, a in enumerate(docs):
            for b in docs[i + 1 :]:
                score = self._cosine(vectors[a], vectors[b])
                if score > 0.0:
                    pairs.append({"doc_a": a, "doc_b": b, "similarity": round(score, 4)})
        pairs.sort(key=lambda x: -x["similarity"])
        return pairs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _doc_vector(self, doc_path: str) -> dict[str, float]:
        """Return a smoothed TF-IDF vector for *doc_path*."""
        terms = self._corpus.inspect_doc_terms(doc_path, limit=500)
        if not terms:
            return {}
        N = max(self._corpus.doc_count(), 1)
        vector: dict[str, float] = {}
        for entry in terms:
            term = entry["term"]
            tf = entry["total_tf"]
            df = len(self._corpus.get_posting_list(term))
            if df == 0:
                continue
            idf = math.log((N + 1) / (df + 1)) + 1.0  # smoothed IDF
            vector[term] = tf * idf
        return vector

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        """Cosine similarity between two sparse float vectors."""
        if not a or not b:
            return 0.0
        dot = sum(a.get(t, 0.0) * v for t, v in b.items())
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)
