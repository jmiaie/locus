"""
Link-popularity retriever — ranks documents by how many other indexed
documents link to them via [[wikilinks]] or markdown [text](url) links.

A document cited by many others is likely a hub / important reference.

Score = inbound_link_count / max_inbound_count  (normalised 0-1).

Popularity is computed lazily on first search and cached until the
corpus size changes (same invalidation pattern as BM25 IDF cache).
Weight in RRF: 0.2 (soft signal, never dominates relevance).
"""

import logging
from ..memory.corpus import Corpus
from .bm25 import ScoredChunk

logger = logging.getLogger(__name__)


class LinkPopularityRetriever:
    """
    Returns corpus documents sorted by inbound link count.
    Documents with zero inbound links are excluded.
    """

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self._cache: dict[str, int] | None = None
        self._cache_doc_count: int = -1

    def search(self, limit: int = 20) -> list[ScoredChunk]:
        popularity = self._popularity()
        if not popularity:
            return []

        max_count = max(popularity.values()) or 1
        scored: list[ScoredChunk] = []

        for doc_path, count in sorted(popularity.items(), key=lambda x: x[1], reverse=True):
            if count == 0:
                continue
            chunks = self.corpus.get_chunks_for_doc(doc_path)
            if not chunks:
                continue
            c = chunks[0]
            scored.append(
                ScoredChunk(
                    chunk_id=c.id,
                    doc_path=c.doc_path,
                    score=count / max_count,
                    content=c.content,
                    provenance="link_pop",
                    entities=c.metadata.get("entities", []),
                )
            )

        return scored[:limit]

    def popularity_map(self) -> dict[str, int]:
        """Return {doc_path: inbound_link_count} for all docs."""
        return dict(self._popularity())

    def _popularity(self) -> dict[str, int]:
        N = self.corpus.doc_count()
        if N != self._cache_doc_count:
            self._cache = None
            self._cache_doc_count = N

        if self._cache is not None:
            return self._cache

        from .link_walker import LinkWalker
        walker = LinkWalker(self.corpus)
        popularity: dict[str, int] = {}

        for doc_path in self.corpus.list_docs():
            for chunk in self.corpus.get_chunks_for_doc(doc_path):
                for link in chunk.metadata.get("links", []):
                    for target in walker._resolve_link(link):
                        popularity[target] = popularity.get(target, 0) + 1

        self._cache = popularity
        logger.debug("Link popularity computed: %d docs with inbound links", len(popularity))
        return popularity
