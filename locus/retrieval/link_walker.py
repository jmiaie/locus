"""
Link walker — follows [[wikilinks]] and markdown links from seed chunks.
Scores decay by LINK_DECAY per hop depth.
"""

import logging

from ..memory.corpus import Corpus
from .bm25 import ScoredChunk

logger = logging.getLogger(__name__)

LINK_DECAY = 0.4


class LinkWalker:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def walk(
        self,
        seed_chunks: list[ScoredChunk],
        depth: int = 2,
        limit: int = 10,
    ) -> list[ScoredChunk]:
        """Walk links from seed chunks. Returns newly discovered chunks only."""
        visited = {c.doc_path for c in seed_chunks}
        frontier = list(seed_chunks)
        results: list[ScoredChunk] = []

        for hop in range(depth):
            next_frontier: list[ScoredChunk] = []
            for chunk in frontier:
                raw = self.corpus.get_chunk(chunk.chunk_id)
                links = raw.metadata.get("links", []) if raw else []
                for link in links:
                    for doc_path in self._resolve_link(link):
                        if doc_path in visited:
                            continue
                        visited.add(doc_path)
                        doc_chunks = self.corpus.get_chunks_for_doc(doc_path)
                        if not doc_chunks:
                            continue
                        c = doc_chunks[0]
                        scored = ScoredChunk(
                            chunk_id=c.id,
                            doc_path=c.doc_path,
                            score=chunk.score * (LINK_DECAY ** (hop + 1)),
                            content=c.content,
                            provenance=f"link:hop{hop + 1}",
                            entities=c.metadata.get("entities", []),
                        )
                        results.append(scored)
                        next_frontier.append(scored)
                        if len(results) >= limit:
                            return results
            frontier = next_frontier

        return results

    def _resolve_link(self, link: str) -> list[str]:
        """Match a link target against known corpus doc paths."""
        link_norm = link.lower().strip().replace("\\", "/")
        if link_norm.endswith(".md"):
            link_norm = link_norm[:-3]
        candidates: list[str] = []
        for doc_path in self.corpus.list_docs():
            dp = doc_path.lower().replace("\\", "/")
            stem = dp[:-3] if dp.endswith(".md") else dp
            stem_name = stem.split("/")[-1]
            if stem_name == link_norm or stem == link_norm or dp == link_norm + ".md":
                candidates.append(doc_path)
        return candidates
