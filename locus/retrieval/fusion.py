"""
Reciprocal Rank Fusion — combines multiple ranked lists deterministically.
Cormack et al. 2009: score = sum(1 / (k + rank)) across all lists.
"""

from .bm25 import ScoredChunk

RRF_K = 60


def rrf_fuse(
    ranked_lists: list[list[ScoredChunk]],
    k: int = RRF_K,
    limit: int = 10,
) -> list[ScoredChunk]:
    """
    Fuse ranked lists via RRF. Deduplicates by chunk_id.
    The highest-scoring original ScoredChunk is kept for each id.
    """
    best: dict[str, ScoredChunk] = {}
    rrf: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            cid = chunk.chunk_id
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in best or chunk.score > best[cid].score:
                best[cid] = chunk

    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [best[cid] for cid, _ in fused if cid in best]
