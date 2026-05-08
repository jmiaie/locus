"""
Reciprocal Rank Fusion — combines multiple ranked lists deterministically.
Cormack et al. 2009: score = sum(weight_i / (k + rank)) across all lists.

Weights default to 1.0 (standard RRF). Pass per-list weights to favour
one retrieval signal over another (e.g. KG-first vs BM25-first routing).
"""

from typing import Optional
from .bm25 import ScoredChunk

RRF_K = 60


def rrf_fuse(
    ranked_lists: list[list[ScoredChunk]],
    weights: Optional[list[float]] = None,
    k: int = RRF_K,
    limit: int = 10,
) -> list[ScoredChunk]:
    """
    Fuse ranked lists via weighted RRF. Deduplicates by chunk_id.
    The highest-scoring original ScoredChunk is kept for each id.

    Args:
        ranked_lists: One list per retrieval signal.
        weights:      Per-list multipliers (default 1.0 each).
        k:            RRF constant (default 60).
        limit:        Max results to return.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    best: dict[str, ScoredChunk] = {}
    rrf: dict[str, float] = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, chunk in enumerate(ranked):
            cid = chunk.chunk_id
            rrf[cid] = rrf.get(cid, 0.0) + weight / (k + rank + 1)
            if cid not in best or chunk.score > best[cid].score:
                best[cid] = chunk

    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [best[cid] for cid, _ in fused if cid in best]
