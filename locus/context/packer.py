"""
ContextPacker — budget-aware context assembly for prompt injection.

Takes a ranked list of ScoredChunks and packs them into a token budget:
  - Greedy fill in rank order
  - Deduplicates near-duplicate chunks from the same document
  - Groups chunks by source document for better LLM readability
  - Reports tokens_used, chunks_included, truncation status

Usage:
    packer = ContextPacker(budget=4000)
    packed = packer.pack(chunks)
    prompt = f"Context:\\n{packed.text}\\n\\nQuestion: {query}"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..retrieval.bm25 import ScoredChunk

logger = logging.getLogger(__name__)

_WORDS_PER_TOKEN = 1.0 / 1.3   # approximate inverse of token estimator


@dataclass
class PackedContext:
    text: str
    chunks_included: int
    chunks_available: int
    tokens_used: int
    token_budget: int
    truncated: bool
    sources: list[str] = field(default_factory=list)


class ContextPacker:
    """
    Packs ScoredChunks into a token budget.

    Args:
        budget:          Token budget for the packed context.
        dedup_threshold: Two chunks from the same doc within this many
                         words of each other are considered duplicates;
                         the lower-ranked one is dropped.
        header_tokens:   Overhead tokens per chunk header (estimated).
    """

    def __init__(
        self,
        budget: int = 4000,
        dedup_threshold: int = 50,
        header_tokens: int = 15,
    ):
        self.budget = budget
        self.dedup_threshold = dedup_threshold
        self.header_tokens = header_tokens

    def pack(
        self,
        chunks: list[ScoredChunk],
        token_budget: int | None = None,
    ) -> PackedContext:
        """
        Pack chunks into a token budget, returning a formatted PackedContext.

        Ordering:
          1. Deduplicate overlapping chunks from the same document.
          2. Fill greedily in rank order until budget is exhausted.
          3. Group chunks by document in the output (locality for LLM).
        """
        budget = token_budget if token_budget is not None else self.budget
        available = len(chunks)

        deduplicated = self._dedup(chunks)
        selected = self._fill(deduplicated, budget)
        text = self._format(selected)
        tokens_used = self._estimate_tokens(text)

        sources = list(dict.fromkeys(c.doc_path for c in selected))

        return PackedContext(
            text=text,
            chunks_included=len(selected),
            chunks_available=available,
            tokens_used=tokens_used,
            token_budget=budget,
            truncated=len(selected) < len(deduplicated),
            sources=sources,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dedup(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """
        Remove lower-ranked chunks that overlap significantly with a
        higher-ranked chunk from the same document.
        """
        kept: list[ScoredChunk] = []
        # track (doc_path, start_word) for included chunks
        included_positions: dict[str, list[int]] = {}

        for chunk in chunks:
            positions = included_positions.get(chunk.doc_path, [])
            # Check if this chunk's content overlaps with any kept chunk
            chunk_start = getattr(chunk, "start_word", -1)
            if chunk_start >= 0 and any(
                abs(chunk_start - pos) < self.dedup_threshold
                for pos in positions
            ):
                continue  # too similar — skip
            kept.append(chunk)
            if chunk_start >= 0:
                included_positions.setdefault(chunk.doc_path, []).append(chunk_start)

        return kept

    def _fill(self, chunks: list[ScoredChunk], budget: int) -> list[ScoredChunk]:
        """Greedy fill: take chunks in rank order until budget exhausted."""
        selected: list[ScoredChunk] = []
        tokens_remaining = budget

        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk.content) + self.header_tokens
            if chunk_tokens > tokens_remaining:
                break
            selected.append(chunk)
            tokens_remaining -= chunk_tokens

        return selected

    def _format(self, chunks: list[ScoredChunk]) -> str:
        """Format chunks grouped by source document."""
        if not chunks:
            return ""

        # Group by doc, preserving rank order within groups
        groups: dict[str, list[ScoredChunk]] = {}
        for chunk in chunks:
            groups.setdefault(chunk.doc_path, []).append(chunk)

        parts: list[str] = []
        for doc_path, doc_chunks in groups.items():
            # Document header
            parts.append(f"### {doc_path}")
            for i, chunk in enumerate(doc_chunks, 1):
                meta_parts = [f"via {chunk.provenance}"]
                if hasattr(chunk, "score"):
                    meta_parts.append(f"score={chunk.score:.4f}")
                parts.append(f"*[{meta_parts[0]}]*")
                parts.append(chunk.content)

        return "\n\n".join(parts)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)
