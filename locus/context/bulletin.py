"""
Tiered context bulletin board — adapted from OMPAminnow's SwarmBulletin.

Tier 0 (Pinned):  max 10, never auto-removed. Manual pin or auto-promoted.
Tier 1 (Hot):     max 50, sorted by effective_score with hit boost and age decay.
Tier 2 (Archive): cold entries written to disk and dropped from memory.

effective_score = base_score + (hits * HIT_BOOST) - (rounds_elapsed * AGE_DECAY)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TIER0_MAX = 10
TIER1_MAX = 50
PROMOTE_THRESHOLD = 0.85
HIT_BOOST = 0.05
AGE_DECAY = 0.02


@dataclass
class BulletinEntry:
    chunk_id: str
    doc_path: str
    content: str
    base_score: float
    hits: int = 0
    rounds_elapsed: int = 0
    tier: int = 1
    provenance: str = ""

    @property
    def effective_score(self) -> float:
        return self.base_score + (self.hits * HIT_BOOST) - (self.rounds_elapsed * AGE_DECAY)


class ContextBulletin:
    def __init__(self, archive_path: Optional[Path] = None):
        self.archive_path = Path(archive_path) if archive_path else None
        self._tier0: list[BulletinEntry] = []
        self._tier1: list[BulletinEntry] = []
        self._all: dict[str, BulletinEntry] = {}

    def record_hit(
        self,
        chunk_id: str,
        content: str = "",
        doc_path: str = "",
        base_score: float = 0.5,
        provenance: str = "",
    ) -> None:
        if chunk_id in self._all:
            entry = self._all[chunk_id]
            entry.hits += 1
            if entry.tier == 1 and entry.effective_score >= PROMOTE_THRESHOLD:
                self._promote_to_pin(chunk_id)
        else:
            entry = BulletinEntry(
                chunk_id=chunk_id,
                doc_path=doc_path,
                content=content[:500],
                base_score=base_score,
                hits=1,
                provenance=provenance,
            )
            self._all[chunk_id] = entry
            self._add_to_tier1(entry)

    def _add_to_tier1(self, entry: BulletinEntry) -> None:
        self._tier1.append(entry)
        self._tier1.sort(key=lambda e: e.effective_score, reverse=True)
        if len(self._tier1) > TIER1_MAX:
            evicted = self._tier1.pop()
            evicted.tier = 2
            self._archive(evicted)

    def _promote_to_pin(self, chunk_id: str) -> None:
        entry = self._all.get(chunk_id)
        if not entry or entry.tier == 0:
            return
        if entry in self._tier1:
            self._tier1.remove(entry)
        entry.tier = 0
        self._tier0.append(entry)
        if len(self._tier0) > TIER0_MAX:
            demoted = min(self._tier0, key=lambda e: e.effective_score)
            self._tier0.remove(demoted)
            demoted.tier = 1
            self._add_to_tier1(demoted)
        logger.info("Promoted %s to Tier 0 (pinned)", chunk_id)

    def promote_to_pin(self, chunk_id: str) -> bool:
        if chunk_id not in self._all:
            return False
        self._promote_to_pin(chunk_id)
        return True

    def tick(self) -> int:
        """Advance one round: age Tier 1 entries, archive those that go negative."""
        archived = 0
        for entry in list(self._tier1):
            entry.rounds_elapsed += 1
            if entry.effective_score < 0:
                self._tier1.remove(entry)
                entry.tier = 2
                self._archive(entry)
                archived += 1
        self._tier1.sort(key=lambda e: e.effective_score, reverse=True)
        return archived

    def _archive(self, entry: BulletinEntry) -> None:
        if not self.archive_path:
            return
        self.archive_path.mkdir(parents=True, exist_ok=True)
        (self.archive_path / f"{entry.chunk_id}.json").write_text(
            json.dumps({
                "chunk_id": entry.chunk_id,
                "doc_path": entry.doc_path,
                "content": entry.content,
                "base_score": entry.base_score,
                "hits": entry.hits,
            }),
            encoding="utf-8",
        )

    def inject(self, token_limit: int = 1500) -> str:
        """Format hot-tier context for injection into a prompt."""
        parts: list[str] = []
        budget = token_limit
        for entry in self._tier0:
            snippet = entry.content[:300]
            cost = len(snippet.split()) + 20
            if cost > budget:
                break
            parts.append(f"[PINNED] {entry.doc_path}: {snippet}")
            budget -= cost
        for entry in self._tier1[:20]:
            snippet = entry.content[:200]
            cost = len(snippet.split()) + 15
            if cost > budget:
                break
            parts.append(f"[HOT] {entry.doc_path}: {snippet}")
            budget -= cost
        return "\n\n".join(parts)

    def stats(self) -> dict:
        return {
            "tier0_pinned": len(self._tier0),
            "tier1_hot": len(self._tier1),
            "total_tracked": len(self._all),
        }
