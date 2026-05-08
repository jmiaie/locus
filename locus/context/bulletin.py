"""
Tiered context bulletin board — adapted from OMPAminnow's SwarmBulletin.

Tier 0 (Pinned):  max 10, never auto-removed. Manual pin or auto-promoted.
Tier 1 (Hot):     max 50, sorted by effective_score with hit boost and age decay.
Tier 2 (Archive): cold entries written to disk and dropped from memory.

effective_score = base_score + (hits * HIT_BOOST) - (rounds_elapsed * AGE_DECAY)

Persistence: when db_path is provided all mutations are written through to SQLite
so the hot tier survives process restarts.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TIER0_MAX = 10
TIER1_MAX = 50
PROMOTE_THRESHOLD = 0.85
HIT_BOOST = 0.05
AGE_DECAY = 0.02

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bulletin_entries (
    chunk_id       TEXT PRIMARY KEY,
    doc_path       TEXT NOT NULL,
    content        TEXT NOT NULL,
    base_score     REAL NOT NULL DEFAULT 0.5,
    hits           INTEGER NOT NULL DEFAULT 0,
    rounds_elapsed INTEGER NOT NULL DEFAULT 0,
    tier           INTEGER NOT NULL DEFAULT 1,
    provenance     TEXT NOT NULL DEFAULT ''
);
"""


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
    def __init__(
        self,
        archive_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        self.archive_path = Path(archive_path) if archive_path else None
        self.db_path = Path(db_path) if db_path else None

        self._tier0: list[BulletinEntry] = []
        self._tier1: list[BulletinEntry] = []
        self._all: dict[str, BulletinEntry] = {}

        if self.db_path:
            self._init_db()
            self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA)

    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    def _load(self) -> None:
        """Reconstruct tier0/tier1 from SQLite on startup."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc_path, content, base_score, hits, "
                "rounds_elapsed, tier, provenance "
                "FROM bulletin_entries WHERE tier IN (0, 1) "
                "ORDER BY tier ASC, base_score DESC"
            ).fetchall()

        for row in rows:
            entry = BulletinEntry(
                chunk_id=row[0],
                doc_path=row[1],
                content=row[2],
                base_score=row[3],
                hits=row[4],
                rounds_elapsed=row[5],
                tier=row[6],
                provenance=row[7],
            )
            self._all[entry.chunk_id] = entry
            if entry.tier == 0:
                self._tier0.append(entry)
            else:
                self._tier1.append(entry)

        self._tier1.sort(key=lambda e: e.effective_score, reverse=True)
        logger.debug("Bulletin loaded: %d pinned, %d hot", len(self._tier0), len(self._tier1))

    def _persist(self, entry: BulletinEntry) -> None:
        if not self.db_path:
            return
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bulletin_entries "
                "(chunk_id, doc_path, content, base_score, hits, rounds_elapsed, tier, provenance) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.chunk_id, entry.doc_path, entry.content,
                    entry.base_score, entry.hits, entry.rounds_elapsed,
                    entry.tier, entry.provenance,
                ),
            )

    def _delete_persisted(self, chunk_id: str) -> None:
        if not self.db_path:
            return
        with self._conn() as conn:
            conn.execute("DELETE FROM bulletin_entries WHERE chunk_id = ?", (chunk_id,))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

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
            self._persist(entry)
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
            self._persist(entry)

    def _add_to_tier1(self, entry: BulletinEntry) -> None:
        self._tier1.append(entry)
        self._tier1.sort(key=lambda e: e.effective_score, reverse=True)
        if len(self._tier1) > TIER1_MAX:
            evicted = self._tier1.pop()
            evicted.tier = 2
            self._archive(evicted)
            self._delete_persisted(evicted.chunk_id)

    def _promote_to_pin(self, chunk_id: str) -> None:
        entry = self._all.get(chunk_id)
        if not entry or entry.tier == 0:
            return
        if entry in self._tier1:
            self._tier1.remove(entry)
        entry.tier = 0
        self._tier0.append(entry)
        self._persist(entry)
        if len(self._tier0) > TIER0_MAX:
            demoted = min(self._tier0, key=lambda e: e.effective_score)
            self._tier0.remove(demoted)
            demoted.tier = 1
            self._add_to_tier1(demoted)
            self._persist(demoted)
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
                self._delete_persisted(entry.chunk_id)
                archived += 1
            else:
                self._persist(entry)
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

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

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
            "persistent": self.db_path is not None,
        }
