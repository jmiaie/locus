"""
RelevanceFeedback — persist and apply relevance signals to future retrieval.

Agents or users can mark a retrieved chunk as relevant (+) or irrelevant (-)
for a given query.  The reranker reads these signals and applies a score
adjustment: +0.25 boost for relevant, -0.40 penalty for irrelevant.

Signals are stored per (chunk_id, query_hash) pair, so different queries
can have different relevance for the same chunk.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


_RELEVANT_BOOST   =  0.25
_IRRELEVANT_BOOST = -0.40


class RelevanceFeedback:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS feedback (
        chunk_id   TEXT NOT NULL,
        query_hash TEXT NOT NULL,
        relevant   INTEGER NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (chunk_id, query_hash)
    );
    CREATE INDEX IF NOT EXISTS idx_fb_chunk ON feedback(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_fb_query ON feedback(query_hash);
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self._SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _hash(query: str) -> str:
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def mark(self, chunk_id: str, query: str, relevant: bool) -> None:
        """Record that *chunk_id* is relevant (True) or irrelevant (False) for *query*."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO feedback (chunk_id, query_hash, relevant) VALUES (?, ?, ?)",
                (chunk_id, self._hash(query), 1 if relevant else -1),
            )

    def clear(self, chunk_id: str, query: str | None = None) -> int:
        """Remove feedback for *chunk_id*, optionally scoped to *query*."""
        with self._conn() as conn:
            if query is not None:
                cur = conn.execute(
                    "DELETE FROM feedback WHERE chunk_id = ? AND query_hash = ?",
                    (chunk_id, self._hash(query)),
                )
            else:
                cur = conn.execute("DELETE FROM feedback WHERE chunk_id = ?", (chunk_id,))
        return cur.rowcount

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def score_adjustment(self, chunk_id: str, query: str) -> float:
        """Return the score adjustment for *chunk_id* given *query*.

        +0.25  if marked relevant
        -0.40  if marked irrelevant
         0.0   if no signal recorded
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT relevant FROM feedback WHERE chunk_id = ? AND query_hash = ?",
                (chunk_id, self._hash(query)),
            ).fetchone()
        if row is None:
            return 0.0
        return _RELEVANT_BOOST if row[0] > 0 else _IRRELEVANT_BOOST

    def get(self, chunk_id: str) -> list[dict]:
        """Return all feedback signals for *chunk_id*."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT query_hash, relevant, created_at FROM feedback WHERE chunk_id = ? ORDER BY created_at DESC",
                (chunk_id,),
            ).fetchall()
        return [
            {
                "query_hash": r[0],
                "relevant": r[1] > 0,
                "created_at": r[2],
            }
            for r in rows
        ]

    def stats(self) -> dict:
        with self._conn() as conn:
            total    = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            positive = conn.execute("SELECT COUNT(*) FROM feedback WHERE relevant > 0").fetchone()[0]
            negative = conn.execute("SELECT COUNT(*) FROM feedback WHERE relevant < 0").fetchone()[0]
        return {
            "total_signals": total,
            "relevant": positive,
            "irrelevant": negative,
        }
