"""
AnnotationStore — SQLite-backed chunk annotations.

Attach labels and optional free-text notes to any chunk ID.  Labels are
arbitrary strings (e.g. "important", "outdated", "reviewed").  Multiple
labels per chunk are supported.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class AnnotationStore:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS annotations (
        chunk_id   TEXT NOT NULL,
        label      TEXT NOT NULL,
        note       TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (chunk_id, label)
    );
    CREATE INDEX IF NOT EXISTS idx_ann_label    ON annotations(label);
    CREATE INDEX IF NOT EXISTS idx_ann_chunk_id ON annotations(chunk_id);
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self._SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def annotate(self, chunk_id: str, label: str, note: str | None = None) -> None:
        """Add or update a label/note on *chunk_id*."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO annotations (chunk_id, label, note) VALUES (?, ?, ?)",
                (chunk_id, label, note),
            )

    def remove(self, chunk_id: str, label: str) -> bool:
        """Remove a specific label from *chunk_id*. Returns True if found."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM annotations WHERE chunk_id = ? AND label = ?",
                (chunk_id, label),
            )
        return cur.rowcount > 0

    def clear_chunk(self, chunk_id: str) -> int:
        """Remove all annotations for *chunk_id*. Returns count removed."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM annotations WHERE chunk_id = ?", (chunk_id,))
        return cur.rowcount

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> list[dict]:
        """Return all annotations for *chunk_id*."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT label, note, created_at FROM annotations WHERE chunk_id = ? ORDER BY created_at",
                (chunk_id,),
            ).fetchall()
        return [{"label": r[0], "note": r[1], "created_at": r[2]} for r in rows]

    def chunks_with_label(self, label: str) -> list[str]:
        """Return all chunk_ids that carry *label*."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM annotations WHERE label = ? ORDER BY created_at DESC",
                (label,),
            ).fetchall()
        return [r[0] for r in rows]

    def all_labels(self) -> list[str]:
        """Return the distinct set of labels in use."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT label FROM annotations ORDER BY label"
            ).fetchall()
        return [r[0] for r in rows]

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
            labels = conn.execute("SELECT COUNT(DISTINCT label) FROM annotations").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(DISTINCT chunk_id) FROM annotations").fetchone()[0]
        return {"total_annotations": total, "distinct_labels": labels, "annotated_chunks": chunks}
