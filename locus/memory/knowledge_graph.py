"""
Temporal knowledge graph — SQLite-backed triple store.
Standalone adaptation of OMPA's KnowledgeGraph with validity windows.
"""

import re
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source: Optional[str] = None


class TemporalKG:
    """SQLite temporal triple store with validity windows."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS triples (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        subject     TEXT NOT NULL,
        predicate   TEXT NOT NULL,
        object      TEXT NOT NULL,
        valid_from  TEXT,
        valid_to    TEXT,
        source      TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject);
    CREATE INDEX IF NOT EXISTS idx_object  ON triples(object);
    CREATE INDEX IF NOT EXISTS idx_source  ON triples(source);
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self._SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def add_triple(
        self,
        subject: str,
        predicate: str,
        object_: str,
        valid_from: str = None,
        valid_to: str = None,
        source: str = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO triples (subject, predicate, object, valid_from, valid_to, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (subject.strip(), predicate.strip(), object_.strip(), valid_from, valid_to, source),
            )

    def query_entity(self, entity: str, as_of: str = None) -> list[Triple]:
        sql = (
            "SELECT subject, predicate, object, valid_from, valid_to, source "
            "FROM triples WHERE (subject = ? OR object = ?)"
        )
        params: list = [entity, entity]
        if as_of:
            sql += (
                " AND (valid_from IS NULL OR valid_from <= ?)"
                " AND (valid_to   IS NULL OR valid_to   >= ?)"
            )
            params += [as_of, as_of]
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Triple(*row) for row in rows]

    def timeline(self, entity: str) -> list[Triple]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT subject, predicate, object, valid_from, valid_to, source "
                "FROM triples WHERE (subject = ? OR object = ?) "
                "ORDER BY COALESCE(valid_from, '0000') ASC",
                (entity, entity),
            ).fetchall()
        return [Triple(*row) for row in rows]

    def sources_for_entity(self, entity: str, as_of: str = None) -> list[str]:
        """All source documents that contain a triple involving this entity."""
        sql = (
            "SELECT DISTINCT source FROM triples "
            "WHERE (subject = ? OR object = ?) AND source IS NOT NULL"
        )
        params: list = [entity, entity]
        if as_of:
            sql += (
                " AND (valid_from IS NULL OR valid_from <= ?)"
                " AND (valid_to   IS NULL OR valid_to   >= ?)"
            )
            params += [as_of, as_of]
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [r[0] for r in rows]

    def populate_from_text(self, text: str, source: str = None) -> int:
        """Extract wikilinks and tags from markdown text, store as triples."""
        count = 0
        for link in re.findall(r"\[\[([^\]]+)\]\]", text):
            entity = link.split("|")[0].strip()
            if entity and source:
                self.add_triple(entity, "mentioned_in", source, source=source)
                count += 1
        for tag in re.findall(r"#([\w/-]+)", text):
            if source:
                self.add_triple(source, "tagged_as", tag, source=source)
                count += 1
        return count

    def populate_from_file(self, path: Path, base_path: Path = None) -> int:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return 0
        source = (
            str(path.relative_to(base_path)).replace("\\", "/")
            if base_path
            else str(path).replace("\\", "/")
        )
        return self.populate_from_text(text, source=source)

    def stats(self) -> dict:
        with self._conn() as conn:
            triples = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
            entities = conn.execute(
                "SELECT COUNT(DISTINCT subject) FROM triples"
            ).fetchone()[0]
            sources = conn.execute(
                "SELECT COUNT(DISTINCT source) FROM triples WHERE source IS NOT NULL"
            ).fetchone()[0]
        return {
            "triple_count": triples,
            "entity_count": entities,
            "source_count": sources,
        }
