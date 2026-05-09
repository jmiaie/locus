"""
Temporal knowledge graph — SQLite-backed triple store.
Standalone adaptation of OMPA's KnowledgeGraph with validity windows.

Entity aliases are resolved transparently via an optional EntityResolver.
Prose triples are extracted automatically during indexing via extractor.py.
"""

import re
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .entity_resolver import EntityResolver

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

    def __init__(
        self,
        db_path: str,
        resolver: Optional["EntityResolver"] = None,
    ):
        self.db_path = str(db_path)
        self.resolver = resolver
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self._SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _resolve(self, entity: str) -> str:
        return self.resolver.resolve(entity) if self.resolver else entity

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_triple(
        self,
        subject: str,
        predicate: str,
        object_: str,
        valid_from: str | None = None,
        valid_to: str | None = None,
        source: str | None = None,
    ) -> None:
        subject = self._resolve(subject.strip())
        object_ = self._resolve(object_.strip())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO triples "
                "(subject, predicate, object, valid_from, valid_to, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (subject, predicate.strip(), object_, valid_from, valid_to, source),
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query_entity(self, entity: str, as_of: str | None = None) -> list[Triple]:
        entity = self._resolve(entity)
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
        entity = self._resolve(entity)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT subject, predicate, object, valid_from, valid_to, source "
                "FROM triples WHERE (subject = ? OR object = ?) "
                "ORDER BY COALESCE(valid_from, '0000') ASC",
                (entity, entity),
            ).fetchall()
        return [Triple(*row) for row in rows]

    def sources_for_entity(self, entity: str, as_of: str | None = None) -> list[str]:
        entity = self._resolve(entity)
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

    def all_entities(self) -> list[str]:
        """All distinct entities (subjects and objects) in the KG."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT subject FROM triples "
                "UNION "
                "SELECT DISTINCT object FROM triples"
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def _all_triples(self) -> list[Triple]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT subject, predicate, object, valid_from, valid_to, source "
                "FROM triples"
            ).fetchall()
        return [Triple(*row) for row in rows]

    # ------------------------------------------------------------------
    # Bulk population
    # ------------------------------------------------------------------

    def populate_from_text(
        self,
        text: str,
        source: str | None = None,
        extract_prose: bool = True,
    ) -> int:
        """
        Extract triples from markdown text.

        Extracts:
          - Wikilinks  → (entity, mentioned_in, source)
          - Tags       → (source, tagged_as, tag)
          - Prose      → relation triples via pattern matching (if extract_prose)
        """
        count = 0
        for link in re.findall(r"\[\[([^\]]+)\]\]", text):
            entity = self._resolve(link.split("|")[0].strip())
            if entity and source:
                self.add_triple(entity, "mentioned_in", source, source=source)
                count += 1
        for tag in re.findall(r"#([\w/-]+)", text):
            if source:
                self.add_triple(source, "tagged_as", tag, source=source)
                count += 1
        if extract_prose:
            from .extractor import extract_triples_from_text
            for pt in extract_triples_from_text(text):
                self.add_triple(pt.subject, pt.predicate, pt.object, source=source)
                count += 1
        return count

    def populate_from_file(
        self,
        path: Path,
        base_path: Path = None,
        extract_prose: bool = True,
    ) -> int:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return 0
        source = (
            str(path.relative_to(base_path)).replace("\\", "/")
            if base_path
            else str(path).replace("\\", "/")
        )
        return self.populate_from_text(text, source=source, extract_prose=extract_prose)

    # ------------------------------------------------------------------
    # Graph traversal & pattern match
    # ------------------------------------------------------------------

    def traverse(
        self,
        start: str,
        max_depth: int = 2,
        predicate_filter: list[str] | None = None,
        direction: str = "both",
    ) -> dict[str, list[Triple]]:
        """
        BFS traversal from a starting entity.

        Args:
            start:            Starting entity name.
            max_depth:        Maximum hops from start.
            predicate_filter: If set, only follow these predicates.
            direction:        "out" (subject→object), "in" (object→subject),
                              or "both".

        Returns:
            {entity: [triples]} for every entity reachable within max_depth,
            including the start entity itself.
        """
        start = self._resolve(start)
        visited: dict[str, int] = {start: 0}
        result: dict[str, list[Triple]] = {}
        frontier = [start]

        for depth in range(max_depth):
            next_frontier: list[str] = []
            for entity in frontier:
                triples = self.query_entity(entity)
                result[entity] = triples
                for t in triples:
                    if predicate_filter and t.predicate not in predicate_filter:
                        continue
                    candidates: list[str] = []
                    if direction in ("out", "both") and t.subject == entity:
                        candidates.append(t.object)
                    if direction in ("in", "both") and t.object == entity:
                        candidates.append(t.subject)
                    for other in candidates:
                        if other not in visited:
                            visited[other] = depth + 1
                            next_frontier.append(other)
            frontier = next_frontier

        # Collect remaining frontier entities (at max_depth but not expanded)
        for entity in frontier:
            if entity not in result:
                result[entity] = self.query_entity(entity)

        return result

    def match(
        self,
        subject: str = "*",
        predicate: str = "*",
        obj: str = "*",
        as_of: str | None = None,
    ) -> list[Triple]:
        """
        Pattern match over the KG with wildcard support.
        "*" matches any value for that position.

        Examples:
            kg.match("Alice", "*",        "*")      # all facts about Alice
            kg.match("*",     "leads",    "*")      # all leadership facts
            kg.match("*",     "works_at", "Acme")   # everyone at Acme
            kg.match("Alice", "leads",    "*")      # what does Alice lead?
        """
        if subject != "*":
            subject = self._resolve(subject)
        if obj != "*":
            obj = self._resolve(obj)

        sql = (
            "SELECT subject, predicate, object, valid_from, valid_to, source "
            "FROM triples WHERE 1=1"
        )
        params: list = []
        if subject != "*":
            sql += " AND subject = ?"
            params.append(subject)
        if predicate != "*":
            sql += " AND predicate = ?"
            params.append(predicate)
        if obj != "*":
            sql += " AND object = ?"
            params.append(obj)
        if as_of:
            sql += (
                " AND (valid_from IS NULL OR valid_from <= ?)"
                " AND (valid_to   IS NULL OR valid_to   >= ?)"
            )
            params += [as_of, as_of]

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Triple(*row) for row in rows]

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    def find_contradictions(self, entity: str = None) -> list[dict]:
        """
        Find triples that conflict: same subject+predicate, different objects,
        with overlapping (or absent) validity windows.

        If entity is given, scopes the search to that entity's triples.
        """
        if entity:
            entity = self._resolve(entity)
            triples = self.query_entity(entity)
        else:
            triples = self._all_triples()

        groups: dict[tuple, list[Triple]] = {}
        for t in triples:
            groups.setdefault((t.subject, t.predicate), []).append(t)

        contradictions: list[dict] = []
        for (subj, pred), group in groups.items():
            if len(group) < 2:
                continue
            for i, t1 in enumerate(group):
                for t2 in group[i + 1:]:
                    if t1.object == t2.object:
                        continue
                    if _windows_overlap(t1, t2):
                        contradictions.append({
                            "subject": subj,
                            "predicate": pred,
                            "conflict": [
                                {
                                    "object": t1.object,
                                    "valid_from": t1.valid_from,
                                    "valid_to": t1.valid_to,
                                    "source": t1.source,
                                },
                                {
                                    "object": t2.object,
                                    "valid_from": t2.valid_from,
                                    "valid_to": t2.valid_to,
                                    "source": t2.source,
                                },
                            ],
                        })
        return contradictions

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

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


def _windows_overlap(t1: Triple, t2: Triple) -> bool:
    """True if two triples' validity windows overlap (or both are open-ended)."""
    s1 = t1.valid_from or "0000-01-01"
    e1 = t1.valid_to or "9999-12-31"
    s2 = t2.valid_from or "0000-01-01"
    e2 = t2.valid_to or "9999-12-31"
    return s1 <= e2 and s2 <= e1
