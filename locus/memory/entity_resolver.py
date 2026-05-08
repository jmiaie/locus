"""
Entity resolver — maps entity name variants to canonical names.

Enables transparent de-duplication in the KG:
    "Jeff", "Jeff Milam", "jmiaie"  →  same canonical entity

Manual aliases are added via add_alias(). suggest_aliases() uses
token-overlap similarity to surface candidates for human review.
Resolution is applied transparently inside TemporalKG when a resolver
is attached.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS aliases (
    alias     TEXT PRIMARY KEY,
    canonical TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_canonical ON aliases(canonical);
"""


def _similarity(a: str, b: str) -> float:
    """Token-overlap similarity in [0, 1]."""
    an, bn = a.lower().strip(), b.lower().strip()
    if an == bn:
        return 1.0
    if an in bn or bn in an:
        return 0.85
    ta, tb = set(an.split()), set(bn.split())
    if ta and tb:
        return len(ta & tb) / max(len(ta), len(tb))
    return 0.0


class EntityResolver:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._cache: dict[str, str] = {}
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def resolve(self, entity: str) -> str:
        """Return the canonical name for entity, or entity itself if unknown."""
        if entity in self._cache:
            return self._cache[entity]
        with self._conn() as conn:
            row = conn.execute(
                "SELECT canonical FROM aliases WHERE alias = ?", (entity,)
            ).fetchone()
        canonical = row[0] if row else entity
        self._cache[entity] = canonical
        return canonical

    def add_alias(self, alias: str, canonical: str) -> None:
        """Register alias → canonical mapping."""
        alias, canonical = alias.strip(), canonical.strip()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO aliases (alias, canonical) VALUES (?, ?)",
                (alias, canonical),
            )
        self._cache[alias] = canonical
        logger.info("Alias added: %r -> %r", alias, canonical)

    def list_aliases(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT alias, canonical FROM aliases ORDER BY canonical, alias"
            ).fetchall()
        return [{"alias": r[0], "canonical": r[1]} for r in rows]

    def suggest_aliases(
        self,
        entities: list[str],
        threshold: float = 0.75,
    ) -> list[dict]:
        """
        Return pairs of entity names that may refer to the same entity.
        Caller provides entity list (e.g. from kg.all_entities()).
        """
        suggestions: list[dict] = []
        seen: set[frozenset] = set()
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                key = frozenset([e1, e2])
                if key in seen:
                    continue
                seen.add(key)
                sim = _similarity(e1, e2)
                if sim >= threshold:
                    suggestions.append({
                        "entity_a": e1,
                        "entity_b": e2,
                        "similarity": round(sim, 3),
                    })
        return sorted(suggestions, key=lambda x: x["similarity"], reverse=True)

    def stats(self) -> dict:
        with self._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
        return {"alias_count": count}
