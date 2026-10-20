"""
OMPA Bridge — import an OMPA vault into a Locus store.

OMPA (Obsidian-MemPalace-Agnostic) stores triples in
``.palace/knowledge_graph.sqlite3`` with columns subject / predicate / object /
valid_from / valid_to / confidence / source_file (see
``ompa/knowledge_graph.py``).  The bridge maps ``source_file`` onto Locus's
``source`` field.  Older or foreign tables that use ``source`` are accepted too.
A triples table missing the required columns raises
:class:`OMPASchemaError` instead of silently importing nothing.

The bridge:
  1. Indexes all markdown files in the OMPA vault into the Locus corpus
  2. Copies triples from .palace/knowledge_graph.sqlite3 into the Locus KG
  3. Optionally imports palace wing/room structure as KG triples

Usage:
    from locus import LocusEngine
    from locus.bridge.ompa import OMPABridge

    engine = LocusEngine(".locus")
    bridge = OMPABridge(engine, vault_path="/path/to/ompa-vault")
    result = bridge.ingest()

CLI:
    locus ingest-ompa /path/to/ompa-vault --store .locus
"""

import logging
import sqlite3
from contextlib import closing
from pathlib import Path

logger = logging.getLogger(__name__)

_OMPA_KG_PATH = Path(".palace") / "knowledge_graph.sqlite3"
_REQUIRED_COLUMNS = ("subject", "predicate", "object", "valid_from", "valid_to")
_SOURCE_COLUMNS = ("source_file", "source")
# OMPA stamps its KG contract version in PRAGMA user_version; 0 = pre-stamp (same shape as 1)
_SUPPORTED_KG_VERSIONS = (0, 1)


class OMPASchemaError(RuntimeError):
    """The OMPA knowledge graph does not have the schema the bridge expects."""


class OMPABridge:
    def __init__(self, engine, vault_path: str | Path):
        from ..core import LocusEngine
        self.engine: LocusEngine = engine
        self.vault_path = Path(vault_path).expanduser()

    def ingest(self, pattern: str = "**/*.md") -> dict:
        """
        Import the OMPA vault into Locus.

        Steps:
          1. Index all markdown files (uses checksum dedup — safe to re-run)
          2. Copy KG triples from OMPA's SQLite database
          3. Return import summary

        Returns:
            dict with chunks_indexed, triples_imported, kg_source
        """
        if not self.vault_path.exists():
            return {"error": f"Vault path not found: {self.vault_path}"}

        # Read (and schema-check) the KG first so a bad vault fails before indexing
        rows = self._read_kg_rows()

        # 1. Index markdown documents
        chunks = self.engine.corpus.add_directory(self.vault_path, pattern=pattern)
        logger.info("OMPA bridge: indexed %d chunks from %s", chunks, self.vault_path)

        # 2. Import KG triples
        triples = self._import_rows(rows)

        return {
            "vault": str(self.vault_path),
            "chunks_indexed": chunks,
            "triples_imported": triples,
            "kg_source": str(self.vault_path / _OMPA_KG_PATH),
        }

    def _import_kg(self) -> int:
        """Copy triples from OMPA's knowledge_graph.sqlite3 into Locus KG."""
        return self._import_rows(self._read_kg_rows())

    def _read_kg_rows(self) -> list[tuple]:
        kg_db = self.vault_path / _OMPA_KG_PATH
        if not kg_db.exists():
            logger.debug("No OMPA KG found at %s — skipping triple import", kg_db)
            return []

        uri = f"{kg_db.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in _SUPPORTED_KG_VERSIONS:
                raise OMPASchemaError(
                    f"OMPA KG at {kg_db} has schema version {version}; "
                    f"this Locus reads versions {list(_SUPPORTED_KG_VERSIONS)}"
                )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(triples)")}
            if not columns:
                raise OMPASchemaError(f"No 'triples' table in OMPA KG at {kg_db}")
            missing = [c for c in _REQUIRED_COLUMNS if c not in columns]
            source_col = next((c for c in _SOURCE_COLUMNS if c in columns), None)
            if missing or source_col is None:
                if source_col is None:
                    missing.append("source_file")
                raise OMPASchemaError(
                    f"OMPA KG at {kg_db} is missing column(s) {missing}; "
                    f"found {sorted(columns)}"
                )
            rows = conn.execute(
                "SELECT subject, predicate, object, valid_from, valid_to, "
                f"{source_col} FROM triples"
            ).fetchall()
        return rows

    def _import_rows(self, rows: list[tuple]) -> int:
        count = 0
        for subject, predicate, obj, valid_from, valid_to, source in rows:
            if subject and predicate and obj:
                try:
                    self.engine.kg.add_triple(
                        subject, predicate, obj,
                        valid_from=valid_from,
                        valid_to=valid_to,
                        source=source,
                    )
                    count += 1
                except Exception:
                    logger.warning("OMPA bridge: skipped triple %r", (subject, predicate, obj), exc_info=True)

        logger.info("OMPA bridge: imported %d KG triples", count)
        return count

    def stats(self) -> dict:
        """Return what's available in the OMPA vault without importing."""
        kg_db = self.vault_path / _OMPA_KG_PATH
        triple_count = 0
        if kg_db.exists():
            try:
                with sqlite3.connect(str(kg_db)) as conn:
                    triple_count = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
            except Exception:
                pass

        doc_count = sum(
            1 for _ in self.vault_path.glob("**/*.md")
            if ".palace" not in str(_) and ".obsidian" not in str(_)
        )
        return {
            "vault": str(self.vault_path),
            "markdown_docs": doc_count,
            "kg_triples": triple_count,
            "kg_available": kg_db.exists(),
        }
