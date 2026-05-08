"""
OMPA Bridge — import an OMPA vault into a Locus store.

OMPA (Obsidian-MemPalace-Agnostic) uses the same triple store schema as
Locus (subject / predicate / object / valid_from / valid_to / source), so
KG import is a direct row copy.

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
from pathlib import Path

logger = logging.getLogger(__name__)

_OMPA_KG_PATH = Path(".palace") / "knowledge_graph.sqlite3"


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

        # 1. Index markdown documents
        chunks = self.engine.corpus.add_directory(self.vault_path, pattern=pattern)
        logger.info("OMPA bridge: indexed %d chunks from %s", chunks, self.vault_path)

        # 2. Import KG triples
        triples = self._import_kg()

        return {
            "vault": str(self.vault_path),
            "chunks_indexed": chunks,
            "triples_imported": triples,
            "kg_source": str(self.vault_path / _OMPA_KG_PATH),
        }

    def _import_kg(self) -> int:
        """Copy triples from OMPA's knowledge_graph.sqlite3 into Locus KG."""
        kg_db = self.vault_path / _OMPA_KG_PATH
        if not kg_db.exists():
            logger.debug("No OMPA KG found at %s — skipping triple import", kg_db)
            return 0

        count = 0
        try:
            with sqlite3.connect(str(kg_db)) as conn:
                rows = conn.execute(
                    "SELECT subject, predicate, object, valid_from, valid_to, source "
                    "FROM triples"
                ).fetchall()
        except Exception as e:
            logger.warning("Could not read OMPA KG: %s", e)
            return 0

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
                    pass

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
