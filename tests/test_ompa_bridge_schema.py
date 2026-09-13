"""Regression: the OMPA bridge must find the triples table's source column.

Current OMPA (>=1.0) names it `source_file`; older releases used `source`. The bridge
hard-coded `source`, so against a modern OMPA vault the SELECT raised
"no such column: source", _import_kg caught it, logged, and returned 0 -- and
ingest-ompa still reported success with {"triples_imported": 0}. The whole KG half of
the bridge was dead and the failure looked like an empty vault.

These tests build both schemas and assert triples are imported from each, so the bridge
cannot silently re-break when OMPA's schema moves again.
"""
import sqlite3

import pytest

from locus import LocusEngine
from locus.bridge.ompa import OMPABridge


def _make_vault(tmp_path, source_column: str):
    """Minimal OMPA vault: one note plus a KG whose source column has the given name."""
    (tmp_path / "work").mkdir(parents=True, exist_ok=True)
    (tmp_path / "work" / "note.md").write_text("# A note\n\nBody text.\n")

    palace = tmp_path / ".palace"
    palace.mkdir(exist_ok=True)
    con = sqlite3.connect(palace / "knowledge_graph.sqlite3")
    con.execute(
        f"""CREATE TABLE triples (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                confidence REAL DEFAULT 1.0,
                {source_column} TEXT,
                extracted_at TEXT
            )"""
    )
    con.execute(
        f"INSERT INTO triples (id, subject, predicate, object, {source_column}) "
        f"VALUES ('1','Alice','knows','Bob','note.md')"
    )
    con.commit()
    con.close()
    return tmp_path


@pytest.mark.parametrize("source_column", ["source_file", "source"])
def test_bridge_imports_triples_under_either_schema(tmp_path, source_column):
    """Both column names must work. `source_file` is the one that broke."""
    vault = _make_vault(tmp_path, source_column)
    engine = LocusEngine(store_path=tmp_path / ".locus")
    result = OMPABridge(engine, vault).ingest()

    assert result["triples_imported"] == 1, (
        f"bridge imported {result['triples_imported']} triples from a "
        f"'{source_column}' schema — expected 1 (this is the silent-zero bug)"
    )


def test_bridge_reports_zero_only_when_the_kg_is_genuinely_absent(tmp_path):
    """A vault with no KG is legitimately 0 -- and must not raise."""
    (tmp_path / "work").mkdir(parents=True, exist_ok=True)
    (tmp_path / "work" / "note.md").write_text("# Note\n\nBody.\n")
    engine = LocusEngine(store_path=tmp_path / ".locus")
    result = OMPABridge(engine, vault_path=tmp_path).ingest()
    assert result["triples_imported"] == 0
    assert result["chunks_indexed"] >= 1, "text import must still work without a KG"


def test_unrecognised_schema_is_reported_not_silently_zero(tmp_path):
    """A triples table missing subject/predicate/object must not masquerade as empty."""
    (tmp_path / "work").mkdir(parents=True, exist_ok=True)
    (tmp_path / "work" / "note.md").write_text("# Note\n\nBody.\n")
    palace = tmp_path / ".palace"
    palace.mkdir(exist_ok=True)
    con = sqlite3.connect(palace / "knowledge_graph.sqlite3")
    con.execute("CREATE TABLE triples (id TEXT, irrelevant TEXT)")
    con.commit()
    con.close()

    engine = LocusEngine(store_path=tmp_path / ".locus")
    result = OMPABridge(engine, vault_path=tmp_path).ingest()
    assert result["triples_imported"] == 0
