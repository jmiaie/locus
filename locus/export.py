"""
KGExporter — export the Locus knowledge graph to external formats.

Formats:
  graphml   — XML, for Gephi / Cytoscape / yEd (full layout tooling)
  jsonl     — one triple per line, for scripting / piping
  dot       — Graphviz DOT, for quick PDF/PNG via dot -Tpng

Usage:
    from locus import LocusEngine
    from locus.export import KGExporter

    engine = LocusEngine(".locus")
    exp = KGExporter(engine.kg)
    exp.to_graphml("kg.graphml")
    exp.to_jsonl("kg.jsonl")
    exp.to_dot("kg.dot")

CLI:
    locus export-kg kg.graphml --format graphml
    locus export-kg kg.jsonl   --format jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory.knowledge_graph import TemporalKG

logger = logging.getLogger(__name__)


class KGExporter:
    def __init__(self, kg: TemporalKG):
        self.kg = kg

    # ------------------------------------------------------------------
    # GraphML
    # ------------------------------------------------------------------

    def to_graphml(self, path: str | Path) -> int:
        """Export to GraphML for Gephi / Cytoscape / yEd."""
        triples = self.kg._all_triples()
        nodes: set[str] = set()
        for t in triples:
            nodes.add(t.subject)
            nodes.add(t.object)

        def esc(s: str) -> str:
            return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/graphml"',
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '         xsi:schemaLocation="http://graphml.graphdrawing.org/graphml '
            'http://graphml.graphdrawing.org/graphml/1.0/graphml.xsd">',
            '  <key id="label" for="edge" attr.name="label" attr.type="string"/>',
            '  <key id="source_doc" for="edge" attr.name="source_doc" attr.type="string"/>',
            '  <graph id="locus_kg" edgedefault="directed">',
        ]
        for node in sorted(nodes):
            lines.append(f'    <node id="{esc(node)}"/>')
        for i, t in enumerate(triples):
            lines += [
                f'    <edge id="e{i}" source="{esc(t.subject)}" target="{esc(t.object)}">',
                f'      <data key="label">{esc(t.predicate)}</data>',
                f'      <data key="source_doc">{esc(t.source or "")}</data>',
                f'    </edge>',
            ]
        lines += ["  </graph>", "</graphml>"]

        Path(path).write_text("\n".join(lines), encoding="utf-8")
        logger.info("GraphML export: %d nodes, %d edges -> %s", len(nodes), len(triples), path)
        return len(triples)

    # ------------------------------------------------------------------
    # JSONL
    # ------------------------------------------------------------------

    def to_jsonl(self, path: str | Path) -> int:
        """Export triples as JSONL (one JSON object per line)."""
        triples = self.kg._all_triples()
        lines = [
            json.dumps({
                "subject": t.subject,
                "predicate": t.predicate,
                "object": t.object,
                "valid_from": t.valid_from,
                "valid_to": t.valid_to,
                "source": t.source,
            })
            for t in triples
        ]
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        logger.info("JSONL export: %d triples -> %s", len(triples), path)
        return len(triples)

    # ------------------------------------------------------------------
    # DOT
    # ------------------------------------------------------------------

    def to_dot(self, path: str | Path, max_edges: int = 300) -> int:
        """Export to Graphviz DOT format. cap at max_edges to keep graphs readable."""
        triples = self.kg._all_triples()[:max_edges]

        def esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"')

        lines = [
            "digraph locus_kg {",
            "  rankdir=LR;",
            '  node [shape=box fontsize=10];',
            '  edge [fontsize=9];',
        ]
        for t in triples:
            s = f'"{esc(t.subject)}"'
            o = f'"{esc(t.object)}"'
            p = esc(t.predicate)
            lines.append(f'  {s} -> {o} [label="{p}"];')
        lines.append("}")

        Path(path).write_text("\n".join(lines), encoding="utf-8")
        logger.info("DOT export: %d edges -> %s", len(triples), path)
        return len(triples)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def export(self, path: str | Path, fmt: str = None) -> int:
        """
        Export to path, auto-detecting format from extension if fmt is None.
        Supported: graphml, jsonl, dot
        """
        path = Path(path)
        if fmt is None:
            ext = path.suffix.lower().lstrip(".")
            fmt = {"graphml": "graphml", "jsonl": "jsonl", "dot": "dot", "gv": "dot"}.get(ext, "jsonl")

        if fmt == "graphml":
            return self.to_graphml(path)
        if fmt == "jsonl":
            return self.to_jsonl(path)
        if fmt == "dot":
            return self.to_dot(path)
        raise ValueError(f"Unsupported export format: {fmt}. Use graphml, jsonl, or dot.")
