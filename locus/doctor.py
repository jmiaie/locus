"""
LocusDoctor — structured health and diagnostic report for a Locus store.

Checks:
  corpus          — doc/chunk counts, empty store warning
  knowledge_graph — triple count, entity count, contradiction count
  bulletin        — tier fill levels, persistence status
  entity_resolver — alias count
  store_size      — disk usage of the .locus directory
  version         — installed Locus version

Each check returns PASS / WARN / FAIL with a message and detail dict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import LocusEngine

logger = logging.getLogger(__name__)

_PASS = "pass"
_WARN = "warn"
_FAIL = "fail"


@dataclass
class HealthCheck:
    name: str
    status: str           # "pass" | "warn" | "fail"
    message: str
    details: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == _PASS


class LocusDoctor:
    """
    Runs a battery of health checks on a LocusEngine instance.

    Usage:
        doc = LocusDoctor(engine)
        print(doc.report())
        checks = doc.run()   # [HealthCheck, ...]
    """

    def __init__(self, engine: LocusEngine):
        self.engine = engine

    def run(self) -> list[HealthCheck]:
        return [
            self._check_corpus(),
            self._check_kg(),
            self._check_bulletin(),
            self._check_resolver(),
            self._check_store_size(),
            self._check_version(),
        ]

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_corpus(self) -> HealthCheck:
        stats = self.engine.corpus.stats()
        if stats["doc_count"] == 0:
            return HealthCheck("corpus", _WARN, "No documents indexed — run locus_index first", stats)
        if stats["chunk_count"] == 0:
            return HealthCheck("corpus", _FAIL, "Documents registered but zero chunks stored", stats)
        ratio = stats["chunk_count"] / max(stats["doc_count"], 1)
        msg = f"{stats['doc_count']} docs | {stats['chunk_count']} chunks | avg {ratio:.1f} chunks/doc"
        return HealthCheck("corpus", _PASS, msg, stats)

    def _check_kg(self) -> HealthCheck:
        stats = self.engine.kg.stats()
        try:
            n_contradictions = len(self.engine.kg.find_contradictions())
        except Exception:
            n_contradictions = 0
        details = {**stats, "contradictions": n_contradictions}

        if stats["triple_count"] == 0:
            return HealthCheck("knowledge_graph", _WARN, "KG is empty — index docs to populate", details)
        status = _WARN if n_contradictions > 10 else _PASS
        msg = (
            f"{stats['triple_count']} triples | "
            f"{stats['entity_count']} entities | "
            f"{n_contradictions} contradictions"
        )
        return HealthCheck("knowledge_graph", status, msg, details)

    def _check_bulletin(self) -> HealthCheck:
        stats = self.engine.bulletin.stats()
        if stats["tier0_pinned"] >= 10:
            return HealthCheck(
                "bulletin", _WARN,
                "Tier 0 (pinned) is full — oldest entries will be demoted",
                stats,
            )
        msg = (
            f"Tier0={stats['tier0_pinned']} pinned | "
            f"Tier1={stats['tier1_hot']} hot | "
            f"persistent={stats['persistent']}"
        )
        return HealthCheck("bulletin", _PASS, msg, stats)

    def _check_resolver(self) -> HealthCheck:
        stats = self.engine.resolver.stats()
        msg = f"{stats['alias_count']} aliases registered"
        return HealthCheck("entity_resolver", _PASS, msg, stats)

    def _check_store_size(self) -> HealthCheck:
        try:
            total = sum(
                f.stat().st_size
                for f in Path(self.engine.store_path).rglob("*")
                if f.is_file()
            )
            mb = total / (1024 * 1024)
        except Exception:
            return HealthCheck("store_size", _WARN, "Could not compute store size", {})
        details = {"size_mb": round(mb, 2), "store_path": str(self.engine.store_path)}
        status = _WARN if mb > 500 else _PASS
        return HealthCheck("store_size", status, f"{mb:.2f} MB at {self.engine.store_path}", details)

    def _check_version(self) -> HealthCheck:
        from .core import __version__
        return HealthCheck("version", _PASS, f"Locus v{__version__}", {"version": __version__})

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    def report(self) -> str:
        checks = self.run()
        icons = {_PASS: "[PASS]", _WARN: "[WARN]", _FAIL: "[FAIL]"}
        lines = ["Locus Doctor", "=" * 44]
        for c in checks:
            lines.append(f"{icons[c.status]:6}  {c.name:<20} {c.message}")
        n_pass = sum(1 for c in checks if c.status == _PASS)
        n_warn = sum(1 for c in checks if c.status == _WARN)
        n_fail = sum(1 for c in checks if c.status == _FAIL)
        lines += ["", f"Result: {n_pass} pass  {n_warn} warn  {n_fail} fail"]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        checks = self.run()
        return {
            "checks": [
                {"name": c.name, "status": c.status, "message": c.message, "details": c.details}
                for c in checks
            ],
            "summary": {
                "pass": sum(1 for c in checks if c.status == _PASS),
                "warn": sum(1 for c in checks if c.status == _WARN),
                "fail": sum(1 for c in checks if c.status == _FAIL),
            },
        }
