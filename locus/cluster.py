"""
LocusCluster — multi-node Locus for cross-corpus retrieval.

Each node is a named LocusEngine backed by its own .locus store.
The cluster queries all (or selected) nodes simultaneously and fuses
results via RRF, tagging provenance with the node name.

Registry is persisted as a JSON file so nodes survive restarts.

Designed for the Jarv/Kai/Tai use case:
    cluster = LocusCluster("~/.locus/cluster.json")
    cluster.add_node("jarv", "/path/to/jarv/.locus")
    cluster.add_node("kai",  "/path/to/kai/.locus")
    cluster.add_node("tai",  "/path/to/tai/.locus")
    chunks = cluster.retrieve("how does auth work?")
    # -> results from all three nodes, fused via RRF
    # -> provenance: "jarv:bm25", "kai:kg", "tai:structural" etc.
"""

import json
import logging
from pathlib import Path

from .core import LocusEngine
from .retrieval.bm25 import ScoredChunk
from .retrieval.fusion import rrf_fuse

logger = logging.getLogger(__name__)


class LocusCluster:
    """
    Multi-node Locus cluster with persistent registry.

    Usage:
        cluster = LocusCluster("cluster.json")
        cluster.add_node("docs",   ".locus/docs")
        cluster.add_node("code",   ".locus/code")
        results = cluster.retrieve("deployment process")
    """

    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        self._nodes: dict[str, LocusEngine] = {}
        if self.registry_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, name: str, store_path: str) -> dict:
        """Register a named node. Creates the store if it doesn't exist."""
        self._nodes[name] = LocusEngine(store_path=store_path)
        self._save()
        logger.info("Cluster node added: %s at %s", name, store_path)
        return {"added": name, "store_path": str(store_path)}

    def remove_node(self, name: str) -> dict:
        """Unregister a node (does not delete its store)."""
        if name not in self._nodes:
            return {"error": f"Node not found: {name}"}
        del self._nodes[name]
        self._save()
        return {"removed": name}

    def get_node(self, name: str) -> LocusEngine | None:
        return self._nodes.get(name)

    def node_names(self) -> list[str]:
        return list(self._nodes.keys())

    # ------------------------------------------------------------------
    # Cross-node retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        nodes: list[str] = None,
        as_of: str = None,
    ) -> list[ScoredChunk]:
        """
        Query all nodes (or a subset) and fuse results via RRF.
        Provenance is prefixed with the node name: "jarv:bm25", "kai:kg".
        """
        targets = (
            {k: v for k, v in self._nodes.items() if k in nodes}
            if nodes
            else dict(self._nodes)
        )
        if not targets:
            return []

        ranked_lists: list[list[ScoredChunk]] = []
        for name, engine in targets.items():
            results = engine.retrieve(query, limit=limit, as_of=as_of)
            for r in results:
                r.provenance = f"{name}:{r.provenance}"
            ranked_lists.append(results)

        return rrf_fuse(ranked_lists, limit=limit)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def list_nodes(self) -> list[dict]:
        return [
            {
                "name": name,
                "store_path": str(engine.store_path),
                "corpus": engine.corpus.stats(),
                "kg": engine.kg.stats(),
            }
            for name, engine in self._nodes.items()
        ]

    def status(self) -> dict:
        return {
            "node_count": len(self._nodes),
            "registry": str(self.registry_path),
            "nodes": {name: engine.status() for name, engine in self._nodes.items()},
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {
                name: str(engine.store_path)
                for name, engine in self._nodes.items()
            }
        }
        self.registry_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            for name, store_path in data.get("nodes", {}).items():
                try:
                    self._nodes[name] = LocusEngine(store_path=store_path)
                    logger.info("Loaded cluster node: %s", name)
                except Exception as e:
                    logger.warning("Could not load node %s: %s", name, e)
        except Exception as e:
            logger.warning("Could not load cluster registry %s: %s", self.registry_path, e)
