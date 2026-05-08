"""
LocusReasoner — multi-hop KG reasoning over the Locus knowledge graph.

find_paths(a, b): BFS to discover all relationship chains between two entities.
reason(question):  extract entities from a question, then explore their
                   KG neighbourhood and surface connecting paths.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory.knowledge_graph import TemporalKG, Triple


@dataclass
class ReasoningPath:
    start: str
    end: str
    hops: list["Triple"] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.hops)

    def narrative(self) -> str:
        """Return a human-readable sentence for this path."""
        if not self.hops:
            return f"{self.start} → {self.end} (direct)"
        parts: list[str] = []
        for triple in self.hops:
            parts.append(f"{triple.subject} –[{triple.predicate}]→ {triple.object}")
        return " → ".join(parts)

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "narrative": self.narrative(),
            "hops": [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object}
                for t in self.hops
            ],
        }


class LocusReasoner:
    """Multi-hop reasoning over a :class:`TemporalKG`."""

    # Simple stopword set for entity extraction
    _STOPWORDS = {
        "what", "who", "where", "when", "why", "how", "is", "are", "was",
        "were", "the", "a", "an", "of", "in", "to", "for", "and", "or",
        "did", "does", "do", "has", "have", "had", "can", "could", "would",
        "should", "tell", "me", "about", "between", "related", "relationship",
        "connection", "link", "connected", "knows", "know",
    }

    def __init__(self, kg: "TemporalKG") -> None:
        self._kg = kg

    # ------------------------------------------------------------------
    # Path finding (BFS)
    # ------------------------------------------------------------------

    def find_paths(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int = 3,
        predicate_filter: list[str] | None = None,
    ) -> list[ReasoningPath]:
        """BFS from *entity_a* to *entity_b*; returns all shortest paths up to *max_depth*."""
        entity_a = self._kg._resolve(entity_a)
        entity_b = self._kg._resolve(entity_b)

        if entity_a == entity_b:
            return [ReasoningPath(start=entity_a, end=entity_b, hops=[])]

        # BFS state: (current_node, path_of_triples)
        queue: deque[tuple[str, list[Any]]] = deque([(entity_a, [])])
        visited: set[str] = {entity_a}
        found: list[ReasoningPath] = []
        target_depth = max_depth

        while queue:
            current, path = queue.popleft()

            if len(path) >= target_depth:
                continue

            # Outgoing edges
            neighbours = self._kg.traverse(
                current,
                max_depth=1,
                predicate_filter=predicate_filter,
                direction="out",
            )
            for node, triples in neighbours.items():
                if node == entity_a:
                    continue
                for triple in triples:
                    new_path = path + [triple]
                    if node == entity_b:
                        found.append(ReasoningPath(start=entity_a, end=entity_b, hops=new_path))
                        target_depth = len(new_path)  # only same-length paths after first find
                    elif node not in visited and len(new_path) < target_depth:
                        visited.add(node)
                        queue.append((node, new_path))

        return found

    # ------------------------------------------------------------------
    # Freeform reasoning
    # ------------------------------------------------------------------

    def reason(self, question: str, max_depth: int = 3) -> dict:
        """Extract entities from *question*, explore their KG neighbourhood,
        and surface connecting paths.

        Returns a dict with keys:
            question, entities_detected, reasoning_chains, entity_neighborhood, chain_count
        """
        entities = self._extract_entities(question)

        # For each entity, gather immediate neighbourhood
        neighbourhood: dict[str, list[dict]] = {}
        for ent in entities:
            resolved = self._kg._resolve(ent)
            triples = self._kg.query_entity(resolved)
            neighbourhood[ent] = [
                {"predicate": t.predicate, "object": t.object}
                for t in triples
            ]

        # Find paths between every pair of detected entities
        chains: list[dict] = []
        ent_list = list(entities)
        for i, a in enumerate(ent_list):
            for b in ent_list[i + 1 :]:
                paths = self.find_paths(a, b, max_depth=max_depth)
                for p in paths:
                    chains.append(p.to_dict())

        return {
            "question": question,
            "entities_detected": ent_list,
            "reasoning_chains": chains,
            "entity_neighborhood": neighbourhood,
            "chain_count": len(chains),
        }

    # ------------------------------------------------------------------
    # Internal: naive entity extraction
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Return capitalized tokens that are not stopwords."""
        # Known entities in the KG take priority
        known = {e.lower(): e for e in self._kg.all_entities()}
        words = re.findall(r"[A-Za-z][A-Za-z0-9_'-]*", text)
        seen: set[str] = set()
        result: list[str] = []
        for w in words:
            low = w.lower()
            if low in self._STOPWORDS:
                continue
            canonical = known.get(low)
            if canonical and canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
            elif w[0].isupper() and w not in seen:
                seen.add(w)
                result.append(w)
        return result
