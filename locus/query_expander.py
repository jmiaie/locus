"""
QueryExpander — KG-backed query expansion for Locus.

Given a query string, enriches it with:
  1. Canonical names for any alias tokens that appear in the query
  2. First-hop KG neighbours of recognised entities (objects of outgoing edges)

No embeddings.  Expansion is purely structural — driven by the resolver alias
table and the existing KG triples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory.knowledge_graph import TemporalKG
    from .memory.entity_resolver import EntityResolver


@dataclass
class ExpansionResult:
    original: str
    expanded: str
    added_terms: list[str] = field(default_factory=list)
    entity_matches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "expanded": self.expanded,
            "added_terms": self.added_terms,
            "entity_matches": self.entity_matches,
        }


class QueryExpander:
    """Expand a query using alias resolution and KG first-hop neighbours."""

    _SPLIT_RE = re.compile(r"[^\w']+")

    def __init__(self, kg: "TemporalKG", resolver: "EntityResolver") -> None:
        self._kg = kg
        self._resolver = resolver

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def expand(self, query: str, max_expansions: int = 5) -> ExpansionResult:
        """Return an *ExpansionResult* with the enriched query string.

        Parameters
        ----------
        query:           Original query text.
        max_expansions:  Maximum number of new terms to inject.
        """
        added: list[str] = []
        entity_matches: list[str] = []

        candidates = self._extract_candidates(query)
        known_lower = {e.lower(): e for e in self._kg.all_entities()}

        for token in candidates:
            if len(added) >= max_expansions:
                break

            low = token.lower()
            # 1. Check if token resolves to a canonical name different from itself
            canonical = self._resolver.resolve(token)
            if canonical.lower() != low and canonical not in added:
                added.append(canonical)
                entity_matches.append(canonical)

            # 2. Check if token matches a known KG entity directly
            matched_entity = canonical if canonical.lower() in known_lower else known_lower.get(low)
            if matched_entity and matched_entity not in entity_matches:
                entity_matches.append(matched_entity)

            # 3. Add first-hop neighbours of the matched entity
            if matched_entity:
                for neighbour in self._neighbours(matched_entity):
                    if len(added) >= max_expansions:
                        break
                    if neighbour.lower() not in query.lower() and neighbour not in added:
                        added.append(neighbour)

        if added:
            expanded = query + " " + " ".join(added)
        else:
            expanded = query

        return ExpansionResult(
            original=query,
            expanded=expanded,
            added_terms=added,
            entity_matches=list(dict.fromkeys(entity_matches)),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_candidates(self, query: str) -> list[str]:
        """Return unique tokens from *query* that could be entity names."""
        tokens = self._SPLIT_RE.split(query)
        seen: set[str] = set()
        result: list[str] = []
        for t in tokens:
            if t and t not in seen and len(t) > 1:
                seen.add(t)
                result.append(t)
                # Also try multi-word combos (bigrams)
        return result

    def _neighbours(self, entity: str) -> list[str]:
        """Return objects of outgoing edges from *entity* (depth-1 BFS)."""
        subgraph = self._kg.traverse(entity, max_depth=1, direction="out")
        result: list[str] = []
        for node, triples in subgraph.items():
            for triple in triples:
                obj = triple.object
                if obj != entity and obj not in result:
                    result.append(obj)
        return result
