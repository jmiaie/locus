"""
Query intent classifier — routes queries to the dominant retrieval signal.

KG-first:   entity/factual queries ("who is X?", "when did Y happen?")
BM25-first: procedural/narrative queries ("how does X work?", "explain Y")
Balanced:   default when neither pattern dominates

Used by LocusEngine.retrieve() to weight the RRF fusion.
"""

import re
from enum import Enum

__all__ = ["QueryIntent", "classify_query", "INTENT_WEIGHTS"]


class QueryIntent(Enum):
    KG_FIRST = "kg_first"
    BM25_FIRST = "bm25_first"
    BALANCED = "balanced"


_KG_PATTERNS = [
    r"\b(who is|who are|who was|who were)\b",
    r"\b(what is|what are|what was|what were)\b.{0,40}\b(role|position|title|status|owner|lead)\b",
    r"\b(when did|when was|when is|when were)\b",
    r"\b(where is|where was|where does|where did)\b",
    r"\b(list|enumerate|show).{0,20}\b(all|every|each)\b",
    r"\b(relationship|connection|link|association).{0,20}\bbetween\b",
    r"\bfacts?\b.{0,20}\babout\b",
    r"\btell me about\b",
    r"\bwhat.{0,20}\bknow.{0,20}\babout\b",
    r"\b(owned by|managed by|led by|created by|built by|reported to)\b",
    r"\bwho\b.{0,30}\b(owns|manages|leads|created|built|reports to)\b",
    r"\b(timeline|history|record|log)\b.{0,20}\bof\b",
]

_BM25_PATTERNS = [
    r"\b(how does|how do|how to|how can|how should|how would)\b",
    r"\b(explain|describe|summarize|overview|walkthrough|elaborate)\b",
    r"\b(process|procedure|workflow|pipeline|steps?|guide|tutorial)\b",
    r"\b(what happened|incident|outage|failure|error|bug)\b",
    r"\b(difference between|compare|versus|vs\.?|contrast)\b",
    r"\b(why|reason|cause|because)\b",
    r"\b(example|sample|instance|demo|illustration)\b",
    r"\brecent(ly)?\b",
    r"\b(architecture|design|structure|pattern)\b",
    r"\b(document|note|file|page|article)\b.{0,20}\babout\b",
]


def classify_query(query: str) -> QueryIntent:
    q = query.lower()
    kg_score = sum(1 for p in _KG_PATTERNS if re.search(p, q, re.I))
    bm25_score = sum(1 for p in _BM25_PATTERNS if re.search(p, q, re.I))
    if kg_score > bm25_score:
        return QueryIntent.KG_FIRST
    if bm25_score > kg_score:
        return QueryIntent.BM25_FIRST
    return QueryIntent.BALANCED


# RRF weights per intent: [bm25_weight, kg_weight, link_weight]
INTENT_WEIGHTS: dict[QueryIntent, list[float]] = {
    QueryIntent.KG_FIRST:   [0.5, 2.0, 0.5],
    QueryIntent.BM25_FIRST: [2.0, 0.5, 0.5],
    QueryIntent.BALANCED:   [1.0, 1.0, 1.0],
}
