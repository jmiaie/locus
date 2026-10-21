"""
Structural retriever — scores documents by frontmatter metadata.

Activates only when the query contains structural signals:
  - Date ranges:  "Q1 2025", "2024", "last year"
  - Tags:         "#engineering", "tagged ops", "about security"
  - Doc types:    "meeting", "decision", "ADR", "incident"

Returns an empty list when no structural signals are detected, so it adds
zero noise to RRF when the query is purely semantic.
"""

import re
import logging
from ..memory.corpus import Corpus
from .bm25 import ScoredChunk

logger = logging.getLogger(__name__)

_DOC_TYPES = frozenset({
    "meeting", "decision", "adr", "incident", "note", "standup",
    "retro", "review", "report", "proposal", "rfc", "spec", "postmortem",
})

_QUARTER_MONTHS = {
    1: ("01", "03"),
    2: ("04", "06"),
    3: ("07", "09"),
    4: ("10", "12"),
}


def _extract_date_range(query: str) -> tuple[str, str] | None:
    m = re.search(r"\bQ([1-4])\s*(\d{4})\b", query, re.I)
    if m:
        qtr, year = int(m.group(1)), m.group(2)
        start_m, end_m = _QUARTER_MONTHS[qtr]
        return (f"{year}-{start_m}-01", f"{year}-{end_m}-31")
    m = re.search(r"\b(\d{4})\b", query)
    if m:
        year = int(m.group(1))
        if 1990 <= year <= 2099:
            return (f"{year}-01-01", f"{year}-12-31")
    return None


def _extract_tags(query: str) -> list[str]:
    tags = re.findall(r"#(\w+)", query)
    tags += re.findall(r"\btagged\s+(\w+)", query, re.I)
    tags += re.findall(r"\babout\s+(\w{4,})\b", query, re.I)
    return [t.lower() for t in tags]


def _extract_type(query: str) -> str | None:
    """First doc type mentioned in the query (by position, then name).

    Deterministic: iterating the frozenset directly made the answer depend on
    PYTHONHASHSEED when a query named two types.
    """
    q = query.lower()
    found: list[tuple[int, str]] = []
    for t in sorted(_DOC_TYPES):
        m = re.search(r"\b" + t + r"\b", q)
        if m:
            found.append((m.start(), t))
    return min(found)[1] if found else None


def _date_in_range(date_str: str, start: str, end: str) -> bool:
    if not date_str:
        return False
    try:
        d = date_str[:10]
        return start[:10] <= d <= end[:10]
    except Exception:
        return False


def _tag_overlap(doc_tags: str, query_tags: list[str]) -> float:
    if not doc_tags or not query_tags:
        return 0.0
    dt = set(re.split(r"[\s,;]+", doc_tags.lower()))
    hits = sum(1 for t in query_tags if t in dt)
    return hits / len(query_tags)


class StructuralRetriever:
    """Scores documents by frontmatter date, tags, and type fields."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def search(self, query: str, limit: int = 10) -> list[ScoredChunk]:
        date_range = _extract_date_range(query)
        query_tags = _extract_tags(query)
        query_type = _extract_type(query)

        if not date_range and not query_tags and not query_type:
            return []

        scored: list[tuple[str, float, str]] = []

        for doc_path in self.corpus.list_docs():
            chunks = self.corpus.get_chunks_for_doc(doc_path)
            if not chunks:
                continue
            fm = chunks[0].metadata.get("frontmatter", {})
            score = 0.0

            if date_range and _date_in_range(fm.get("date", ""), *date_range):
                score += 1.0

            if query_tags:
                score += _tag_overlap(fm.get("tags", ""), query_tags)

            if query_type:
                doc_type = fm.get("type", fm.get("category", "")).lower()
                if query_type in doc_type or doc_type in query_type:
                    score += 1.0

            if score > 0:
                scored.append((doc_path, score, chunks[0].id))

        scored.sort(key=lambda x: (-x[1], x[0]))

        results: list[ScoredChunk] = []
        for doc_path, score, chunk_id in scored[:limit]:
            chunk = self.corpus.get_chunk(chunk_id)
            if chunk:
                results.append(ScoredChunk(
                    chunk_id=chunk_id,
                    doc_path=doc_path,
                    score=score,
                    content=chunk.content,
                    provenance="structural",
                    entities=chunk.metadata.get("entities", []),
                ))
        return results


def score_frontmatter(query: str, frontmatter: dict) -> float:
    """Structural match score of one document's frontmatter for *query*."""
    date_range = _extract_date_range(query)
    query_tags = _extract_tags(query)
    query_type = _extract_type(query)
    if not date_range and not query_tags and not query_type:
        return 0.0
    fm = frontmatter or {}
    score = 0.0
    if date_range and _date_in_range(str(fm.get("date", "")), *date_range):
        score += 1.0
    if query_tags:
        score += _tag_overlap(str(fm.get("tags", "")), query_tags)
    if query_type:
        doc_type = str(fm.get("type", fm.get("category", ""))).lower()
        if doc_type and (query_type in doc_type or doc_type in query_type):
            score += 1.0
    return score
