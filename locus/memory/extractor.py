"""
Prose triple extractor — extracts subject-predicate-object triples from
natural language text using pattern matching.

No models or external dependencies required. Handles the most common
relationship patterns in technical and organisational writing:

    Alice leads the Infrastructure team  → (Alice, leads, Infrastructure team)
    Bob works at Acme Corp               → (Bob, works_at, Acme Corp)
    ProjectX is part of Platform         → (ProjectX, part_of, Platform)
    Locus replaced the old RAG system    → (Locus, replaced, old RAG system)

Quality guards:
  - Minimum entity length: 2 chars
  - Maximum entity length: 60 chars
  - Sentences longer than 40 words are skipped (noisy extractions)
  - Cap at 60 triples per document
  - Pronouns and generic terms filtered
"""

import re
from dataclasses import dataclass

__all__ = ["extract_triples_from_text", "ProseTriple"]

_NON_ENTITY = frozenset({
    "this", "that", "these", "those", "it", "they", "he", "she", "we",
    "you", "i", "there", "here", "what", "which", "who", "the", "a", "an",
    "its", "their", "his", "her", "our", "your", "my", "any", "each", "all",
})

_MAX_SENTENCE_WORDS = 40
_MAX_ENTITY_LEN = 60
_MIN_ENTITY_LEN = 2
_MAX_TRIPLES = 60


@dataclass
class ProseTriple:
    subject: str
    predicate: str
    object: str


# (compiled_pattern, subject_group, predicate_string, object_group)
_RAW_PATTERNS: list[tuple[str, int, str, int]] = [
    # Alice is a/an/the engineer  (object may start lower-case after article)
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+is\s+(?:a|an|the)\s+([A-Za-z][^,.!?\n]{1,55})", 1, "is_a", 2),
    # Alice is EngineerRole (no article, object starts capital)
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+is\s+([A-Z][A-Za-z][^,.!?\n]{1,55})", 1, "is", 2),
    # Alice works at/for Acme
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+works?\s+(?:at|for)\s+([A-Z][A-Za-z][^,.!?\n]{1,55})", 1, "works_at", 2),
    # Alice leads/led (the) X
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:leads?|led)\s+(?:the\s+)?([A-Za-z][^,.!?\n]{1,55})", 1, "leads", 2),
    # Alice owns/owned (the) X
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:owns?|owned)\s+(?:the\s+)?([A-Za-z][^,.!?\n]{1,55})", 1, "owns", 2),
    # Alice created/built/developed/authored (the) X
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:created?|built|developed|authored|wrote)\s+(?:the\s+)?([A-Z][A-Za-z][^,.!?\n]{1,55})", 1, "created", 2),
    # Alice reports to Bob
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+reports?\s+to\s+([A-Z][A-Za-z][^,.!?\n]{1,55})", 1, "reports_to", 2),
    # X is part of Y
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+is\s+part\s+of\s+(?:the\s+)?([A-Z][A-Za-z][^,.!?\n]{1,55})", 1, "part_of", 2),
    # X replaced (the) Y
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+replaced?\s+(?:the\s+)?([A-Z][A-Za-z][^,.!?\n]{1,55})", 1, "replaced", 2),
    # X uses/used (the) Y
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+uses?\s+(?:the\s+)?([A-Z][A-Za-z][^,.!?\n]{1,55})", 1, "uses", 2),
    # X depends on (the) Y
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+depends?\s+on\s+(?:the\s+)?([A-Z][A-Za-z][^,.!?\n]{1,55})", 1, "depends_on", 2),
    # X is responsible for Y
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+is\s+responsible\s+for\s+([A-Za-z][^,.!?\n]{1,55})", 1, "responsible_for", 2),
    # X is managed/owned/led/run by Y
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+is\s+(?:managed|owned|led|run)\s+by\s+([A-Z][A-Za-z][^,.!?\n]{1,55})", 1, "managed_by", 2),
    # X introduced Y / X introduced the Y
    (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+introduced\s+(?:the\s+)?([A-Z][A-Za-z][^,.!?\n]{1,55})", 1, "introduced", 2),
]

_PATTERNS = [(re.compile(p), sg, pred, og) for p, sg, pred, og in _RAW_PATTERNS]


def _clean(text: str) -> str:
    return re.sub(r"[,;.!?\s]+$", "", text).strip()


def _valid(text: str) -> bool:
    t = text.strip()
    return (
        _MIN_ENTITY_LEN <= len(t) <= _MAX_ENTITY_LEN
        and t.lower() not in _NON_ENTITY
        and not re.match(r"^\d+$", t)
    )


def extract_triples_from_text(text: str) -> list[ProseTriple]:
    """Extract relation triples from plain text using pattern matching."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    results: list[ProseTriple] = []

    for sentence in sentences:
        if len(sentence.split()) > _MAX_SENTENCE_WORDS:
            continue
        for pattern, sg, predicate, og in _PATTERNS:
            for match in pattern.finditer(sentence):
                subj = _clean(match.group(sg))
                obj = _clean(match.group(og))
                if _valid(subj) and _valid(obj):
                    results.append(ProseTriple(subject=subj, predicate=predicate, object=obj))
                if len(results) >= _MAX_TRIPLES:
                    return results

    return results
