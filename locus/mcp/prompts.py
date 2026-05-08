"""
MCP Prompt templates for Locus.

Each template is rendered with live context from a LocusEngine instance,
so the returned messages arrive pre-filled with retrieved knowledge.

Templates:
  locus_research             — retrieve + format context for a topic
  locus_entity_summary       — KG facts + document context for an entity
  locus_timeline             — temporal diff for an entity between two dates
  locus_contradiction_analysis — find + frame contradictions for review
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import LocusEngine

__all__ = ["list_prompts", "render_prompt"]

_DEFS: list[dict] = [
    {
        "name": "locus_research",
        "description": "Research a topic using the Locus knowledge base. Retrieves relevant context and formats it for analysis.",
        "arguments": [
            {"name": "topic", "description": "The topic to research", "required": True},
            {"name": "limit", "description": "Max chunks to retrieve (default 5)", "required": False},
        ],
    },
    {
        "name": "locus_entity_summary",
        "description": "Summarize all KG facts and relevant documents for a specific entity.",
        "arguments": [
            {"name": "entity", "description": "Entity name", "required": True},
        ],
    },
    {
        "name": "locus_timeline",
        "description": "Show how knowledge about an entity changed between two dates using the temporal KG.",
        "arguments": [
            {"name": "entity",    "description": "Entity name",              "required": True},
            {"name": "date_from", "description": "Start date (YYYY-MM-DD)", "required": True},
            {"name": "date_to",   "description": "End date (YYYY-MM-DD)",   "required": True},
        ],
    },
    {
        "name": "locus_contradiction_analysis",
        "description": "Identify and frame KG contradictions for human resolution.",
        "arguments": [
            {"name": "entity", "description": "Scope to entity (omit for all)", "required": False},
        ],
    },
]


def list_prompts() -> list[dict]:
    return _DEFS


def render_prompt(name: str, args: dict, engine: LocusEngine) -> list[dict]:
    """Return a list of MCP message objects with context pre-filled."""
    dispatch = {
        "locus_research":              _research,
        "locus_entity_summary":        _entity_summary,
        "locus_timeline":              _timeline,
        "locus_contradiction_analysis": _contradictions,
    }
    fn = dispatch.get(name)
    if fn is None:
        return [_user(f"Unknown prompt: {name}")]
    try:
        return fn(args, engine)
    except Exception as e:
        return [_user(f"Error rendering prompt '{name}': {type(e).__name__}: {e}")]


# ------------------------------------------------------------------
# Renderers
# ------------------------------------------------------------------

def _research(args: dict, engine: LocusEngine) -> list[dict]:
    topic = args.get("topic", "")
    limit = min(int(args.get("limit", 5)), 20)
    chunks = engine.retrieve(topic, limit=limit)
    context = engine.format_context(chunks, include_hot=False)
    return [_user(
        f"Research the following topic using the Locus knowledge base context below.\n\n"
        f"**Topic:** {topic}\n\n"
        f"{context or '(No context found — run locus_index first.)'}"
    )]


def _entity_summary(args: dict, engine: LocusEngine) -> list[dict]:
    entity = args.get("entity", "")
    facts_result = engine.query_entity(entity)
    facts = facts_result.get("facts", [])
    facts_lines = "\n".join(
        f"  - {f['subject']} --{f['predicate']}--> {f['object']}"
        + (f"  [{f['valid_from']}..{f['valid_to'] or 'present'}]" if f.get("valid_from") else "")
        for f in facts
    ) or "  (no KG facts found)"
    chunks = engine.retrieve(entity, limit=3)
    context = engine.format_context(chunks, include_hot=False)
    return [_user(
        f"Summarize everything known about **{entity}**.\n\n"
        f"## Knowledge Graph ({len(facts)} facts)\n{facts_lines}\n\n"
        f"{context or '(No document context found.)'}"
    )]


def _timeline(args: dict, engine: LocusEngine) -> list[dict]:
    entity    = args.get("entity", "")
    date_from = args.get("date_from", "")
    date_to   = args.get("date_to", "")
    triples   = engine.kg.timeline(entity)

    in_range = [t for t in triples if _window_overlaps(t.valid_from, t.valid_to, date_from, date_to)]
    lines = [
        f"  - [{t.valid_from or '?'} → {t.valid_to or 'present'}] "
        f"{t.subject} --{t.predicate}--> {t.object}"
        + (f"  ({t.source})" if t.source else "")
        for t in in_range
    ]
    facts_text = "\n".join(lines) or "  (no temporal facts in this range)"
    return [_user(
        f"Describe how knowledge about **{entity}** changed "
        f"between {date_from} and {date_to}.\n\n"
        f"## Timeline ({len(in_range)} facts)\n{facts_text}"
    )]


def _contradictions(args: dict, engine: LocusEngine) -> list[dict]:
    entity = args.get("entity")
    result = engine.find_contradictions(entity)
    items  = result["contradictions"]

    if not items:
        scope = f" for '{entity}'" if entity else ""
        return [_user(f"No contradictions found{scope}. The knowledge graph is consistent.")]

    lines = []
    for c in items:
        a, b = c["conflict"][0], c["conflict"][1]
        lines.append(
            f"  - **{c['subject']} --{c['predicate']}-->** conflict:\n"
            f"    '{a['object']}' (from {a.get('source') or 'unknown'}) "
            f"vs '{b['object']}' (from {b.get('source') or 'unknown'})"
        )
    scope = f" for '{entity}'" if entity else " (all entities)"
    return [_user(
        f"Analyze the following {len(items)} contradiction(s){scope}. "
        f"Determine which is correct and recommend a resolution.\n\n"
        + "\n".join(lines)
    )]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _user(text: str) -> dict:
    return {"role": "user", "content": {"type": "text", "text": text}}


def _window_overlaps(
    valid_from: str | None,
    valid_to: str | None,
    date_from: str,
    date_to: str,
) -> bool:
    f = valid_from or "0000-01-01"
    t = valid_to   or "9999-12-31"
    return f <= date_to and date_from <= t
