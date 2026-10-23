"""
Metadata filters for retrieval: restrict results by source path, document
type, tags, dates (range or point-in-time) and arbitrary frontmatter fields.

Filters act on the *document* a chunk belongs to (its path and frontmatter)
and are applied after ranking, so they never change the BM25 scores of the
chunks that pass.  :meth:`LocusEngine.retrieve` deepens the candidate pool
until ``limit`` matching chunks are found or the matches are exhausted, so a
filter never returns fewer results than exist.

Accepted as a :class:`MetadataFilter` or a plain dict (the MCP / JSON form)::

    {
      "path": "specs/**" | ["specs/*", "rfis/*"],   # glob(s) on doc_path, any
      "exclude_path": "archive/**",                  # glob(s), none
      "type": "spec" | ["spec", "rfi"],              # frontmatter `type`, any (case-insensitive)
      "tags": ["concrete", "mri"],                   # doc has ALL of these tags
      "tags_any": ["roofing", "sprinkler"],          # doc has ANY of these tags
      "date_from": "2026-03-01", "date_to": "2026-06-30",   # frontmatter `date` (inclusive)
      "as_of": "2026-04-01",      # in force on that date: effective/date <= as_of and not
                                  # superseded (frontmatter `supersedes`) by then
      "meta": {"status": "Closed",                   # equality (case-insensitive string compare)
               "revision": {"gte": 1},               # ops: eq ne in nin gt gte lt lte contains exists
               "spec_section": {"in": ["03 30 00", "07 54 23"]}}
    }

Dates compare as ISO-8601 strings (``YYYY-MM-DD``).  A document without the
field a date bound needs does not match that bound.  Pure standard library,
deterministic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Callable, Iterable, Mapping, Optional

__all__ = ["MetadataFilter", "parse_tags", "META_OPS"]

META_OPS = ("eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "contains", "exists")
_KEYS = {"path", "exclude_path", "type", "tags", "tags_any", "date_from", "date_to", "as_of", "meta"}
_DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?")


def _tuple(v: Any) -> tuple[str, ...]:
    if v is None or v == "":
        return ()
    if isinstance(v, str):
        return (v,)
    return tuple(str(x) for x in v)


def parse_tags(raw: Any) -> set[str]:
    """Frontmatter tags (``a, b`` / ``[a, b]`` / ``#a #b`` / list) -> lower-case set."""
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return set().union(*(parse_tags(x) for x in raw)) if raw else set()
    items = re.split(r"[,\s]+", str(raw).strip().strip("[]"))
    return {t.strip().strip("'\"#").lower() for t in items if t.strip().strip("'\"#")}


def _date(v: Any) -> Optional[str]:
    if v is None:
        return None
    m = _DATE.match(str(v).strip().strip("'\""))
    return m.group(0) if m else None


def _norm(v: Any) -> Any:
    """Numbers compare as numbers, everything else as lower-case strings."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().strip("'\"")
    try:
        return float(s)
    except ValueError:
        return s.lower()


def _cmp(a: Any, b: Any, op: str) -> bool:
    x, y = _norm(a), _norm(b)
    if type(x) is not type(y):  # mixed number/string: compare as strings
        x, y = str(a).strip().lower(), str(b).strip().lower()
    if op == "gt":
        return x > y
    if op == "gte":
        return x >= y
    if op == "lt":
        return x < y
    return x <= y  # lte


def _meta_match(value: Any, cond: Any) -> bool:
    present = value is not None and value != ""
    if not isinstance(cond, Mapping):
        cond = {"eq": cond}
    for op, arg in cond.items():
        if op not in META_OPS:
            raise ValueError(f"Unknown meta filter operator {op!r}; use one of {META_OPS}")
        if op == "exists":
            if bool(arg) != present:
                return False
            continue
        if not present:
            return False
        if op == "eq" and _norm(value) != _norm(arg):
            return False
        if op == "ne" and _norm(value) == _norm(arg):
            return False
        if op in ("in", "nin"):
            hit = _norm(value) in {_norm(a) for a in _tuple(arg)}
            if hit != (op == "in"):
                return False
        if op == "contains" and str(arg).lower() not in str(value).lower():
            return False
        if op in ("gt", "gte", "lt", "lte") and not _cmp(value, arg, op):
            return False
    return True


@dataclass(frozen=True)
class MetadataFilter:
    path: tuple[str, ...] = ()
    exclude_path: tuple[str, ...] = ()
    doc_type: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    tags_any: tuple[str, ...] = ()
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    as_of: Optional[str] = None
    meta: tuple[tuple[str, str], ...] = field(default=())  # (field, json-encoded condition)

    # -- construction ---------------------------------------------------
    @classmethod
    def from_value(cls, value: "MetadataFilter | Mapping | None") -> Optional["MetadataFilter"]:
        if value is None or isinstance(value, MetadataFilter):
            return value if (value is None or not value.is_empty()) else None
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, Mapping):
            raise TypeError("filters must be a dict or MetadataFilter")
        unknown = set(value) - _KEYS - {"doc_type"}
        if unknown:
            raise ValueError(f"Unknown filter key(s) {sorted(unknown)}; use {sorted(_KEYS)}")
        meta = value.get("meta") or {}
        if not isinstance(meta, Mapping):
            raise ValueError("filters.meta must be an object {field: value | {op: value}}")
        for cond in meta.values():  # validate operators early
            if isinstance(cond, Mapping):
                bad = set(cond) - set(META_OPS)
                if bad:
                    raise ValueError(f"Unknown meta filter operator(s) {sorted(bad)}; use {META_OPS}")
        dates = {}
        for k in ("date_from", "date_to", "as_of"):
            if value.get(k):
                d = _date(value[k])
                if d is None:
                    raise ValueError(f"filters.{k} must be an ISO date (YYYY-MM-DD), got {value[k]!r}")
                dates[k] = d
        f = cls(
            path=_tuple(value.get("path")),
            exclude_path=_tuple(value.get("exclude_path")),
            doc_type=tuple(t.lower() for t in _tuple(value.get("type", value.get("doc_type")))),
            tags=tuple(sorted(parse_tags(list(_tuple(value.get("tags")))))),
            tags_any=tuple(sorted(parse_tags(list(_tuple(value.get("tags_any")))))),
            meta=tuple(sorted((str(k), json.dumps(v, sort_keys=True)) for k, v in meta.items())),
            **dates,
        )
        return None if f.is_empty() else f

    def is_empty(self) -> bool:
        return not any((self.path, self.exclude_path, self.doc_type, self.tags, self.tags_any,
                        self.date_from, self.date_to, self.as_of, self.meta))

    def to_dict(self) -> dict:
        d: dict[str, Any] = {}
        for k, v in (("path", self.path), ("exclude_path", self.exclude_path), ("type", self.doc_type),
                     ("tags", self.tags), ("tags_any", self.tags_any)):
            if v:
                d[k] = list(v)
        for k in ("date_from", "date_to", "as_of"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.meta:
            d["meta"] = {k: json.loads(v) for k, v in self.meta}
        return d

    def key(self) -> str:
        """Stable string for cache keys."""
        return json.dumps(self.to_dict(), sort_keys=True)

    # -- matching -------------------------------------------------------
    def matches(
        self,
        doc_path: str,
        frontmatter: Mapping[str, Any],
        superseded_by: Callable[[str], Iterable[Mapping[str, Any]]] = lambda p: (),
    ) -> bool:
        """True if the document (*doc_path*, *frontmatter*) passes every condition.

        *superseded_by(doc_path)* yields the frontmatter of documents whose
        ``supersedes`` names *doc_path* (only needed for ``as_of``).
        """
        p = doc_path.replace("\\", "/")
        fm = frontmatter or {}
        if self.path and not any(fnmatchcase(p, g) for g in self.path):
            return False
        if self.exclude_path and any(fnmatchcase(p, g) for g in self.exclude_path):
            return False
        if self.doc_type and str(fm.get("type", "")).strip().lower() not in self.doc_type:
            return False
        if self.tags or self.tags_any:
            have = parse_tags(fm.get("tags"))
            if self.tags and not set(self.tags) <= have:
                return False
            if self.tags_any and not (set(self.tags_any) & have):
                return False
        if self.date_from or self.date_to:
            d = _date(fm.get("date"))
            if d is None:
                return False
            if self.date_from and d < self.date_from:
                return False
            if self.date_to and d[: len(self.date_to)] > self.date_to:
                return False
        if self.as_of:
            start = _date(fm.get("effective")) or _date(fm.get("date"))
            if start is not None and start > self.as_of:
                return False
            for other in superseded_by(p):
                o_start = _date(other.get("effective")) or _date(other.get("date"))
                if o_start is not None and o_start <= self.as_of:
                    return False
        for name, cond in self.meta:
            if not _meta_match(fm.get(name), json.loads(cond)):
                return False
        return True
