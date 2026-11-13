"""
Optional date-aware ranking: boost documents dated inside a time window the
query refers to ("two weeks ago", "last Friday", "yesterday", ...).

Off by default; ``LocusEngine(temporal=True)`` or ``retrieve(..., temporal=True)``.
Pure standard library.

Parser — :func:`parse_relative_windows` recognises point-in-time references
only (case-insensitive, word boundaries; ``n`` is digits or one of
a/an/one..ten, "(a) couple of" = 2, "(a) few" = 3):

1. ``N day|week|month|year(s) ago`` — centre ``ref - n*unit`` (unit in days:
   day 1, week 7, month 30, year 365), half-width ``min(tol * (n, or 1 for
   days), 2 * tol)`` with tol day 1, week 3, month 10, year 45.
2. ``last monday`` .. ``last sunday`` — the most recent such day strictly
   before *ref*, ±1 day.
3. ``yesterday`` — ``ref - 2`` .. ``ref``.
4. ``last weekend`` — the most recent Saturday before *ref*, -1 .. +2 days.
5. ``last week`` — ``ref - 11`` .. ``ref - 3``.
6. ``last month`` — ``ref - 45`` .. ``ref - 15``.

Windows wider than ``MAX_WINDOW_DAYS`` are dropped (so every "N years ago").
Duration ranges ("past 2 weeks", "last 3 months", "past few months", "past
month") are deliberately *not* recognised.  This narrowing, and the 45-day
cap, were decided post hoc: a prototype that also matched duration ranges
showed small losses on LongMemEval multi-session / single-session-user
questions.

Ranking — :func:`time_window_boost`: the base ranking's documents are fused
by Reciprocal Rank Fusion (k=60, equal weights) with the sub-list of those
dated (frontmatter ``date``) inside a window.  Undated documents are never
boosted.  When nothing parses or no candidate is in a window, the base
ranking is returned unchanged.  There is deliberately no general recency
prior (one measured strongly harmful).
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime, timedelta

#: Windows wider than this (end - start, in days) are dropped.
MAX_WINDOW_DAYS = 45
#: RRF constant of the time-window fusion.
RRF_K = 60

_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a couple of": 2, "couple of": 2, "a few": 3, "few": 3,
}
_UNIT = {"day": 1, "week": 7, "month": 30, "year": 365}
_TOL = {"day": 1, "week": 3, "month": 10, "year": 45}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_AGO = re.compile(
    r"\b(\d+|a couple of|couple of|a few|few|an|a|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s+(day|week|month|year)s?\s+ago\b", re.I)
_LAST_WEEKDAY = re.compile(r"\blast\s+(" + "|".join(_WEEKDAYS) + r")\b", re.I)
_YESTERDAY = re.compile(r"\byesterday\b", re.I)
_LAST_WEEKEND = re.compile(r"\blast\s+weekend\b", re.I)
_LAST_WEEK = re.compile(r"\blast\s+week\b", re.I)
_LAST_MONTH = re.compile(r"\blast\s+month\b", re.I)
_DATE = re.compile(r"(\d{4})([-/])(\d{1,2})\2(\d{1,2})")


def _days(n: int) -> timedelta:
    return timedelta(days=n)


def parse_relative_windows(query: str, reference: date) -> list[tuple[date, date]]:
    """Inclusive ``(start, end)`` date windows referred to by *query*, relative to *reference*.

    Returns ``[]`` when nothing matches; never raises.
    """
    try:
        if not isinstance(query, str) or not query:
            return []
        ref = to_date(reference)
        out: list[tuple[date, date]] = []

        def add(c: date, lo: int, hi: int) -> None:
            try:
                out.append((c - _days(lo), c + _days(hi)))
            except (OverflowError, ValueError):
                pass

        for m in _AGO.finditer(query):
            word, unit = m.group(1).lower(), m.group(2).lower()
            n = int(word) if word.isdigit() else _NUM[word]
            t = min(_TOL[unit] * (n if unit != "day" else 1), 2 * _TOL[unit])
            try:
                c = ref - _days(n * _UNIT[unit])
            except (OverflowError, ValueError):
                continue
            add(c, t, t)
        for m in _LAST_WEEKDAY.finditer(query):
            delta = (ref.weekday() - _WEEKDAYS.index(m.group(1).lower())) % 7 or 7
            add(ref - _days(delta), 1, 1)
        if _YESTERDAY.search(query):
            add(ref, 2, 0)
        if _LAST_WEEKEND.search(query):
            delta = (ref.weekday() - 5) % 7 or 7
            add(ref - _days(delta), 1, 2)
        if _LAST_WEEK.search(query):
            add(ref, 11, -3)
        if _LAST_MONTH.search(query):
            add(ref, 45, -15)
        return [(a, b) for a, b in out if (b - a).days <= MAX_WINDOW_DAYS]
    except Exception:  # pragma: no cover - defensive: the parser must never raise
        return []


def to_date(value) -> date:
    """``date`` from a date, datetime or ``"YYYY-MM-DD"`` / ``"YYYY/MM/DD"`` string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        d = parse_doc_date(value)
        if d is not None:
            return d
    raise ValueError(f"reference_date must be a date or 'YYYY-MM-DD' / 'YYYY/MM/DD', got {value!r}")


def parse_doc_date(value) -> date | None:
    """Leading ``YYYY-MM-DD`` or ``YYYY/MM/DD`` of a frontmatter value, else None."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    m = _DATE.match(value.strip().strip("\"'"))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(3)), int(m.group(4)))
    except ValueError:
        return None


def rrf_doc_order(lists: list[list[str]], k: int = RRF_K) -> tuple[list[str], dict[str, float]]:
    """Fuse ranked doc lists; ties by position in ``lists[0]`` (which must contain every doc)."""
    score: dict[str, float] = {}
    for docs in lists:
        for rank, d in enumerate(docs):
            score[d] = score.get(d, 0.0) + 1.0 / (k + rank + 1)
    pos = {d: i for i, d in enumerate(lists[0])}
    return sorted(pos, key=lambda d: (-score[d], pos[d])), score


def reorder_by_docs(pool: list, order: list[str], score: dict[str, float], tag: str,
                    limit: int | None) -> list:
    """Pool chunks grouped by *order* (stable within a doc), scored by their doc's fused score."""
    by_doc: dict[str, list] = {}
    for c in pool:
        by_doc.setdefault(c.doc_path, []).append(c)
    out = [replace(c, score=score[d], provenance=c.provenance + tag) for d in order for c in by_doc[d]]
    return out[:limit] if limit is not None else out


def time_window_boost(pool: list, windows: list[tuple[date, date]], doc_dates: dict,
                      limit: int) -> list:
    """Boost pool chunks whose document date (``doc_dates[doc_path]``) lies in any window."""
    bm = list(dict.fromkeys(c.doc_path for c in pool))
    inw = [d for d in bm if doc_dates.get(d) is not None
           and any(a <= doc_dates[d] <= b for a, b in windows)]
    if not windows or not inw:
        return pool[:limit]
    order, score = rrf_doc_order([bm, inw])
    return reorder_by_docs(pool, order, score, "+time", limit)
