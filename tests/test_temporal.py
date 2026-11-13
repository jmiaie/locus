"""Optional date-aware (temporal) ranking: parser + engine integration (stdlib only)."""
from datetime import date, datetime

import pytest

from locus.core import LocusEngine
from locus.retrieval.temporal import (
    MAX_WINDOW_DAYS,
    parse_doc_date,
    parse_relative_windows,
    time_window_boost,
    to_date,
)
from locus.retrieval.bm25 import ScoredChunk

REF = date(2023, 5, 30)  # a Tuesday


def W(q, ref=REF):
    return parse_relative_windows(q, ref)


def test_n_units_ago():
    assert W("what did I buy 3 days ago?") == [(date(2023, 5, 26), date(2023, 5, 28))]
    assert W("Two weeks ago I met someone") == [(date(2023, 5, 10), date(2023, 5, 22))]
    assert W("a week ago") == [(date(2023, 5, 20), date(2023, 5, 26))]
    assert W("a couple of days ago") == [(date(2023, 5, 27), date(2023, 5, 29))]
    assert W("a few weeks ago") == [(date(2023, 5, 3), date(2023, 5, 15))]
    # month: centre ref - 60, half-width min(10*2, 20) = 20
    assert W("two months ago") == [(date(2023, 3, 11), date(2023, 4, 20))]


def test_last_weekday_weekend_yesterday():
    assert W("last tuesday") == [(date(2023, 5, 22), date(2023, 5, 24))]  # same weekday -> a week back
    assert W("last Friday") == [(date(2023, 5, 25), date(2023, 5, 27))]
    assert W("yesterday") == [(date(2023, 5, 28), date(2023, 5, 30))]
    assert W("last weekend") == [(date(2023, 5, 26), date(2023, 5, 29))]  # Saturday 27th, -1..+2
    sat = date(2023, 6, 3)
    assert W("last weekend", sat) == [(date(2023, 5, 26), date(2023, 5, 29))]


def test_last_week_and_month():
    assert W("last week") == [(date(2023, 5, 19), date(2023, 5, 27))]
    assert W("last month") == [(date(2023, 4, 15), date(2023, 5, 15))]


@pytest.mark.parametrize("q", [
    "in the past few months", "over the last two months", "the past month", "past 2 weeks",
    "last 3 months", "two years ago", "over the past few weeks", "how many hotels have I stayed at?",
    "", "ago ago last", "9999999999 years ago", "last weekday",
])
def test_negatives(q):
    assert W(q) == []


def test_never_raises_and_cap():
    assert parse_relative_windows(None, REF) == []  # type: ignore[arg-type]
    assert all((b - a).days <= MAX_WINDOW_DAYS for a, b in W("3 months ago, last month, 2 weeks ago"))


def test_date_parsing():
    assert parse_doc_date("2023/05/20 (Sat) 02:21") == date(2023, 5, 20)
    assert parse_doc_date("2023-05-20") == date(2023, 5, 20)
    assert parse_doc_date(datetime(2023, 5, 20, 3)) == date(2023, 5, 20)
    assert parse_doc_date("May 20") is None and parse_doc_date(None) is None and parse_doc_date("2023-02-30") is None
    assert to_date("2023/05/20") == date(2023, 5, 20)
    with pytest.raises(ValueError):
        to_date("yesterday")


def _c(doc, i=0):
    return ScoredChunk(f"{doc}#{i}", doc, 1.0, "x", "bm25")


def test_boost_math():
    pool = [_c("a"), _c("b"), _c("a", 1), _c("c"), _c("d")]
    dates = {"a": None, "b": date(2023, 1, 1), "c": date(2023, 5, 20), "d": date(2023, 5, 21)}
    out = time_window_boost(pool, [(date(2023, 5, 19), date(2023, 5, 22))], dates, limit=10)
    # base a,b,c,d; in-window c(0), d(1): c = 1/63 + 1/61, d = 1/64 + 1/62 > a = 1/61
    assert [x.doc_path for x in out] == ["c", "d", "a", "a", "b"]
    assert all(x.provenance.endswith("+time") for x in out)
    assert time_window_boost(pool, [(date(2020, 1, 1), date(2020, 1, 2))], dates, 3) == pool[:3]


DOCS = {
    "old-trip.md": ("2023-01-02", "We went hiking in the mountains and the trip was great."),
    "mid-trip.md": ("2023-05-16", "Short note: hiking trip planned, booked a cabin."),
    "chat.md": (None, "Hiking trip ideas and the mountains, hiking hiking trip."),
    "other.md": ("2023-05-15", "Groceries and cooking dinner."),
}


@pytest.fixture()
def engine(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for name, (dt, body) in DOCS.items():
        fm = f"---\ndate: {dt}\n---\n" if dt else ""
        (d / name).write_text(fm + body + "\n")
    e = LocusEngine(store_path=tmp_path / ".locus")
    e.index(d)
    return e


def _docs(res):
    return list(dict.fromkeys(c.doc_path.split("/")[-1] for c in res))


def test_engine_default_off_identical(engine):
    q = "what hiking trip did I take two weeks ago?"
    base = engine.retrieve(q, limit=10, use_cache=False)
    assert [c.chunk_id for c in engine.retrieve(q, limit=10, use_cache=False, temporal=False)] == [c.chunk_id for c in base]
    assert _docs(base)[0] != "mid-trip.md"


def test_engine_boosts_dated_doc_in_window(engine):
    q = "what hiking trip did I take two weeks ago?"
    res = engine.retrieve(q, limit=10, use_cache=False, temporal=True, reference_date="2023-05-30")
    assert _docs(res)[0] == "mid-trip.md"
    assert res[0].provenance.endswith("+time")


def test_engine_no_phrase_or_no_match_identical(engine):
    q = "hiking trip mountains"
    base = [c.chunk_id for c in engine.retrieve(q, limit=10, use_cache=False)]
    assert [c.chunk_id for c in engine.retrieve(q, limit=10, use_cache=False, temporal=True, reference_date=REF)] == base
    q2 = "hiking trip 2 days ago"  # window has no dated candidate
    base2 = [c.chunk_id for c in engine.retrieve(q2, limit=10, use_cache=False)]
    assert [c.chunk_id for c in engine.retrieve(q2, limit=10, use_cache=False, temporal=True, reference_date=REF)] == base2


def test_undated_never_boosted(engine):
    res = engine.retrieve("hiking trip ideas yesterday", limit=10, use_cache=False, temporal=True,
                          reference_date="2023-01-03")
    base = _docs(engine.retrieve("hiking trip ideas yesterday", limit=10, use_cache=False))
    got = _docs(res)
    assert got[0] == "old-trip.md"  # the only doc dated in the window
    assert [d for d in got if d != "old-trip.md"] == [d for d in base if d != "old-trip.md"]  # others keep base order


def test_engine_default_and_cache(tmp_path, engine):
    q = "what hiking trip did I take two weeks ago?"
    a = engine.retrieve(q, limit=10, temporal=True, reference_date="2023-05-30")
    b = engine.retrieve(q, limit=10)
    assert _docs(a)[0] == "mid-trip.md" and _docs(b)[0] != "mid-trip.md"  # cached separately
    on = LocusEngine(store_path=engine.store_path, temporal=True)
    assert _docs(on.retrieve(q, limit=10, reference_date="2023-05-30", use_cache=False))[0] == "mid-trip.md"
    with pytest.raises(ValueError):
        LocusEngine(store_path=engine.store_path, temporal_depth=0)
