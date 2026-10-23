"""Metadata filters: API, CLI and MCP (correctness; uses the synthetic construction docs)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from locus.core import LocusEngine
from locus.mcp.server import _call_tool
from locus.retrieval.filters import MetadataFilter, parse_tags

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "examples" / "construction" / "docs"


@pytest.fixture(scope="module")
def eng(tmp_path_factory):
    e = LocusEngine(store_path=tmp_path_factory.mktemp("cx") / ".locus")
    e.index(DOCS)
    return e


def paths(chunks):
    return [c.doc_path for c in chunks]


def test_parse_tags_forms():
    assert parse_tags("concrete, MRI, structural") == {"concrete", "mri", "structural"}
    assert parse_tags("[a, b]") == {"a", "b"}
    assert parse_tags("#ops #infra") == {"ops", "infra"}
    assert parse_tags(["X", "y"]) == {"x", "y"}
    assert parse_tags(None) == set()


def test_no_filter_is_identity(eng):
    q = "concrete compressive strength"
    assert paths(eng.retrieve(q, limit=8, use_cache=False)) == paths(
        eng.retrieve(q, limit=8, use_cache=False, filters={}))
    assert MetadataFilter.from_value({}) is None


def test_path_glob_and_exclude(eng):
    r = eng.retrieve("concrete slab", limit=10, use_cache=False, filters={"path": "specs/*"})
    assert r and all(p.startswith("specs/") for p in paths(r))
    r = eng.retrieve("concrete slab", limit=10, use_cache=False,
                     filters={"path": ["specs/*", "rfis/*"], "exclude_path": "*rev0*"})
    assert r and all(p.split("/")[0] in ("specs", "rfis") and "rev0" not in p for p in paths(r))


def test_type_and_tags(eng):
    r = eng.retrieve("vibration", limit=10, use_cache=False, filters={"type": "RFI"})
    assert r and all(p.startswith("rfis/") for p in paths(r))
    r = eng.retrieve("concrete", limit=10, use_cache=False, filters={"tags": ["concrete", "MRI"]})
    fms = eng.corpus.doc_frontmatter(paths(r))
    assert r and all({"concrete", "mri"} <= parse_tags(fm.get("tags")) for fm in fms.values())
    r = eng.retrieve("upgrade", limit=10, use_cache=False, filters={"tags_any": ["roofing", "sprinkler"]})
    fms = eng.corpus.doc_frontmatter(paths(r))
    assert r and all(parse_tags(fm.get("tags")) & {"roofing", "sprinkler"} for fm in fms.values())


def test_date_range(eng):
    r = eng.retrieve("meeting minutes", limit=20, use_cache=False,
                     filters={"date_from": "2026-06-01", "date_to": "2026-06-30"})
    fms = eng.corpus.doc_frontmatter(paths(r))
    assert r and all("2026-06-01" <= fm["date"][:10] <= "2026-06-30" for fm in fms.values())


def test_as_of_respects_supersedes(eng):
    q = "Level 2 slab concrete compressive strength psi"
    early = paths(eng.retrieve(q, limit=20, use_cache=False, filters={"as_of": "2026-02-15", "type": "spec"}))
    late = paths(eng.retrieve(q, limit=20, use_cache=False, filters={"as_of": "2026-04-01", "type": "spec"}))
    rev0, rev1 = "specs/03-30-00-cast-in-place-concrete-rev0.md", "specs/03-30-00-cast-in-place-concrete-rev1.md"
    assert rev0 in early and rev1 not in early
    assert rev1 in late and rev0 not in late


def test_meta_operators(eng):
    r = eng.retrieve("roof", limit=20, use_cache=False, filters={"meta": {"status": "closed"}})
    fms = eng.corpus.doc_frontmatter(paths(r))
    assert r and all(fm["status"].lower() == "closed" for fm in fms.values())
    r = eng.retrieve("concrete roofing hardware permit", limit=20, use_cache=False, filters={"meta": {"revision": {"gte": 1}}})
    fms = eng.corpus.doc_frontmatter(paths(r))
    assert r and all(float(fm["revision"]) >= 1 for fm in fms.values())
    r = eng.retrieve("concrete roofing", limit=20, use_cache=False,
                     filters={"meta": {"spec_section": {"in": ["03 30 00", "07 54 23"]}, "supersedes": {"exists": False}}})
    fms = eng.corpus.doc_frontmatter(paths(r))
    assert r and all(fm["spec_section"] in ("03 30 00", "07 54 23") and not fm.get("supersedes") for fm in fms.values())
    r = eng.retrieve("change order", limit=20, use_cache=False, filters={"meta": {"title": {"contains": "roof"}}})
    assert r and all("roof" in eng.corpus.doc_frontmatter([p])[p]["title"].lower() for p in paths(r))


def test_filter_never_reorders_and_deepens(eng):
    q = "concrete"
    full = paths(eng.retrieve(q, limit=500, use_cache=False))
    f = {"path": "permits/*"}
    r = eng.retrieve(q, limit=1, use_cache=False, filters=f)
    want = [p for p in full if p.startswith("permits/")]
    assert full.index(want[0]) >= 5
    assert r and paths(r) == want[:1]  # found even though it ranks far below the first page
    r3 = eng.retrieve(q, limit=3, use_cache=False, filters={"path": ["specs/*", "rfis/*"]})
    assert paths(r3) == [p for p in full if p.split("/")[0] in ("specs", "rfis")][:3]


def test_filter_with_no_matches(eng):
    assert eng.retrieve("concrete", limit=5, use_cache=False, filters={"path": "nope/**"}) == []


def test_hybrid_mode_filters(eng):
    r = eng.retrieve("roof membrane upgrade", limit=5, mode="hybrid", use_cache=False, filters={"type": "change_order"})
    fms = eng.corpus.doc_frontmatter(paths(r))
    assert r and all(fm.get("type") == "change_order" for fm in fms.values())


def test_cache_keys_include_filters(eng):
    a = paths(eng.retrieve("concrete", limit=3, filters={"type": "spec"}))
    b = paths(eng.retrieve("concrete", limit=3, filters={"type": "rfi"}))
    assert a != b and all(p.startswith("specs/") for p in a) and all(p.startswith("rfis/") for p in b)


@pytest.mark.parametrize("bad,msg", [
    ({"colour": "red"}, "Unknown filter key"),
    ({"meta": {"x": {"approx": 1}}}, "Unknown meta filter operator"),
    ({"date_from": "last week"}, "ISO date"),
    ({"meta": ["x"]}, "filters.meta"),
])
def test_invalid_filters(eng, bad, msg):
    with pytest.raises(ValueError, match=msg):
        eng.retrieve("concrete", use_cache=False, filters=bad)


def test_filter_object_roundtrip():
    f = MetadataFilter.from_value({"path": "a/*", "type": ["Spec"], "tags": "x, Y", "as_of": "2026-01-02",
                                   "meta": {"rev": {"gte": 1}}})
    assert MetadataFilter.from_value(f.to_dict()) == f
    assert f.to_dict()["type"] == ["spec"] and f.tags == ("x", "y")


def test_mcp_retrieve_filters(eng):
    out = _call_tool("locus_retrieve", {"query": "vibration", "limit": 5, "track": False,
                                        "filters": {"type": "rfi"}}, engine=eng)
    assert out["results"] and all(r["doc_path"].startswith("rfis/") for r in out["results"])
    bad = _call_tool("locus_retrieve", {"query": "x", "track": False, "filters": {"bogus": 1}}, engine=eng)
    assert "error" in bad


def test_mcp_schema_has_filters():
    from locus.mcp.tools import TOOLS
    props = TOOLS["locus_retrieve"]["input_schema"]["properties"]
    assert "filters" in props and "as_of" in props["filters"]["properties"]


def test_cli_filters(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    store = tmp_path / ".locus"
    subprocess.run([sys.executable, "-m", "locus", "--store", str(store), "index", str(DOCS)],
                   check=True, capture_output=True, env=env)
    out = subprocess.run(
        [sys.executable, "-m", "locus", "--store", str(store), "retrieve", "slab concrete",
         "--limit", "5", "--raw", "--type", "spec", "--in-force", "2026-02-15", "--meta", "spec_section=03 30 00"],
        check=True, capture_output=True, text=True, env=env).stdout
    got = [r["doc_path"] for r in json.loads(out)]
    assert got and set(got) == {"specs/03-30-00-cast-in-place-concrete-rev0.md"}
    out = subprocess.run(
        [sys.executable, "-m", "locus", "--store", str(store), "retrieve", "roof", "--raw",
         "--filters", json.dumps({"path": "change-orders/*"})],
        check=True, capture_output=True, text=True, env=env).stdout
    assert all(r["doc_path"].startswith("change-orders/") for r in json.loads(out))
