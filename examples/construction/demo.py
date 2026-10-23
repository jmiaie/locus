#!/usr/bin/env python3
"""
Locus construction / real-estate demo (SYNTHETIC documents).

    python examples/construction/demo.py            # walkthrough + evaluation
    python examples/construction/demo.py --eval     # evaluation only (JSON summary to stdout)
    python examples/construction/demo.py --store /tmp/cx-demo   # keep the index

Everything runs locally with the standard library and Locus; no network, no LLM.
The documents in docs/ are fictional and small (24 files); treat the metrics as a
smoke test of the workflow, not as evidence about real project archives.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs"
sys.path.insert(0, str(HERE.parent.parent))  # run from a source checkout without installing

from locus import LocusEngine  # noqa: E402
from locus.memory.chunker import _extract_frontmatter  # noqa: E402

TOP_K = 5


# ----------------------------------------------------------------------------
# Document metadata and the point-in-time view
# ----------------------------------------------------------------------------

def load_meta(docs: Path = DOCS) -> dict[str, dict]:
    """Frontmatter of every document, keyed by path relative to docs/."""
    meta = {}
    for p in sorted(docs.rglob("*.md")):
        fm, _ = _extract_frontmatter(p.read_text(encoding="utf-8"))
        meta[p.relative_to(docs).as_posix()] = fm
    return meta


def rel(doc_path: str) -> str:
    """Locus doc_path -> path relative to docs/ (Locus stores it relative to the indexed folder)."""
    return doc_path.replace("\\", "/").lstrip("./")


def in_force(path: str, as_of: str, meta: dict[str, dict]) -> bool:
    """True if *path* existed on *as_of* and had not been superseded by then.

    A document exists from its ``effective`` date (or ``date``).  A revision is
    hidden once another document that ``supersedes`` it is effective.  ISO
    dates compare correctly as strings.
    """
    fm = meta.get(path, {})
    start = fm.get("effective") or fm.get("date") or "0000-00-00"
    if start > as_of:
        return False
    for other, ofm in meta.items():
        if ofm.get("supersedes") == path and (ofm.get("effective") or ofm.get("date", "9999")) <= as_of:
            return False
    return True


def ranked_docs(engine: LocusEngine, query: str, meta: dict, as_of: str | None = None, k: int = TOP_K):
    """Top-*k* distinct documents (with their best chunk) for *query*, optionally as of a date."""
    seen: dict[str, object] = {}
    for c in engine.retrieve(query, limit=60, use_cache=False, mode="bm25"):
        path = rel(c.doc_path)
        if path in seen:
            continue
        if as_of and not in_force(path, as_of, meta):
            continue
        seen[path] = c
        if len(seen) == k:
            break
    return list(seen.items())


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def evaluate(engine: LocusEngine, meta: dict, questions: list[dict]) -> dict:
    rows = []
    for q in questions:
        exp = set(q["expected_docs"])
        as_of = q.get("as_of")
        got = [p for p, _ in ranked_docs(engine, q["query"], meta, as_of=as_of)]
        ranks = [i + 1 for i, p in enumerate(got) if p in exp]
        row = {"id": q["id"], "kind": q["kind"], "query": q["query"], "as_of": as_of,
               "expected": sorted(exp), "top5": got,
               "hit@1": bool(ranks and ranks[0] == 1), "hit@5": bool(ranks),
               "all@5": exp <= set(got), "rr": 1.0 / ranks[0] if ranks else 0.0}
        if as_of:  # the same query without the point-in-time view, for contrast
            plain = [p for p, _ in ranked_docs(engine, q["query"], meta)]
            row["top1_without_as_of"] = plain[0] if plain else None
            row["hit@1_without_as_of"] = bool(plain and plain[0] in exp)
        rows.append(row)

    def agg(rs):
        n = len(rs)
        return {"n": n,
                "R@1": round(sum(r["hit@1"] for r in rs) / n, 3),
                "R@5": round(sum(r["hit@5"] for r in rs) / n, 3),
                "R_all@5": round(sum(r["all@5"] for r in rs) / n, 3),
                "MRR": round(sum(r["rr"] for r in rs) / n, 3)} if n else {"n": 0}

    plain_q = [r for r in rows if not r["as_of"]]
    temporal = [r for r in rows if r["as_of"]]
    summary = {"all": agg(rows), "lookup_and_rfi_trace": agg(plain_q), "point_in_time": agg(temporal)}
    if temporal:
        summary["point_in_time"]["R@1_without_as_of_view"] = round(
            sum(r["hit@1_without_as_of"] for r in temporal) / len(temporal), 3)
    return {"summary": summary, "questions": rows}


# ----------------------------------------------------------------------------
# Walkthrough
# ----------------------------------------------------------------------------

def show(engine, meta, title, query, as_of=None, k=3):
    print(f"\n### {title}\n> {query}" + (f"   (as of {as_of})" if as_of else ""))
    for i, (path, c) in enumerate(ranked_docs(engine, query, meta, as_of=as_of, k=k), 1):
        fm = meta.get(path, {})
        section = (c.content.splitlines() or [""])[0][:70]
        snippet = " ".join(c.content.split())[:220]
        print(f"  [{i}] {path}  ({fm.get('type', '?')}, {fm.get('date', '?')}"
              + (f", rev {fm['revision']}" if "revision" in fm else "") + f")  score={c.score:.2f}")
        print(f"      section: {section}")
        print(textwrap.indent(textwrap.fill(snippet, 96), "      "))


def walkthrough(engine: LocusEngine, meta: dict) -> None:
    print("=" * 78)
    print("Locus construction demo — SYNTHETIC project 'Harbor Point Medical Office Building'")
    print("=" * 78)
    show(engine, meta, "1. RFI trace: why did the Level 2 slab change, and what did it cost?",
         "Level 2 slab MRI vibration change order cost")
    show(engine, meta, "2. Point-in-time spec: what did 03 30 00 say before ASI-004?",
         "Level 2 elevated slab specified compressive strength", as_of="2026-02-15", k=2)
    show(engine, meta, "   ... and after ASI-004",
         "Level 2 elevated slab specified compressive strength", as_of="2026-04-01", k=2)
    show(engine, meta, "3. Permit compliance: can the GC work Saturdays in March vs. May?",
         "construction work hours Saturday", as_of="2026-03-01", k=1)
    show(engine, meta, "", "construction work hours Saturday", as_of="2026-05-01", k=2)
    show(engine, meta, "4. Lease exposure: what roof warranty does the tenant require?",
         "roof warranty landlord delivery condition lease")

    print("\n### 5. Knowledge-graph timeline (curated facts with source citations)")
    for subject, dates in (("Level 2 slab", ("2026-02-15", "2026-04-01")),
                           ("Roof membrane", ("2026-04-20", "2026-06-01")),
                           ("Stair 2 door HW-5", ("2026-05-30", "2026-06-10"))):
        for d in dates:
            facts = engine.query_entity(subject, as_of=d)["facts"]
            facts = [f for f in facts if f.get("source")]  # curated facts carry a source path
            for f in facts:
                print(f"  {subject} as of {d}: {f['object']}   [source: {f['source']}, "
                      f"valid {f['valid_from']} .. {f['valid_to'] or 'open'}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None, help="Index directory (default: a temporary directory)")
    ap.add_argument("--eval", action="store_true", help="Only run the evaluation and print JSON")
    ap.add_argument("--out", default=None, help="Write the full evaluation (per question) to this JSON file")
    args = ap.parse_args()

    tmp = None if args.store else tempfile.TemporaryDirectory()
    store = Path(args.store or tmp.name) / ".locus"
    engine = LocusEngine(store_path=store)
    engine.index(DOCS)
    meta = load_meta()
    for f in json.loads((HERE / "facts.json").read_text())["facts"]:
        engine.add_fact(f["subject"], f["predicate"], f["object"],
                        valid_from=f["valid_from"], valid_to=f["valid_to"], source=f["source"])

    questions = json.loads((HERE / "eval_questions.json").read_text())["questions"]
    result = evaluate(engine, meta, questions)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    if args.eval:
        print(json.dumps(result["summary"], indent=1))
    else:
        walkthrough(engine, meta)
        s = result["summary"]
        print("\n### 6. Evaluation (21 labelled questions, document level, top 5, BM25 default mode)")
        for name, v in s.items():
            line = f"  {name:22s} n={v['n']:2d}  R@1={v['R@1']:.3f}  R@5={v['R@5']:.3f}  " \
                   f"R_all@5={v['R_all@5']:.3f}  MRR={v['MRR']:.3f}"
            if "R@1_without_as_of_view" in v:
                line += f"  (R@1 without the as-of view: {v['R@1_without_as_of_view']:.3f})"
            print(line)
        misses = [r for r in result["questions"] if not r["hit@1"]]
        for r in misses:
            print(f"  not top-1: {r['id']} expected {r['expected']} got {r['top5'][:2]}")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
