#!/usr/bin/env python3
"""
Locus end-to-end MCP sales / consulting demo.

Runs the same tool path Claude Desktop / Cursor would hit when Locus is
registered as an MCP server: sample notes → index → retrieve → explain →
KG entity query → doctor. No MCP client binary required — we call the
server's tool dispatcher directly (stdlib + locus only).

Usage (from repo root, after install):
    pip install -e .
    python examples/mcp_sales_demo.py

    # Keep the temp store + notes for inspection:
    python examples/mcp_sales_demo.py --keep

Deps: Python 3.11+, locus-rag (zero runtime deps). Optional: none.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Sample consulting corpus — interconnected notes with entities, wikilinks,
# and frontmatter so BM25 + KG + link/structural signals all fire.
# ---------------------------------------------------------------------------

SAMPLE_NOTES: dict[str, str] = {
    "auth-jwt.md": """\
---
date: 2025-01-15
type: guide
tags: security, authentication, jwt
---

# JWT Authentication

Alice leads the Auth team. JWT tokens are validated on every request.
The token expiry is 24 hours. Alice works_at Acme.

See also [[oauth-partners]] and [[incident-auth-outage]].
""",
    "oauth-partners.md": """\
---
date: 2025-01-20
type: guide
tags: security, oauth, partners
---

# OAuth for Partners

Bob originally built the legacy token system. Alice replaced that with OAuth 2.0
for third-party partner integrations. Bob works_at Acme. OAuth depends_on JWT.

Partner apps must use the Auth team's JWT validation layer — see [[auth-jwt]].
""",
    "incident-auth-outage.md": """\
---
date: 2025-02-10
type: runbook
tags: ops, incident, authentication
---

# Auth Outage Runbook (2025-02-10)

On 2025-02-10 the JWT validator returned elevated 401 rates. Alice led the
incident response. Escalate to the Auth team when token failures spike.

Related: [[auth-jwt]].
""",
    "deploy-checklist.md": """\
---
date: 2025-02-01
type: runbook
tags: operations, deployment
---

# Deployment Checklist

1. Confirm JWT and OAuth paths are healthy (see [[auth-jwt]], [[oauth-partners]]).
2. Run the test suite.
3. Tag the release.

Carol works_at Acme and owns the deploy pipeline. Deploy depends_on Auth.
""",
}

DEMO_QUERIES: list[tuple[str, str]] = [
    ("who leads the Auth team?", "Entity / people query — expect KG + BM25"),
    ("JWT authentication validation", "Keyword query — expect BM25 provenance"),
    ("auth outage runbook February", "Temporal / ops query — expect structural/recency"),
]


def _banner(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def _pp(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


def write_sample_notes(docs_dir: Path) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, body in SAMPLE_NOTES.items():
        path = docs_dir / name
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


def run_demo(keep: bool = False) -> int:
    try:
        from locus.mcp.server import _call_tool
        from locus.core import __version__
    except ImportError:
        print(
            "ERROR: locus is not installed.\n"
            "  pip install -e .          # from a clone\n"
            "  pip install locus-rag     # from PyPI",
            file=sys.stderr,
        )
        return 1

    root = Path(tempfile.mkdtemp(prefix="locus-mcp-demo-"))
    docs_dir = root / "notes"
    store_path = str(root / ".locus")

    _banner(f"Locus MCP Sales Demo  (locus {__version__})")
    print(f"Working directory : {root}")
    print(f"Store path        : {store_path}")
    print("Tool path         : locus.mcp.server._call_tool  (same as MCP)")

    # --- 1. Sample notes ---------------------------------------------------
    _banner("1. Ingest sample consulting notes")
    paths = write_sample_notes(docs_dir)
    for p in paths:
        print(f"  wrote {p.name}  ({p.stat().st_size} bytes)")

    # --- 2. Session open + index (MCP tools) -------------------------------
    _banner("2. MCP: locus_session_start + locus_index")
    session = _call_tool("locus_session_start", {"store_path": store_path})
    print("locus_session_start →")
    _pp(session)

    indexed = _call_tool(
        "locus_index",
        {"path": str(docs_dir), "pattern": "**/*.md", "store_path": store_path},
    )
    print("locus_index →")
    _pp(indexed)

    # Seed a few explicit temporal facts so the KG story is demo-reliable
    # even if prose extraction varies across versions.
    _banner("3. MCP: locus_add_fact  (explicit temporal KG)")
    seeded = [
        ("Alice", "leads", "Auth team", "2024-06-01", None, "auth-jwt.md"),
        ("Alice", "works_at", "Acme", "2023-01-01", None, "auth-jwt.md"),
        ("Bob", "works_at", "Acme", "2022-01-01", None, "oauth-partners.md"),
        ("OAuth", "depends_on", "JWT", "2025-01-20", None, "oauth-partners.md"),
        ("Alice", "led", "auth outage response", "2025-02-10", "2025-02-11", "incident-auth-outage.md"),
    ]
    for s, p, o, vf, vt, src in seeded:
        args = {
            "subject": s,
            "predicate": p,
            "object": o,
            "valid_from": vf,
            "source": src,
            "store_path": store_path,
        }
        if vt:
            args["valid_to"] = vt
        result = _call_tool("locus_add_fact", args)
        print(f"  {s} --{p}--> {o}  → {result}")

    stats = _call_tool("locus_kg_stats", {"store_path": store_path})
    print("locus_kg_stats →")
    _pp(stats)

    # --- 4. Retrieve + explain (the deck money shot) -----------------------
    _banner("4. MCP: locus_retrieve + locus_explain  (retrieval path)")
    for query, why in DEMO_QUERIES:
        print(f"\n--- Query: {query!r}")
        print(f"    ({why})")
        retrieved = _call_tool(
            "locus_retrieve",
            {
                "query": query,
                "limit": 3,
                "full_content": False,
                "store_path": store_path,
            },
        )
        if "error" in retrieved:
            print("  ERROR:", retrieved["error"])
            continue
        print(f"  intent={retrieved.get('intent')}")
        for i, hit in enumerate(retrieved.get("results", []), 1):
            print(
                f"  [{i}] {hit['doc_path']}  score={hit['score']}  "
                f"via={hit['provenance']}  entities={hit.get('entities')}"
            )
            print(f"      {hit['content'][:140].replace(chr(10), ' ')}…")

        if retrieved.get("results"):
            top = retrieved["results"][0]
            explained = _call_tool(
                "locus_explain",
                {
                    "chunk_id": top["chunk_id"],
                    "query": query,
                    "store_path": store_path,
                },
            )
            narrative = explained.get("narrative") or explained
            print("  explain →", narrative if isinstance(narrative, str) else json.dumps(narrative)[:300])

    # --- 5. Entity + doctor ------------------------------------------------
    _banner("5. MCP: locus_query_entity + locus_doctor")
    entity = _call_tool(
        "locus_query_entity",
        {"entity": "Alice", "store_path": store_path},
    )
    print("locus_query_entity(Alice) →")
    _pp(entity)

    doctor = _call_tool("locus_doctor", {"store_path": store_path})
    print("locus_doctor →")
    _pp(doctor)

    wrap = _call_tool("locus_wrap_up", {"store_path": store_path})
    print("locus_wrap_up →")
    _pp(wrap)

    # --- Closing pitch -----------------------------------------------------
    _banner("What this demo showed")
    print(
        """
  • Zero GPU / zero embedding API — BM25 + temporal KG + link signals
  • Every hit carries a provenance tag (bm25 / kg / link / structural / …)
  • locus_explain returns a plain-English "why" (impossible with cosine RAG)
  • Same tool names an MCP client would call in Claude Desktop / Cursor

  Next steps for a live deck:
    claude mcp add locus -- python -m locus.mcp.server --store /path/to/.locus
    # then ask: "What does the vault say about who leads Auth?"

  Companion stack:
    Locus        = explainable retrieval   (this repo)
    OMPA         = vault memory + temporal KG
    CognitionOS  = compliance product on top of both
""".rstrip()
    )

    if keep:
        print(f"\nKept demo artifacts at: {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
        print("\n(Cleaned temp store. Re-run with --keep to inspect files.)")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete the temp notes + .locus store after the run",
    )
    args = parser.parse_args()
    return run_demo(keep=args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
