"""
locus CLI — minimal command-line interface for Locus vectorless RAG.
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="locus",
        description="Locus — Vectorless RAG with MCP",
    )
    parser.add_argument("--store", default=".locus", metavar="PATH", help="Store path (default: .locus)")

    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    p_index = sub.add_parser("index", help="Index a file or directory")
    p_index.add_argument("path")
    p_index.add_argument("--pattern", default="**/*.md")

    p_ret = sub.add_parser("retrieve", help="Retrieve context for a query")
    p_ret.add_argument("query")
    p_ret.add_argument("--limit", type=int, default=5)
    p_ret.add_argument("--as-of", dest="as_of", default=None)
    p_ret.add_argument("--no-links", action="store_true")
    p_ret.add_argument("--raw", action="store_true", help="Output JSON instead of formatted text")

    sub.add_parser("status", help="Show corpus/KG/bulletin stats")
    sub.add_parser("session-start", help="Warm context open")
    sub.add_parser("wrap-up", help="Session close + bulletin tick")

    p_fact = sub.add_parser("add-fact", help="Add a KG triple")
    p_fact.add_argument("subject")
    p_fact.add_argument("predicate")
    p_fact.add_argument("object")
    p_fact.add_argument("--valid-from", dest="valid_from", default=None)
    p_fact.add_argument("--valid-to", dest="valid_to", default=None)
    p_fact.add_argument("--source", default=None)

    p_entity = sub.add_parser("query-entity", help="Query KG facts for an entity")
    p_entity.add_argument("entity")
    p_entity.add_argument("--as-of", dest="as_of", default=None)

    p_forget = sub.add_parser("forget", help="Remove a document from the corpus")
    p_forget.add_argument("doc_path")

    p_sync = sub.add_parser("sync", help="Full reindex from a directory")
    p_sync.add_argument("path")
    p_sync.add_argument("--pattern", default="**/*.md")

    p_mcp = sub.add_parser("mcp", help="Start the MCP server (JSON-RPC stdio)")
    p_mcp.add_argument("--store", default=".locus", dest="mcp_store")

    args = parser.parse_args()

    if args.cmd == "mcp":
        from .mcp.server import main as mcp_main
        mcp_main(store_path=getattr(args, "mcp_store", ".locus"))
        return

    if args.cmd is None:
        parser.print_help()
        return

    from .core import LocusEngine
    engine = LocusEngine(store_path=args.store)

    if args.cmd == "index":
        result = engine.index(args.path, pattern=args.pattern)
        print(json.dumps(result, indent=2))

    elif args.cmd == "retrieve":
        chunks = engine.retrieve(
            query=args.query,
            limit=args.limit,
            as_of=args.as_of,
            use_links=not args.no_links,
        )
        if args.raw:
            print(json.dumps([
                {"chunk_id": c.chunk_id, "doc_path": c.doc_path,
                 "score": round(c.score, 4), "provenance": c.provenance,
                 "content": c.content[:400]}
                for c in chunks
            ], indent=2))
        else:
            if not chunks:
                print("No results.")
            for i, c in enumerate(chunks, 1):
                print(f"\n[{i}] {c.doc_path}  score={c.score:.4f}  via={c.provenance}")
                print("-" * 60)
                print(c.content[:400])

    elif args.cmd == "status":
        print(json.dumps(engine.status(), indent=2))

    elif args.cmd == "session-start":
        print(json.dumps(engine.session_start(), indent=2))

    elif args.cmd == "wrap-up":
        print(json.dumps(engine.wrap_up(), indent=2))

    elif args.cmd == "add-fact":
        result = engine.add_fact(
            subject=args.subject,
            predicate=args.predicate,
            object_=args.object,
            valid_from=args.valid_from,
            valid_to=args.valid_to,
            source=args.source,
        )
        print(json.dumps(result, indent=2))

    elif args.cmd == "query-entity":
        print(json.dumps(engine.query_entity(args.entity, as_of=args.as_of), indent=2))

    elif args.cmd == "forget":
        print(json.dumps(engine.forget(args.doc_path), indent=2))

    elif args.cmd == "sync":
        print(json.dumps(engine.sync(args.path, pattern=args.pattern), indent=2))


if __name__ == "__main__":
    main()
