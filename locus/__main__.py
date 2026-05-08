"""
locus CLI — command-line interface for Locus vectorless RAG.
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="locus",
        description="Locus — Vectorless RAG with MCP",
    )
    parser.add_argument("--store", default=".locus", metavar="PATH")

    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    # ----------------------------------------------------------------
    # Core RAG
    # ----------------------------------------------------------------
    p_index = sub.add_parser("index", help="Index a file or directory")
    p_index.add_argument("path")
    p_index.add_argument("--pattern", default="**/*.md")

    p_ret = sub.add_parser("retrieve", help="Retrieve context for a query")
    p_ret.add_argument("query")
    p_ret.add_argument("--limit", type=int, default=5)
    p_ret.add_argument("--as-of", dest="as_of", default=None)
    p_ret.add_argument("--no-links", action="store_true")
    p_ret.add_argument("--raw", action="store_true", help="Output JSON")

    p_explain = sub.add_parser("explain", help="Explain why a chunk was retrieved")
    p_explain.add_argument("chunk_id")
    p_explain.add_argument("--query", default=None)

    sub.add_parser("status",        help="Show corpus/KG/bulletin stats")
    sub.add_parser("session-start", help="Warm context open")
    sub.add_parser("wrap-up",       help="Session close + bulletin tick")

    p_fact = sub.add_parser("add-fact", help="Add a KG triple")
    p_fact.add_argument("subject")
    p_fact.add_argument("predicate")
    p_fact.add_argument("object")
    p_fact.add_argument("--valid-from", dest="valid_from", default=None)
    p_fact.add_argument("--valid-to",   dest="valid_to",   default=None)
    p_fact.add_argument("--source",     default=None)

    p_entity = sub.add_parser("query-entity", help="Query KG facts for an entity")
    p_entity.add_argument("entity")
    p_entity.add_argument("--as-of", dest="as_of", default=None)

    p_alias = sub.add_parser("add-alias", help="Add entity alias")
    p_alias.add_argument("alias")
    p_alias.add_argument("canonical")

    p_contra = sub.add_parser("contradictions", help="Find KG contradictions")
    p_contra.add_argument("entity", nargs="?", default=None)

    p_forget = sub.add_parser("forget", help="Remove a document from the corpus")
    p_forget.add_argument("doc_path")

    p_sync = sub.add_parser("sync", help="Full reindex from a directory")
    p_sync.add_argument("path")
    p_sync.add_argument("--pattern", default="**/*.md")

    # ----------------------------------------------------------------
    # MCP servers
    # ----------------------------------------------------------------
    p_mcp = sub.add_parser("mcp", help="Start the MCP stdio server")
    p_mcp.add_argument("--store", default=".locus", dest="mcp_store")

    p_serve = sub.add_parser("serve", help="Start the HTTP JSON-RPC server")
    p_serve.add_argument("--host",  default="0.0.0.0")
    p_serve.add_argument("--port",  type=int, default=7391)
    p_serve.add_argument("--token", default=None, help="Optional Bearer token")

    # ----------------------------------------------------------------
    # Phase 6 — Watch
    # ----------------------------------------------------------------
    p_watch = sub.add_parser("watch", help="Watch a directory and auto-reindex changes")
    p_watch.add_argument("path")
    p_watch.add_argument("--interval", type=float, default=5.0, help="Poll interval seconds")
    p_watch.add_argument("--pattern",  default="**/*.md")

    # ----------------------------------------------------------------
    # Phase 6 — OMPA bridge
    # ----------------------------------------------------------------
    p_ompa = sub.add_parser("ingest-ompa", help="Import an OMPA vault into Locus")
    p_ompa.add_argument("vault_path", help="Path to OMPA vault root")
    p_ompa.add_argument("--pattern",  default="**/*.md")

    # ----------------------------------------------------------------
    # Phase 6 — Evaluation
    # ----------------------------------------------------------------
    p_bench = sub.add_parser("benchmark", help="Measure retrieval quality (recall@K, MRR)")
    p_bench.add_argument("qa_file", help="JSON file with [{query, expected_docs}] pairs")
    p_bench.add_argument("--k", default="1,3,5", help="Comma-separated K values")

    sub.add_parser("doctor", help="Health check: corpus, KG, bulletin, store size")

    p_export = sub.add_parser("export-kg", help="Export KG to GraphML / JSONL / DOT")
    p_export.add_argument("path", help="Output file path")
    p_export.add_argument("--format", choices=["graphml", "jsonl", "dot"], default=None)

    p_traverse = sub.add_parser("kg-traverse", help="BFS traversal from an entity")
    p_traverse.add_argument("start")
    p_traverse.add_argument("--depth", type=int, default=2)
    p_traverse.add_argument("--direction", choices=["out", "in", "both"], default="both")

    p_match = sub.add_parser("kg-match", help="Pattern match over the KG (* = wildcard)")
    p_match.add_argument("--subject",   default="*")
    p_match.add_argument("--predicate", default="*")
    p_match.add_argument("--obj",       default="*")
    p_match.add_argument("--as-of",     dest="as_of", default=None)

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Dispatch — non-engine commands first
    # ----------------------------------------------------------------
    if args.cmd == "mcp":
        from .mcp.server import main as mcp_main
        mcp_main(store_path=getattr(args, "mcp_store", ".locus"))
        return

    if args.cmd == "serve":
        import logging
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        from .core import LocusEngine
        from .mcp.http_server import serve
        serve(
            LocusEngine(store_path=args.store),
            host=args.host,
            port=args.port,
            token=args.token,
        )
        return

    if args.cmd is None:
        parser.print_help()
        return

    from .core import LocusEngine
    engine = LocusEngine(store_path=args.store)

    # ----------------------------------------------------------------
    # Engine commands
    # ----------------------------------------------------------------
    if args.cmd == "index":
        print(json.dumps(engine.index(args.path, pattern=args.pattern), indent=2))

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

    elif args.cmd == "explain":
        print(json.dumps(engine.explain(args.chunk_id, query=args.query), indent=2))

    elif args.cmd == "status":
        print(json.dumps(engine.status(), indent=2))

    elif args.cmd == "session-start":
        print(json.dumps(engine.session_start(), indent=2))

    elif args.cmd == "wrap-up":
        print(json.dumps(engine.wrap_up(), indent=2))

    elif args.cmd == "add-fact":
        print(json.dumps(engine.add_fact(
            args.subject, args.predicate, args.object,
            valid_from=args.valid_from, valid_to=args.valid_to, source=args.source,
        ), indent=2))

    elif args.cmd == "query-entity":
        print(json.dumps(engine.query_entity(args.entity, as_of=args.as_of), indent=2))

    elif args.cmd == "add-alias":
        print(json.dumps(engine.add_alias(args.alias, args.canonical), indent=2))

    elif args.cmd == "contradictions":
        print(json.dumps(engine.find_contradictions(args.entity), indent=2))

    elif args.cmd == "forget":
        print(json.dumps(engine.forget(args.doc_path), indent=2))

    elif args.cmd == "sync":
        print(json.dumps(engine.sync(args.path, pattern=args.pattern), indent=2))

    # ----------------------------------------------------------------
    # Phase 6 commands
    # ----------------------------------------------------------------
    elif args.cmd == "watch":
        import logging
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        from .watcher import LocusWatcher
        watcher = LocusWatcher(
            engine,
            watch_dir=args.path,
            pattern=args.pattern,
            interval=args.interval,
        )
        print(f"Watching {args.path}  (interval={args.interval}s)  Ctrl-C to stop")
        watcher.start(background=False)

    elif args.cmd == "doctor":
        from .doctor import LocusDoctor
        print(LocusDoctor(engine).report())

    elif args.cmd == "export-kg":
        print(json.dumps(engine.export_kg(args.path, fmt=args.format), indent=2))

    elif args.cmd == "kg-traverse":
        print(json.dumps(engine.kg_traverse(args.start, max_depth=args.depth, direction=args.direction), indent=2))

    elif args.cmd == "kg-match":
        print(json.dumps(engine.kg_match(subject=args.subject, predicate=args.predicate, obj=args.obj, as_of=args.as_of), indent=2))

    elif args.cmd == "ingest-ompa":
        from .bridge.ompa import OMPABridge
        bridge = OMPABridge(engine, vault_path=args.vault_path)
        result = bridge.ingest(pattern=args.pattern)
        print(json.dumps(result, indent=2))

    elif args.cmd == "benchmark":
        from .eval import LocusEval
        k_values = [int(k.strip()) for k in args.k.split(",")]
        ev = LocusEval(engine, k_values=k_values)
        report = ev.score_from_file(args.qa_file)
        print(report.summary())


if __name__ == "__main__":
    main()
