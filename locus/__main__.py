"""
locus CLI — command-line interface for Locus vectorless RAG.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _cli_filters(args) -> dict | None:
    """Build a filters dict from `locus retrieve` flags (None when no filter is set)."""
    f: dict = json.loads(args.filters) if args.filters else {}
    for key, val in (("path", args.path), ("exclude_path", args.exclude_path), ("type", args.doc_type),
                     ("tags", args.tag), ("tags_any", args.any_tag), ("date_from", args.date_from),
                     ("date_to", args.date_to), ("as_of", args.in_force)):
        if val:
            f[key] = val
    for item in args.meta or []:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise SystemExit(f"--meta expects FIELD=VALUE, got {item!r}")
        f.setdefault("meta", {})[name.strip()] = value.strip()
    return f or None


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="locus",
        description="Locus — Vectorless RAG with MCP",
    )
    parser.add_argument("--store", default=os.environ.get("LOCUS_STORE", ".locus"), metavar="PATH",
                        help="Store directory (env LOCUS_STORE; default .locus)")

    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    # ----------------------------------------------------------------
    # Core RAG
    # ----------------------------------------------------------------
    p_index = sub.add_parser("index", help="Index a file or directory")
    p_index.add_argument("path")
    p_index.add_argument("--pattern", default="**/*.md")
    p_index.add_argument("--follow-symlinks", choices=["no", "inside", "all"], default=None,
                         help="Symlinks under the directory: skip (default), follow only those resolving "
                              "inside it, or follow all (also LOCUS_FOLLOW_SYMLINKS)")
    p_index.add_argument("--chunk-words", type=int, default=None,
                         help="Chunk size in words (default 400; stored with the index)")
    p_index.add_argument("--overlap-words", type=int, default=None, help="Window overlap (default 50)")
    p_index.add_argument("--heading-context", action="store_true", default=None,
                         help="Prefix every chunk with its heading path")
    p_index.add_argument("--boundaries", choices=["words", "lines"], default=None,
                         help="End windows at line breaks when possible (lines) or fixed word counts (words)")

    p_ret = sub.add_parser("retrieve", help="Retrieve context for a query")
    p_ret.add_argument("query")
    p_ret.add_argument("--limit", type=int, default=5)
    p_ret.add_argument("--as-of", dest="as_of", default=None)
    p_ret.add_argument("--no-links", action="store_true")
    p_ret.add_argument("--mode", choices=["bm25", "hybrid"], default=None,
                       help="bm25 (default) or hybrid re-rank")
    p_ret.add_argument("--track", action="store_true",
                       help="Record hits in the hot-context bulletin")
    p_ret.add_argument("--raw", action="store_true", help="Output JSON")
    p_ret.add_argument("--expand", choices=["rm3"], default=None,
                       help="LLM-free query expansion: RM3 pseudo-relevance feedback")
    p_ret.add_argument("--parent", choices=["section", "doc"], default=None,
                       help="Return the enclosing section/document of matched chunks, with cited spans")
    p_ret.add_argument("--llm", choices=["rewrite", "multi", "hyde"], default=None,
                       help="Optional LLM query transformation (needs OPENROUTER_API_KEY or "
                            "OPENAI_API_KEY[/OPENAI_BASE_URL]; makes network calls)")
    g = p_ret.add_argument_group("metadata filters (applied to the source document)")
    g.add_argument("--path", action="append", default=None, metavar="GLOB",
                   help="Only docs whose path matches GLOB (repeatable: any)")
    g.add_argument("--exclude-path", action="append", default=None, metavar="GLOB")
    g.add_argument("--type", dest="doc_type", action="append", default=None,
                   help="Frontmatter `type` (repeatable: any)")
    g.add_argument("--tag", action="append", default=None, help="Doc must have this tag (repeatable: all)")
    g.add_argument("--any-tag", action="append", default=None, help="Doc has any of these tags (repeatable)")
    g.add_argument("--date-from", default=None, metavar="YYYY-MM-DD", help="Frontmatter `date` >= this")
    g.add_argument("--date-to", default=None, metavar="YYYY-MM-DD", help="Frontmatter `date` <= this")
    g.add_argument("--in-force", dest="in_force", default=None, metavar="YYYY-MM-DD",
                   help="Docs in force on this date (effective/date <= it, not superseded)")
    g.add_argument("--meta", action="append", default=None, metavar="FIELD=VALUE",
                   help="Frontmatter field equals value (repeatable: all)")
    g.add_argument("--filters", default=None, metavar="JSON",
                   help='Full filter object, e.g. \'{"type": "rfi", "meta": {"status": "Closed"}}\'')

    p_reindex = sub.add_parser("reindex", help="Rebuild the lexical index from stored chunks")
    p_reindex.add_argument("--backend", choices=["fts5", "legacy"], default=None)
    p_reindex.add_argument("--analyzer", choices=["plain", "porter", "porter+english"], default=None,
                           help="Term analyzer (default: keep the store's; new stores use porter+english)")

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
    p_mcp.add_argument("--store", default=None, dest="mcp_store")

    p_serve = sub.add_parser("serve", help="Start the HTTP JSON-RPC server")
    p_serve.add_argument("--host",  default=os.environ.get("LOCUS_HTTP_HOST", "127.0.0.1"),
                         help="Bind address (env LOCUS_HTTP_HOST; default 127.0.0.1)")
    p_serve.add_argument("--port",  type=int, default=int(os.environ.get("LOCUS_HTTP_PORT", "7391")),
                         help="Port (env LOCUS_HTTP_PORT; default 7391)")
    p_serve.add_argument("--token", default=os.environ.get("LOCUS_HTTP_TOKEN") or None,
                         help="Bearer token, required for non-loopback hosts (env LOCUS_HTTP_TOKEN; "
                              "prefer --token-file)")
    p_serve.add_argument("--token-file", default=os.environ.get("LOCUS_HTTP_TOKEN_FILE") or None,
                         help="Read the bearer token from a file, e.g. a Docker secret "
                              "(env LOCUS_HTTP_TOKEN_FILE)")
    p_serve.add_argument("--cors-origin", default=None, help="Allowed browser origin (CORS off by default)")

    # ----------------------------------------------------------------
    # Watch
    # ----------------------------------------------------------------
    p_watch = sub.add_parser("watch", help="Watch a directory and auto-reindex changes")
    p_watch.add_argument("path")
    p_watch.add_argument("--interval", type=float, default=5.0, help="Poll interval seconds")
    p_watch.add_argument("--pattern",  default="**/*.md")

    # ----------------------------------------------------------------
    # OMPA bridge
    # ----------------------------------------------------------------
    p_ompa = sub.add_parser("ingest-ompa", help="Import an OMPA vault into Locus")
    p_ompa.add_argument("vault_path", help="Path to OMPA vault root")
    p_ompa.add_argument("--pattern",  default="**/*.md")

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Reasoning, corpus inspection, GitHub bridge
    # ----------------------------------------------------------------
    p_reason = sub.add_parser("reason", help="Multi-hop KG reasoning for a question")
    p_reason.add_argument("question")
    p_reason.add_argument("--depth", type=int, default=3)

    p_paths = sub.add_parser("find-paths", help="Find KG paths between two entities")
    p_paths.add_argument("entity_a")
    p_paths.add_argument("entity_b")
    p_paths.add_argument("--depth", type=int, default=3)

    p_inspect = sub.add_parser("inspect", help="Inspect a document's corpus stats and terms")
    p_inspect.add_argument("doc_path")
    p_inspect.add_argument("--limit", type=int, default=20)

    p_top = sub.add_parser("top-terms", help="Show corpus-wide top terms")
    p_top.add_argument("--limit", type=int, default=20)

    p_gh = sub.add_parser("ingest-github", help="Ingest markdown from a GitHub repository")
    p_gh.add_argument("repo", help="owner/repo")
    p_gh.add_argument("--branch", default="main")
    p_gh.add_argument("--path",   default="")
    p_gh.add_argument("--pattern", default="*.md")
    p_gh.add_argument("--token",  default=None, help="GitHub personal access token")
    p_gh.add_argument("--cache-dir", dest="cache_dir", default=None)

    # ----------------------------------------------------------------
    # Document intelligence
    # ----------------------------------------------------------------
    p_related = sub.add_parser("related", help="Find documents similar to a given doc")
    p_related.add_argument("doc_path")
    p_related.add_argument("--limit", type=int, default=5)

    p_diff = sub.add_parser("diff", help="Preview index changes vs filesystem")
    p_diff.add_argument("path")
    p_diff.add_argument("--pattern", default="**/*.md")

    # ----------------------------------------------------------------
    # Annotations & feedback
    # ----------------------------------------------------------------
    p_ann = sub.add_parser("annotate", help="Attach a label to a chunk")
    p_ann.add_argument("chunk_id")
    p_ann.add_argument("label")
    p_ann.add_argument("--note", default=None)

    p_get_ann = sub.add_parser("annotations", help="Show annotations for a chunk")
    p_get_ann.add_argument("chunk_id")

    p_relevant = sub.add_parser("mark-relevant", help="Mark a chunk as relevant for a query")
    p_relevant.add_argument("chunk_id")
    p_relevant.add_argument("query")

    p_irrelevant = sub.add_parser("mark-irrelevant", help="Mark a chunk as irrelevant for a query")
    p_irrelevant.add_argument("chunk_id")
    p_irrelevant.add_argument("query")

    # ----------------------------------------------------------------
    # Query intelligence & snapshot
    # ----------------------------------------------------------------
    p_expand = sub.add_parser("expand", help="Expand a query using KG aliases and neighbours")
    p_expand.add_argument("query")
    p_expand.add_argument("--max", type=int, default=5, dest="max_expansions")

    p_multi = sub.add_parser("multi-retrieve", help="Retrieve and fuse results for multiple query variants")
    p_multi.add_argument("queries", nargs="+")
    p_multi.add_argument("--limit", type=int, default=5)
    p_multi.add_argument("--as-of", dest="as_of", default=None)

    p_timeline = sub.add_parser("timeline", help="Show chronological KG timeline for an entity")
    p_timeline.add_argument("entity")
    p_timeline.add_argument("--as-of", dest="as_of", default=None)

    p_snap = sub.add_parser("snapshot", help="Archive the store to a .tar.gz file")
    p_snap.add_argument("output", help="Output archive path")

    p_restore = sub.add_parser("restore", help="Restore a store from a snapshot archive")
    p_restore.add_argument("snapshot", help="Path to .tar.gz snapshot")
    p_restore.add_argument("--target", default=".locus-restored", help="Destination store path")
    p_restore.add_argument("--overwrite", action="store_true")

    p_snap_info = sub.add_parser("snapshot-info", help="Inspect a snapshot archive without extracting")
    p_snap_info.add_argument("snapshot")

    p_bench = sub.add_parser("benchmark", help="Measure retrieval quality (recall@K, MRR)")
    p_bench.add_argument("qa_file", help="JSON file with [{query, expected_docs}] pairs")
    p_bench.add_argument("--k", default="1,3,5", help="Comma-separated K values")

    p_prep = sub.add_parser("prepare-context", help="All-in-one: retrieve+rerank+pack+KG context")
    p_prep.add_argument("query")
    p_prep.add_argument("--limit",  type=int, default=5)
    p_prep.add_argument("--budget", type=int, default=4000, help="Token budget")
    p_prep.add_argument("--no-rerank", action="store_true")
    p_prep.add_argument("--as-of", dest="as_of", default=None)

    sub.add_parser("cache-stats", help="Query cache statistics")
    sub.add_parser("cache-clear", help="Invalidate query cache")

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
        mcp_main(store_path=args.mcp_store or args.store)
        return

    if args.cmd == "serve":
        import logging
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        from .core import LocusEngine
        from .mcp.http_server import resolve_token, serve
        try:
            token = resolve_token(args.token, args.token_file)
            serve(
                LocusEngine(store_path=args.store),
                host=args.host,
                port=args.port,
                token=token,
                cors_origin=args.cors_origin,
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        return

    if args.cmd is None:
        parser.print_help()
        return

    from .core import LocusEngine
    chunking = None
    if args.cmd == "index":
        opts = {"chunk_words": args.chunk_words, "overlap_words": args.overlap_words,
                "heading_context": args.heading_context, "boundaries": args.boundaries}
        opts = {k: v for k, v in opts.items() if v is not None}
        if opts:
            from .memory.chunker import normalize_chunking
            from .memory.corpus import Corpus
            # Start from the store's current settings, then apply the flags.
            current = Corpus(Path(args.store) / "corpus").chunking
            chunking = normalize_chunking({**current, **opts})
    follow = None
    if args.cmd == "index" and args.follow_symlinks:
        follow = {"no": False, "inside": "inside", "all": True}[args.follow_symlinks]
    engine = LocusEngine(store_path=args.store, chunking=chunking, follow_symlinks=follow)

    # ----------------------------------------------------------------
    # Engine commands
    # ----------------------------------------------------------------
    if args.cmd == "index":
        print(json.dumps(engine.index(args.path, pattern=args.pattern), indent=2))

    elif args.cmd == "reindex":
        print(json.dumps(engine.reindex(backend=args.backend, analyzer=args.analyzer), indent=2))

    elif args.cmd == "retrieve":
        if args.llm:
            from .llm import LLMError, OpenAICompatibleClient, llm_retrieve
            try:
                client = OpenAICompatibleClient.from_env()
                chunks = llm_retrieve(engine, args.query, strategy=args.llm, client=client,
                                      limit=args.limit, as_of=args.as_of, use_links=not args.no_links,
                                      mode=args.mode, filters=_cli_filters(args),
                                      expansion=args.expand, parent=args.parent)
            except LLMError as e:
                raise SystemExit(f"LLM error: {e}")
            print(f"[llm] {args.llm} usage: {json.dumps(client.usage)}", file=sys.stderr)
        else:
            chunks = engine.retrieve(
                query=args.query,
                limit=args.limit,
                as_of=args.as_of,
                use_links=not args.no_links,
                mode=args.mode,
                track=args.track,
                filters=_cli_filters(args),
                expansion=args.expand,
                parent=args.parent,
            )
        if args.raw:
            print(json.dumps([
                {"chunk_id": c.chunk_id, "doc_path": c.doc_path,
                 "score": round(c.score, 4), "provenance": c.provenance,
                 "content": c.content[:400] if not c.spans else c.content,
                 **({"spans": c.spans} if c.spans else {})}
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
    # Watch, bridge, eval commands
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

    elif args.cmd == "prepare-context":
        print(json.dumps(engine.prepare_context(
            args.query,
            limit=args.limit,
            token_budget=args.budget,
            rerank=not args.no_rerank,
            as_of=args.as_of,
        ), indent=2))

    elif args.cmd == "cache-stats":
        print(json.dumps(engine.cache_stats(), indent=2))

    elif args.cmd == "cache-clear":
        print(json.dumps(engine.clear_cache(), indent=2))

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

    elif args.cmd == "reason":
        print(json.dumps(engine.reason(args.question, max_depth=args.depth), indent=2))

    elif args.cmd == "find-paths":
        print(json.dumps(engine.find_paths(args.entity_a, args.entity_b, max_depth=args.depth), indent=2))

    elif args.cmd == "inspect":
        print(json.dumps(engine.inspect_doc(args.doc_path, limit=args.limit), indent=2))

    elif args.cmd == "top-terms":
        print(json.dumps(engine.top_terms(limit=args.limit), indent=2))

    elif args.cmd == "ingest-github":
        from .bridge.github import GitHubBridge
        bridge = GitHubBridge(engine, repo=args.repo, token=args.token)
        result = bridge.ingest(
            branch=args.branch,
            path=args.path,
            pattern=args.pattern,
            cache_dir=args.cache_dir,
        )
        print(json.dumps(result, indent=2))

    elif args.cmd == "benchmark":
        from .eval import LocusEval
        k_values = [int(k.strip()) for k in args.k.split(",")]
        ev = LocusEval(engine, k_values=k_values)
        report = ev.score_from_file(args.qa_file)
        print(report.summary())

    # ----------------------------------------------------------------
    # Query intelligence & snapshot commands
    # ----------------------------------------------------------------
    elif args.cmd == "related":
        print(json.dumps(engine.related_docs(args.doc_path, limit=args.limit), indent=2))

    elif args.cmd == "diff":
        print(json.dumps(engine.diff(args.path, pattern=args.pattern), indent=2))

    elif args.cmd == "annotate":
        print(json.dumps(engine.annotate(args.chunk_id, args.label, args.note), indent=2))

    elif args.cmd == "annotations":
        print(json.dumps(engine.get_annotations(args.chunk_id), indent=2))

    elif args.cmd == "mark-relevant":
        print(json.dumps(engine.mark_relevant(args.chunk_id, args.query), indent=2))

    elif args.cmd == "mark-irrelevant":
        print(json.dumps(engine.mark_irrelevant(args.chunk_id, args.query), indent=2))

    elif args.cmd == "expand":
        print(json.dumps(engine.expand_query(args.query, max_expansions=args.max_expansions), indent=2))

    elif args.cmd == "multi-retrieve":
        chunks = engine.multi_retrieve(args.queries, limit=args.limit, as_of=args.as_of)
        if not chunks:
            print("No results.")
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] {c.doc_path}  score={c.score:.4f}  via={c.provenance}")
            print("-" * 60)
            print(c.content[:400])

    elif args.cmd == "timeline":
        print(json.dumps(engine.timeline(args.entity, as_of=args.as_of), indent=2))

    elif args.cmd == "snapshot":
        print(json.dumps(engine.snapshot(args.output), indent=2))

    elif args.cmd == "restore":
        from .snapshot import LocusSnapshot
        print(json.dumps(
            LocusSnapshot.load(args.snapshot, store_path=args.target, overwrite=args.overwrite),
            indent=2,
        ))

    elif args.cmd == "snapshot-info":
        from .snapshot import LocusSnapshot
        print(json.dumps(LocusSnapshot.inspect(args.snapshot), indent=2))


if __name__ == "__main__":
    main()
