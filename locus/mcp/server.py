"""
Locus MCP Server — Vectorless RAG as Model Context Protocol tools.
JSON-RPC over stdin/stdout. Compatible with Claude Desktop, Cursor, Windsurf.

Capabilities:
  tools     — 20 RAG + KG + cluster tools
  resources — indexed documents browsable as locus://doc/{path}
  prompts   — 4 pre-built prompt templates with live context injection

Usage:
    claude mcp add locus -- py -3 -m locus.mcp.server --store /path/to/.locus
    locus mcp --store /path/to/.locus
"""

import argparse
import json
import logging
import sys

from ..core import LocusEngine, __version__
from .tools import TOOLS
from .prompts import list_prompts, render_prompt

logger = logging.getLogger(__name__)

# Per-store engine cache
_engines: dict[str, LocusEngine] = {}
# Per-registry cluster cache
_clusters: dict[str, object] = {}   # LocusCluster, typed as object to avoid import cycle


def _get_engine(store_path: str = ".locus") -> LocusEngine:
    if store_path not in _engines:
        _engines[store_path] = LocusEngine(store_path=store_path)
    return _engines[store_path]


def _get_cluster(cluster_path: str):
    if cluster_path not in _clusters:
        from ..cluster import LocusCluster
        _clusters[cluster_path] = LocusCluster(registry_path=cluster_path)
    return _clusters[cluster_path]


def _list_tools() -> dict:
    return {
        "tools": [
            {
                "name": name,
                "description": spec["description"],
                "inputSchema": spec["input_schema"],
            }
            for name, spec in TOOLS.items()
        ]
    }


def _call_tool(name: str, arguments: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}

    try:
        # Cluster tools — separate code path
        if name in ("locus_cluster_retrieve", "locus_add_node", "locus_remove_node", "locus_list_nodes"):
            return _handle_cluster_tool(name, arguments)

        store_path = str(arguments.get("store_path", ".locus"))
        if ".." in store_path:
            return {"error": "Invalid store_path"}
        engine = _get_engine(store_path)

        if name == "locus_index":
            return engine.index(arguments["path"], pattern=arguments.get("pattern", "**/*.md"))

        if name == "locus_retrieve":
            from ..retrieval.classifier import QueryIntent
            raw_intent = arguments.get("intent")
            intent = QueryIntent(raw_intent) if raw_intent else None
            full_content = bool(arguments.get("full_content", False))
            chunks = engine.retrieve(
                query=arguments["query"],
                limit=min(int(arguments.get("limit", 5)), 20),
                as_of=arguments.get("as_of"),
                use_links=bool(arguments.get("use_links", True)),
                intent=intent,
            )
            return {
                "intent": intent.value if intent else "auto",
                "results": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_path": c.doc_path,
                        "score": round(c.score, 4),
                        "provenance": c.provenance,
                        "content": c.content if full_content else c.content[:600],
                        "entities": c.entities or [],
                    }
                    for c in chunks
                ],
            }

        if name == "locus_explain":
            return engine.explain(
                chunk_id=arguments["chunk_id"],
                query=arguments.get("query"),
            )

        if name == "locus_add_fact":
            return engine.add_fact(
                subject=arguments["subject"],
                predicate=arguments["predicate"],
                object_=arguments["object"],
                valid_from=arguments.get("valid_from"),
                valid_to=arguments.get("valid_to"),
                source=arguments.get("source"),
            )

        if name == "locus_query_entity":
            return engine.query_entity(arguments["entity"], as_of=arguments.get("as_of"))

        if name == "locus_hot_context":
            hot = engine.bulletin.inject(token_limit=int(arguments.get("token_limit", 1500)))
            return {"hot_context": hot or "(empty)"}

        if name == "locus_promote":
            success = engine.bulletin.promote_to_pin(arguments["chunk_id"])
            return {"success": success, "chunk_id": arguments["chunk_id"]}

        if name == "locus_session_start":
            return engine.session_start()

        if name == "locus_wrap_up":
            return engine.wrap_up()

        if name == "locus_status":
            return engine.status()

        if name == "locus_forget":
            return engine.forget(arguments["doc_path"])

        if name == "locus_kg_stats":
            return engine.kg.stats()

        if name == "locus_sync":
            return engine.sync(arguments["path"], pattern=arguments.get("pattern", "**/*.md"))

        if name == "locus_contradictions":
            return engine.find_contradictions(entity=arguments.get("entity"))

        if name == "locus_add_alias":
            return engine.add_alias(arguments["alias"], arguments["canonical"])

        if name == "locus_suggest_aliases":
            return engine.suggest_aliases(threshold=float(arguments.get("threshold", 0.75)))

        if name == "locus_prepare_context":
            return engine.prepare_context(
                query=arguments["query"],
                limit=min(int(arguments.get("limit", 5)), 20),
                token_budget=int(arguments.get("token_budget", 4000)),
                rerank=bool(arguments.get("rerank", True)),
                as_of=arguments.get("as_of"),
            )

        if name == "locus_cache_stats":
            return engine.cache_stats()

        if name == "locus_clear_cache":
            return engine.clear_cache()

        if name == "locus_kg_traverse":
            return engine.kg_traverse(
                start=arguments["start"],
                max_depth=int(arguments.get("max_depth", 2)),
                predicate_filter=arguments.get("predicate_filter"),
                direction=arguments.get("direction", "both"),
            )

        if name == "locus_kg_match":
            return engine.kg_match(
                subject=arguments.get("subject", "*"),
                predicate=arguments.get("predicate", "*"),
                obj=arguments.get("obj", "*"),
                as_of=arguments.get("as_of"),
            )

        if name == "locus_doctor":
            return engine.doctor()

        if name == "locus_export_kg":
            return engine.export_kg(
                path=arguments["path"],
                fmt=arguments.get("format"),
            )

        if name == "locus_ingest_ompa":
            from ..bridge.ompa import OMPABridge
            bridge = OMPABridge(engine, vault_path=arguments["vault_path"])
            return bridge.ingest(pattern=arguments.get("pattern", "**/*.md"))

        if name == "locus_benchmark":
            from ..eval import LocusEval
            k_values = arguments.get("k_values", [1, 3, 5])
            ev = LocusEval(engine, k_values=k_values)
            report = ev.score_from_file(arguments["qa_file"])
            return report.to_dict()

        # Phase 9 — reasoning, corpus inspection, GitHub bridge
        if name == "locus_reason":
            return engine.reason(
                question=arguments["question"],
                max_depth=int(arguments.get("max_depth", 3)),
            )

        if name == "locus_find_paths":
            return {
                "paths": engine.find_paths(
                    entity_a=arguments["entity_a"],
                    entity_b=arguments["entity_b"],
                    max_depth=int(arguments.get("max_depth", 3)),
                    predicate_filter=arguments.get("predicate_filter"),
                )
            }

        if name == "locus_inspect":
            return engine.inspect_doc(
                doc_path=arguments["doc_path"],
                limit=int(arguments.get("limit", 20)),
            )

        if name == "locus_top_terms":
            return {"terms": engine.top_terms(limit=int(arguments.get("limit", 20)))}

        if name == "locus_ingest_github":
            from ..bridge.github import GitHubBridge
            bridge = GitHubBridge(engine, repo=arguments["repo"], token=arguments.get("token"))
            return bridge.ingest(
                branch=arguments.get("branch", "main"),
                path=arguments.get("path", ""),
                pattern=arguments.get("pattern", "*.md"),
            )

        # Phase 10 — query intelligence + snapshot
        if name == "locus_expand_query":
            return engine.expand_query(
                query=arguments["query"],
                max_expansions=int(arguments.get("max_expansions", 5)),
            )

        if name == "locus_multi_retrieve":
            queries = arguments["queries"]
            chunks = engine.multi_retrieve(
                queries=queries,
                limit=min(int(arguments.get("limit", 5)), 20),
                as_of=arguments.get("as_of"),
            )
            return {
                "queries": queries,
                "results": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_path": c.doc_path,
                        "score": round(c.score, 4),
                        "provenance": c.provenance,
                        "content": c.content[:600],
                    }
                    for c in chunks
                ],
            }

        if name == "locus_timeline":
            return engine.timeline(
                entity=arguments["entity"],
                as_of=arguments.get("as_of"),
            )

        if name == "locus_snapshot":
            return engine.snapshot(arguments["output_path"])

        if name == "locus_restore":
            from ..snapshot import LocusSnapshot
            return LocusSnapshot.load(
                snapshot_path=arguments["snapshot_path"],
                store_path=arguments.get("store_path", ".locus-restored"),
                overwrite=bool(arguments.get("overwrite", False)),
            )

        if name == "locus_snapshot_info":
            from ..snapshot import LocusSnapshot
            return LocusSnapshot.inspect(arguments["snapshot_path"])

        return {"error": f"Unhandled tool: {name}"}

    except KeyError as e:
        return {"error": f"Missing required argument: {e}"}
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return {"error": type(e).__name__}


def _handle_cluster_tool(name: str, arguments: dict) -> dict:
    cluster_path = arguments.get("cluster_path", "")
    if not cluster_path or ".." in cluster_path:
        return {"error": "Invalid cluster_path"}
    cluster = _get_cluster(cluster_path)

    if name == "locus_cluster_retrieve":
        chunks = cluster.retrieve(
            query=arguments["query"],
            limit=min(int(arguments.get("limit", 5)), 20),
            nodes=arguments.get("nodes"),
            as_of=arguments.get("as_of"),
        )
        return {
            "results": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_path": c.doc_path,
                    "score": round(c.score, 4),
                    "provenance": c.provenance,
                    "content": c.content[:600],
                }
                for c in chunks
            ]
        }

    if name == "locus_add_node":
        return cluster.add_node(arguments["name"], arguments["store_path"])

    if name == "locus_remove_node":
        return cluster.remove_node(arguments["name"])

    if name == "locus_list_nodes":
        return {"nodes": cluster.list_nodes()}

    return {"error": f"Unhandled cluster tool: {name}"}


# ------------------------------------------------------------------
# MCP Resources
# ------------------------------------------------------------------

def _handle_resources_list(engine: LocusEngine) -> dict:
    return {
        "resources": [
            {
                "uri": f"locus://doc/{doc_path}",
                "name": doc_path,
                "mimeType": "text/markdown",
                "description": f"Indexed document: {doc_path}",
            }
            for doc_path in engine.corpus.list_docs()
        ]
    }


def _handle_resources_read(uri: str, engine: LocusEngine) -> dict:
    if not uri.startswith("locus://doc/"):
        return {"error": f"Unknown URI scheme: {uri}"}
    doc_path = uri[len("locus://doc/"):]
    chunks = engine.corpus.get_chunks_for_doc(doc_path)
    if not chunks:
        return {"error": f"Document not found: {doc_path}"}
    content = "\n\n---\n\n".join(c.content for c in chunks)
    return {
        "contents": [
            {"uri": uri, "mimeType": "text/markdown", "text": content}
        ]
    }


# ------------------------------------------------------------------
# MCP Protocol
# ------------------------------------------------------------------

def _send(response: dict) -> None:
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def main(store_path: str = ".locus") -> None:
    # Pre-warm the default engine
    _engines[store_path] = LocusEngine(store_path=store_path)

    request: dict | None = None
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line.strip())
            method  = request.get("method", "")
            req_id  = request.get("id")
            params  = request.get("params", {})

            # Core MCP lifecycle
            if method == "initialize":
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools":     {},
                            "resources": {},
                            "prompts":   {},
                        },
                        "serverInfo": {"name": "locus", "version": __version__},
                    },
                })

            elif method == "tools/list":
                _send({"jsonrpc": "2.0", "id": req_id, "result": _list_tools()})

            elif method == "tools/call":
                result = _call_tool(
                    name=params["name"],
                    arguments=params.get("arguments", {}),
                )
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                    },
                })

            # Resources
            elif method == "resources/list":
                sp = params.get("store_path", store_path)
                engine = _get_engine(sp)
                _send({"jsonrpc": "2.0", "id": req_id, "result": _handle_resources_list(engine)})

            elif method == "resources/read":
                sp = params.get("store_path", store_path)
                engine = _get_engine(sp)
                result = _handle_resources_read(params.get("uri", ""), engine)
                _send({"jsonrpc": "2.0", "id": req_id, "result": result})

            # Prompts
            elif method == "prompts/list":
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"prompts": list_prompts()},
                })

            elif method == "prompts/get":
                sp = params.get("store_path", store_path)
                engine = _get_engine(sp)
                messages = render_prompt(
                    name=params.get("name", ""),
                    args=params.get("arguments", {}),
                    engine=engine,
                )
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"messages": messages},
                })

            elif method in ("notifications/initialized",):
                pass

            elif method in ("shutdown", "exit"):
                break

        except Exception as e:
            req_id = request.get("id") if isinstance(request, dict) else None
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": type(e).__name__},
            })
            request = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locus MCP server")
    parser.add_argument("--store", default=".locus")
    args = parser.parse_args()
    main(store_path=args.store)
