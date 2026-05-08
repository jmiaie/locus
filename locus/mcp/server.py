"""
Locus MCP Server — Vectorless RAG as Model Context Protocol tools.
JSON-RPC over stdin/stdout. Compatible with Claude Desktop, Cursor, Windsurf,
and any MCP-compatible client.

Usage:
    # Claude Code / Desktop
    claude mcp add locus -- python -m locus.mcp.server --store /path/to/.locus

    # Or via the CLI
    locus mcp --store /path/to/.locus
"""

import argparse
import json
import logging
import sys

from ..core import LocusEngine, __version__
from .tools import TOOLS

logger = logging.getLogger(__name__)

_engine: LocusEngine | None = None


def _get_engine(store_path: str = ".locus") -> LocusEngine:
    global _engine
    if _engine is None or str(_engine.store_path) != str(store_path):
        _engine = LocusEngine(store_path=store_path)
    return _engine


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
        store_path = str(arguments.get("store_path", ".locus"))
        if ".." in store_path:
            return {"error": "Invalid store_path"}

        engine = _get_engine(store_path)

        if name == "locus_index":
            return engine.index(
                arguments["path"],
                pattern=arguments.get("pattern", "**/*.md"),
            )

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
            return engine.query_entity(
                entity=arguments["entity"],
                as_of=arguments.get("as_of"),
            )

        if name == "locus_hot_context":
            hot = engine.bulletin.inject(
                token_limit=int(arguments.get("token_limit", 1500))
            )
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
            return engine.sync(
                arguments["path"],
                pattern=arguments.get("pattern", "**/*.md"),
            )

        return {"error": f"Unhandled tool: {name}"}

    except KeyError as e:
        return {"error": f"Missing required argument: {e}"}
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return {"error": type(e).__name__}


def _send(response: dict) -> None:
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def main(store_path: str = ".locus") -> None:
    global _engine
    _engine = LocusEngine(store_path=store_path)

    request: dict | None = None
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line.strip())
            method = request.get("method", "")
            req_id = request.get("id")

            if method == "initialize":
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "locus", "version": __version__},
                    },
                })

            elif method == "tools/list":
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": _list_tools(),
                })

            elif method == "tools/call":
                result = _call_tool(
                    name=request["params"]["name"],
                    arguments=request["params"].get("arguments", {}),
                )
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                    },
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
    parser.add_argument("--store", default=".locus", help="Store path")
    args = parser.parse_args()
    main(store_path=args.store)
