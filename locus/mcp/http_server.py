"""
Locus HTTP JSON-RPC server — alternative transport for MCP-style tool calls.

Exposes the same JSON-RPC request/response interface as the stdio MCP
server, but over HTTP POST so remote clients (other Jarv/Kai/Tai nodes,
web clients) can call Locus tools without a subprocess.

Endpoints:
  POST /rpc     — JSON-RPC 2.0 request → response (same as stdio)
  GET  /health  — {"ok": true, "version": "0.5.0"}
  GET  /tools   — list of available tools (convenience, non-MCP)

Optional bearer-token auth via --token.  Zero external dependencies —
uses stdlib http.server only.

Usage:
    locus serve --store /path/.locus --port 7391
    # or
    py -3 -m locus.mcp.http_server --store /path/.locus --port 7391

Remote call example:
    curl -s -X POST http://localhost:7391/rpc \\
         -H "Content-Type: application/json" \\
         -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"locus_status","arguments":{}}}'
"""

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable

from ..core import LocusEngine, __version__

logger = logging.getLogger(__name__)


def _build_handler(engine: LocusEngine, token: str | None = None) -> type:
    """Return a handler class closed over engine and token."""

    class _Handler(BaseHTTPRequestHandler):
        _engine = engine
        _token = token

        # ----------------------------------------------------------------
        # Auth
        # ----------------------------------------------------------------

        def _authorized(self) -> bool:
            if not self._token:
                return True
            auth = self.headers.get("Authorization", "")
            return auth == f"Bearer {self._token}"

        def _send_json(self, data: dict, status: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        # ----------------------------------------------------------------
        # Routes
        # ----------------------------------------------------------------

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json({"ok": True, "version": __version__})
            elif self.path == "/tools":
                from .tools import TOOLS
                self._send_json({
                    "tools": [
                        {"name": k, "description": v["description"]}
                        for k, v in TOOLS.items()
                    ]
                })
            else:
                self._send_json({"error": "Not found"}, 404)

        def do_POST(self) -> None:
            if not self._authorized():
                self._send_json({"error": "Unauthorized"}, 401)
                return

            if self.path != "/rpc":
                self._send_json({"error": "Not found"}, 404)
                return

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                request = json.loads(body)
            except json.JSONDecodeError:
                self._send_json(
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
                    400,
                )
                return

            response = _dispatch(request, self._engine)
            self._send_json(response)

        def log_message(self, fmt, *args) -> None:  # silence default access log
            logger.debug(fmt, *args)

    return _Handler


def _dispatch(request: dict, engine: LocusEngine) -> dict:
    """Handle a single JSON-RPC request and return a response dict."""
    from .server import _list_tools, _call_tool, _handle_resources_list, _handle_resources_read
    from .prompts import list_prompts, render_prompt

    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params", {})

    try:
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "serverInfo": {"name": "locus-http", "version": __version__},
            }
        elif method == "tools/list":
            result = _list_tools()
        elif method == "tools/call":
            tool_result = _call_tool(params.get("name", ""), params.get("arguments", {}))
            result = {"content": [{"type": "text", "text": json.dumps(tool_result, indent=2)}]}
        elif method == "resources/list":
            result = _handle_resources_list(engine)
        elif method == "resources/read":
            result = _handle_resources_read(params.get("uri", ""), engine)
        elif method == "prompts/list":
            result = {"prompts": list_prompts()}
        elif method == "prompts/get":
            messages = render_prompt(params.get("name", ""), params.get("arguments", {}), engine)
            result = {"messages": messages}
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    except Exception as e:
        logger.exception("RPC dispatch error for %s", method)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32603, "message": type(e).__name__},
        }


def serve(
    engine: LocusEngine,
    host: str = "0.0.0.0",
    port: int = 7391,
    token: str | None = None,
) -> None:
    """Start the HTTP server (blocking)."""
    handler = _build_handler(engine, token=token)
    server = HTTPServer((host, port), handler)
    logger.info("Locus HTTP server at http://%s:%d", host, port)
    if token:
        logger.info("Auth enabled — Bearer token required")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("HTTP server stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locus HTTP server")
    parser.add_argument("--store", default=".locus")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7391)
    parser.add_argument("--token", default=None, help="Optional Bearer token")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    serve(LocusEngine(store_path=args.store), host=args.host, port=args.port, token=args.token)
