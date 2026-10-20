"""
Locus HTTP JSON-RPC server — alternative transport for MCP-style tool calls.

Exposes the same JSON-RPC request/response interface as the stdio MCP
server, but over HTTP POST so remote clients (other Jarv/Kai/Tai nodes,
web clients) can call Locus tools without a subprocess.

Endpoints:
  POST /rpc     — JSON-RPC 2.0 request → response (same as stdio)
  GET  /health  — {"ok": true, "version": "<locus version>"}
  GET  /tools   — list of available tools (convenience, non-MCP)

Security defaults:
  - binds 127.0.0.1 unless --host says otherwise
  - a bearer token (--token) is required for any non-loopback bind
  - no CORS headers unless --cors-origin names an allowed origin
  - without a token, only loopback Host headers are served (DNS-rebinding guard);
    a browser Origin must be loopback or the --cors-origin value

Zero external dependencies — uses stdlib http.server only.

Usage:
    locus --store /path/.locus serve --port 7391
    # or
    python3 -m locus.mcp.http_server --store /path/.locus --port 7391   # Windows: py -3 -m ...

Remote call example:
    curl -s -X POST http://localhost:7391/rpc \\
         -H "Content-Type: application/json" \\
         -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"locus_status","arguments":{}}}'
"""

import argparse
import hmac
import ipaddress
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..core import LocusEngine, __version__

logger = logging.getLogger(__name__)


DEFAULT_HOST = "127.0.0.1"


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _strip_port(hostport: str) -> str:
    hostport = hostport.strip().lower()
    if hostport.startswith("["):  # [::1]:7391
        return hostport[1:].split("]", 1)[0]
    if hostport.count(":") == 1:
        return hostport.rsplit(":", 1)[0]
    return hostport


def _is_loopback_origin(origin: str) -> bool:
    scheme, sep, rest = origin.partition("://")
    if not sep or scheme not in ("http", "https"):
        return False
    return is_loopback(_strip_port(rest.split("/", 1)[0]))


def _build_handler(
    engine: LocusEngine,
    token: str | None = None,
    cors_origin: str | None = None,
) -> type:
    """Return a handler class closed over engine, token and CORS origin."""

    class _Handler(BaseHTTPRequestHandler):
        _engine = engine
        _token = token
        _cors_origin = cors_origin

        # ----------------------------------------------------------------
        # Auth
        # ----------------------------------------------------------------

        def _authorized(self) -> bool:
            if not self._token:
                return True
            auth = self.headers.get("Authorization", "")
            return hmac.compare_digest(auth.encode(), f"Bearer {self._token}".encode())

        def _request_allowed(self) -> bool:
            """Block DNS-rebinding and cross-site requests.

            Without a token the server only trusts loopback ``Host`` headers, so a
            page on an attacker-controlled hostname that re-resolves to 127.0.0.1
            cannot reach the tools.  A browser ``Origin`` header, when present, must
            be the configured CORS origin or a loopback origin.
            """
            if not self._token:
                host = self.headers.get("Host", "")
                if not host or not is_loopback(_strip_port(host)):
                    return False
            origin = self.headers.get("Origin")
            if origin is not None:
                if self._cors_origin and origin == self._cors_origin:
                    return True
                return _is_loopback_origin(origin)
            return True

        def _send_cors(self) -> None:
            if self._cors_origin:
                self.send_header("Access-Control-Allow-Origin", self._cors_origin)
                self.send_header("Vary", "Origin")

        def _send_json(self, data: dict, status: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)

        # ----------------------------------------------------------------
        # Routes
        # ----------------------------------------------------------------

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            if self._cors_origin:
                self._send_cors()
                self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json({"ok": True, "version": __version__})
            elif self.path == "/tools":
                if not self._request_allowed():
                    self._send_json({"error": "Forbidden host or origin"}, 403)
                    return
                if not self._authorized():
                    self._send_json({"error": "Unauthorized"}, 401)
                    return
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
            if not self._request_allowed():
                self._send_json({"error": "Forbidden host or origin"}, 403)
                return
            if not self._authorized():
                self._send_json({"error": "Unauthorized"}, 401)
                return

            if self.path != "/rpc":
                self._send_json({"error": "Not found"}, 404)
                return

            # A JSON content type forces a CORS preflight for cross-site browser
            # requests, so pages on other origins cannot fire blind tool calls.
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype != "application/json":
                self._send_json({"error": "Content-Type must be application/json"}, 415)
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
            tool_result = _call_tool(params.get("name", ""), params.get("arguments", {}), engine=engine)
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
        elif method == "ping":
            result = {}
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


def make_server(
    engine: LocusEngine,
    host: str = DEFAULT_HOST,
    port: int = 7391,
    token: str | None = None,
    cors_origin: str | None = None,
) -> HTTPServer:
    """Build (but do not start) the HTTP server, enforcing the bind policy."""
    if not is_loopback(host) and not token:
        raise ValueError(
            f"Refusing to bind {host!r} without a token: pass --token for any "
            "non-loopback host, or bind 127.0.0.1"
        )
    if cors_origin == "*":
        raise ValueError("Wildcard CORS is not allowed; pass an explicit origin")
    handler = _build_handler(engine, token=token, cors_origin=cors_origin)
    return HTTPServer((host, port), handler)


def serve(
    engine: LocusEngine,
    host: str = DEFAULT_HOST,
    port: int = 7391,
    token: str | None = None,
    cors_origin: str | None = None,
) -> None:
    """Start the HTTP server (blocking)."""
    server = make_server(engine, host=host, port=port, token=token, cors_origin=cors_origin)
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
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=7391)
    parser.add_argument("--token", default=None, help="Bearer token (required for non-loopback hosts)")
    parser.add_argument("--cors-origin", default=None, help="Allowed browser origin (CORS off by default)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    serve(
        LocusEngine(store_path=args.store),
        host=args.host,
        port=args.port,
        token=args.token,
        cors_origin=args.cors_origin,
    )
