"""Local gateway (127.0.0.1:8765):
  /or/<tag>/v1/...   -> forwards to https://openrouter.ai/api/v1/... adding the key from env (clients use a dummy key),
                        disk-caches non-streaming responses, logs per-request usage/cost tagged by system.
  /emb/v1/embeddings -> OpenAI-compatible embeddings from local sentence-transformers all-MiniLM-L6-v2 (384-d).
The key is never logged or written."""
import os, json, hashlib, sqlite3, threading, time, urllib.request, urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch; torch.set_num_threads(2)
from sentence_transformers import SentenceTransformer
EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"); EL = threading.Lock()
KEY = os.environ["OPENROUTER_API_KEY"]; UP = "https://openrouter.ai/api/v1"
db = sqlite3.connect("/workspace/llmbench/gateway.sqlite", check_same_thread=False); DL = threading.Lock()
db.execute("create table if not exists cache (k text primary key, status int, body blob)")
db.execute("create table if not exists log (ts real, tag text, path text, model text, cached int, status int, ms real, pt int, ct int, cost real)"); db.commit()
BUDGET_STOP = float(os.environ.get("GW_BUDGET_STOP", "24.0"))
def spent():
    with DL: return db.execute("select coalesce(sum(cost),0) from log").fetchone()[0]
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def send(self, status, body, ctype="application/json"):
        self.send_response(status); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith("/emb/v1/models"): return self.send(200, json.dumps({"data": [{"id": "all-MiniLM-L6-v2", "object": "model"}]}).encode())
        if self.path.startswith("/or/"):
            tag, rest = self.path[4:].split("/", 1); rest = rest[2:] if rest.startswith("v1") else rest
            try:
                r = urllib.request.urlopen(urllib.request.Request(UP + rest, headers={"Authorization": "Bearer " + KEY}), timeout=60); return self.send(r.status, r.read())
            except urllib.error.HTTPError as e: return self.send(e.code, e.read())
        self.send(404, b"{}")
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n)
        if self.path.startswith("/emb/"):
            req = json.loads(body); inp = req["input"]; inp = [inp] if isinstance(inp, str) else inp
            with EL: v = EMB.encode(inp, batch_size=32, normalize_embeddings=True).tolist()
            out = {"object": "list", "model": "all-MiniLM-L6-v2", "data": [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(v)],
                   "usage": {"prompt_tokens": 0, "total_tokens": 0}}
            return self.send(200, json.dumps(out).encode())
        if not self.path.startswith("/or/"): return self.send(404, b"{}")
        tag, rest = self.path[4:].split("/", 1); rest = rest[2:] if rest.startswith("v1") else rest
        try: j = json.loads(body)
        except Exception: j = {}
        stream = bool(j.get("stream"))
        k = hashlib.sha256(rest.encode() + b"\0" + body).hexdigest()
        if not stream:
            with DL: r = db.execute("select status, body from cache where k=?", (k,)).fetchone()
            if r:
                with DL: db.execute("insert into log values (?,?,?,?,?,?,?,?,?,?)", (time.time(), tag, rest, j.get("model"), 1, r[0], 0, 0, 0, 0.0)); db.commit()
                return self.send(r[0], r[1])
        if spent() >= BUDGET_STOP:
            return self.send(402, json.dumps({"error": {"message": "local gateway budget stop reached"}}).encode())
        if stream:
            j.setdefault("stream_options", {})["include_usage"] = True; body = json.dumps(j).encode()
        t = time.perf_counter(); status = 599; data = b""
        for attempt in range(4):
            try:
                rq = urllib.request.Request(UP + rest, data=body, method="POST", headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
                with urllib.request.urlopen(rq, timeout=180) as resp: status, data = resp.status, resp.read()
                break
            except urllib.error.HTTPError as e:
                status, data = e.code, e.read()
                if e.code not in (429, 500, 502, 503, 504): break
            except Exception as e:
                status, data = 599, json.dumps({"error": str(e)[:200]}).encode()
            time.sleep(2 ** attempt)
        ms = (time.perf_counter() - t) * 1000; pt = ct = 0; cost = 0.0
        try:
            if stream:
                for line in data.decode().splitlines():
                    if line.startswith("data: {"):
                        u = json.loads(line[6:]).get("usage")
                        if u: pt, ct, cost = u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cost", 0.0)
            else:
                u = json.loads(data).get("usage") or {}; pt, ct, cost = u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cost", 0.0)
        except Exception: pass
        with DL:
            db.execute("insert into log values (?,?,?,?,?,?,?,?,?,?)", (time.time(), tag, rest, j.get("model"), 0, status, ms, pt, ct, cost or 0.0))
            if status == 200 and not stream: db.execute("insert or replace into cache values (?,?,?)", (k, status, data))
            db.commit()
        self.send(status, data, "text/event-stream" if stream else "application/json")
ThreadingHTTPServer.daemon_threads = True
print("gateway up", flush=True); ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
