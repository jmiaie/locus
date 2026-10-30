"""Gemini gateway (127.0.0.1:8766), free tier only, multi-key rotation. Keys are never logged or written;
they are referred to by env var name (GEMINI_API_KEY, GEMINI_API_KEY_2, ...), re-read at start.
  /gm/<tag>/v1/...   -> https://generativelanguage.googleapis.com/v1beta/openai/...
  /emb/v1/embeddings -> local sentence-transformers all-MiniLM-L6-v2 (384-d), same as gateway.py.
Per key: free-tier probe at start (gemini-4-argon has 0 free quota -> a free key gets 429 with only *-FreeTier quotaIds;
anything else -> key excluded). Throttle per quota pool (keys found to share a project are merged into one pool):
min interval 60/GM_RPM s and a sliding 60 s input-token budget GM_TPM. 429 per-minute -> cool that pool for retryDelay;
429 per-day -> pool exhausted until 00:03 America/Los_Angeles, move to another pool; if all pools are exhausted the
request is held until the earliest reset. Spend guard: any non-FreeTier quotaId or billing signal -> GEMINI_STOP, all
further requests get 402. Shared-project detection: a key whose first daily-429 comes after < 50 own calls today while
another pool hit its daily limit within the last 10 min is merged into that pool (logged + gemini_quota_state.json)."""
import os, json, hashlib, sqlite3, threading, time, urllib.request, urllib.error, datetime, re
from zoneinfo import ZoneInfo
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch; torch.set_num_threads(2)
from sentence_transformers import SentenceTransformer
EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"); EL = threading.Lock()
UP = "https://generativelanguage.googleapis.com/v1beta/openai"
ROOT = "/workspace/llmbench"; STOP = ROOT + "/GEMINI_STOP"; STATE = ROOT + "/gemini_quota_state.json"
RPM = float(os.environ.get("GM_RPM", "13")); TPM = float(os.environ.get("GM_TPM", "230000"))
PT = ZoneInfo("America/Los_Angeles")
NAMES = sorted([k for k in os.environ if re.fullmatch(r"GEMINI_API_KEY(_\d+)?", k)], key=lambda k: int(k.rsplit("_", 1)[1]) if k[-1].isdigit() else 1)
KEYS = {n: os.environ.pop(n) for n in NAMES}  # secrets live only in this dict
def log(*a): print(datetime.datetime.now().isoformat(timespec="seconds"), *a, flush=True)
def day0(): return datetime.datetime.now(PT).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
def next_reset():
    n = datetime.datetime.now(PT); return (n + datetime.timedelta(days=1)).replace(hour=0, minute=3, second=0, microsecond=0).timestamp()
db = sqlite3.connect(ROOT + "/gateway_gemini.sqlite", check_same_thread=False); DL = threading.Lock()
db.execute("create table if not exists cache (k text primary key, status int, body blob)")
db.execute("create table if not exists log (ts real, tag text, path text, model text, cached int, status int, ms real, pt int, ct int, cost real, note text)")
if "key_name" not in [r[1] for r in db.execute("pragma table_info(log)")]: db.execute("alter table log add column key_name text")
db.commit()
def calls_today(name):
    with DL:
        q = "select count(*) from log where cached=0 and status=200 and ts>=? and (key_name=? or (key_name is null and ?='GEMINI_API_KEY'))"
        return db.execute(q, (day0(), name, name)).fetchone()[0]
# ---- persistent quota state ----
try: ST = json.load(open(STATE))
except Exception: ST = {}
ST.setdefault("keys", {}); ST.setdefault("pools", {}); ST.setdefault("events", [])
def save():
    json.dump(ST, open(STATE + ".tmp", "w"), indent=1); os.replace(STATE + ".tmp", STATE)
# legacy single-key state: GEMINI_API_KEY hit its daily limit earlier today
if ST.get("sleeping_until") and "GEMINI_API_KEY" in KEYS and ST["sleeping_until"] > time.time():
    ST["keys"].setdefault("GEMINI_API_KEY", {})["exhausted_until"] = ST.pop("sleeping_until")
ML = threading.Lock()
if ST.get("daily_exhausted_at") and ST["daily_exhausted_at"] >= day0() and not any(e.get("key") == "GEMINI_API_KEY" for e in ST["events"]):
    ST["events"].insert(0, {"ts": ST["daily_exhausted_at"], "event": "daily", "key": "GEMINI_API_KEY", "own_calls_today": ST.get("calls_today_at_exhaustion", 0), "quota": "legacy single-key gateway"})
def root(n):
    while ST["pools"].get(n, n) != n: n = ST["pools"][n]
    return n
def merge(a, b, why):
    ra, rb = root(a), root(b)
    if ra == rb: return
    ST["pools"][rb] = ra; ST["events"].append({"ts": time.time(), "event": "merge", "keys": [a, b], "why": why}); save()
    log(f"POOL MERGE: {b} shares a quota pool with {a} ({why})")
P = {}  # per pool root: runtime throttle {"next": ts, "win": [(ts, tok)], "cool": ts}
def pool(n):
    r = root(n); return P.setdefault(r, {"next": 0.0, "win": [], "cool": 0.0})
def exhausted(n):
    r = root(n); return ST["keys"].get(r, {}).get("exhausted_until", 0) > time.time() or ST["keys"].get(n, {}).get("exhausted_until", 0) > time.time()
# ---- free-tier probe ----
def probe(n):
    b = json.dumps({"model": "gemini-4-argon", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(UP + "/chat/completions", data=b, method="POST", headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEYS[n]}), timeout=60)
        return False, "200 on a model with no free quota -> paid/billing project"
    except urllib.error.HTTPError as e:
        d = e.read().decode(errors="replace"); ids = set(re.findall(r'"quotaId":\s*"([^"]+)"', d))
        if e.code == 429 and ids and all("FreeTier" in q for q in ids): return True, "free tier"
        return False, f"HTTP {e.code}, quotaIds {sorted(ids)}"
    except Exception as e: return False, f"probe error {type(e).__name__}"
def replay_merges():  # apply the shared-project rule to today's daily-limit events (restart-safe)
    ev = [e for e in ST["events"] if e.get("event") == "daily" and e["ts"] >= day0()]
    for i, b in enumerate(ev):
        for a in ev[:i]:
            if root(a["key"]) == root(b["key"]): continue
            if abs(a["own_calls_today"] + b["own_calls_today"] - 500) <= 30 and (b["own_calls_today"] < 50 or b["ts"] - a["ts"] < 300):
                merge(a["key"], b["key"], f"replay: own calls {a['own_calls_today']} ({a['key']}) + {b['own_calls_today']} ({b['key']}) ~= 500")
replay_merges()
ACTIVE = []
for n in NAMES:
    ok, why = probe(n); ST["keys"].setdefault(n, {})["probe"] = why
    log(f"key {n}: {why} -> {'USE' if ok else 'EXCLUDED'}")
    if ok: ACTIVE.append(n)
save()
if not ACTIVE: log("no usable free-tier keys"); open(STOP, "w").write("no free-tier keys")
def quota_info(data):
    try: err = json.loads(data); err = err[0] if isinstance(err, list) else err; err = err.get("error", err)
    except Exception: return [], None, data[:300].decode(errors="replace")
    ids, delay = [], None
    for d in err.get("details", []) or []:
        for v in d.get("violations", []) or []: ids.append((v.get("quotaId", ""), v.get("quotaValue")))
        if d.get("retryDelay"): delay = float(re.sub(r"[^\d.]", "", d["retryDelay"]) or 0)
    if not ids: ids = [(m, None) for m in re.findall(r"quotaId[\"':\s]+([A-Za-z\-]+)", json.dumps(err))]
    return ids, delay, str(err.get("message", ""))[:300]
def pick(est):
    """reserve a slot on the best available pool; returns (key_name, wait_s) or (None, wait_s)"""
    with ML:
        now = time.time(); best = None
        for n in ACTIVE:
            if exhausted(n): continue
            p = pool(n); p["win"][:] = [w for w in p["win"] if w[0] > now - 60]
            t = max(p["next"], p["cool"], now)
            if sum(w[1] for w in p["win"]) + est > TPM and p["win"]: t = max(t, p["win"][0][0] + 60.5)
            # among keys of the same pool prefer the one with fewest own calls (spreads per-key counters)
            if best is None or t < best[1] - 1e-6: best = (n, t)
        if best is None:
            us = [max(ST["keys"].get(n, {}).get("exhausted_until", 0), ST["keys"].get(root(n), {}).get("exhausted_until", 0)) for n in ACTIVE]
            return None, max(30.0, min(us) - now if us else 60.0), None
        n, t = best; p = pool(n); p["next"] = t + 60.0 / RPM; ent = [t, est]; p["win"].append(ent)
        return n, t - now, ent
def upstream(rest, body, est):
    attempt = 0
    while True:
        if os.path.exists(STOP): return 402, json.dumps({"error": {"message": "gemini gateway stopped (spend guard)"}}).encode(), "stopped", None
        n, wait, ent = pick(est)
        if n is None:
            log(f"all pools exhausted; holding request ({wait/3600:.2f} h to earliest reset)"); time.sleep(min(wait, 600)); continue
        if wait > 0: time.sleep(wait)
        try:
            rq = urllib.request.Request(UP + rest, data=body, method="POST", headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEYS[n]})
            with urllib.request.urlopen(rq, timeout=300) as resp:
                data = resp.read()
                try: ent[1] = int((json.loads(data).get("usage") or {}).get("prompt_tokens") or ent[1])
                except Exception: pass
                return resp.status, data, "", n
        except urllib.error.HTTPError as e: status, data = e.code, e.read()
        except Exception as e: status, data = 599, json.dumps({"error": str(e)[:200]}).encode()
        attempt += 1
        if status == 429:
            ids, delay, msg = quota_info(data); qs = " ".join(f"{q}={v}" for q, v in ids) or msg
            if (ids and not all("FreeTier" in q for q, _ in ids)) or (not ids and re.search(r"charged|prepay|credits? (are )?depleted", msg, re.I)):
                log(f"SPEND GUARD: non-free-tier 429 on {n}:", qs); open(STOP, "w").write(f"{n}: {qs}"); continue
            if any("PerDay" in q for q, _ in ids):
                with ML:
                    own = calls_today(n); now = time.time(); r = next_reset()
                    recent = [e for e in ST["events"] if e.get("event") == "daily" and now - e["ts"] < 600 and root(e["key"]) != root(n) and day0() <= e["ts"]]
                    lim = int(next((v for q, v in ids if "PerDay" in q and v), 500) or 500)
                    if own < 50 and recent: merge(recent[-1]["key"], n, f"daily 429 after only {own} own calls, {int(now-recent[-1]['ts'])} s after {recent[-1]['key']} hit its daily limit")
                    else:
                        # another key already exhausted today whose own count + this key's own count ~= the daily limit -> same project
                        for e in ST["events"]:
                            if e.get("event") == "daily" and e["ts"] >= day0() and root(e["key"]) != root(n) and abs(e["own_calls_today"] + own - lim) <= 0.06 * lim and (own < 50 or now - e["ts"] < 300):
                                merge(e["key"], n, f"own calls {e['own_calls_today']} ({e['key']}) + {own} ({n}) ~= daily limit {lim}"); break
                    ST["events"].append({"ts": now, "event": "daily", "key": n, "own_calls_today": own, "quota": qs}); ST["keys"].setdefault(n, {})["exhausted_until"] = r
                    ST["keys"][n]["calls_at_exhaustion"] = own; save()
                log(f"daily quota exhausted on {n} ({qs}) after {own} own calls today; parked until", datetime.datetime.fromtimestamp(r, PT).isoformat())
                continue
            d = (delay or 20) + 2
            with ML: pool(n)["cool"] = time.time() + d
            log(f"429 per-minute on {n}", qs, "pool cooled", d); continue
        if status == 402 or (status == 403 and re.search(rb"billing", data, re.I)):
            log("SPEND GUARD: status", status, "on", n); open(STOP, "w").write(f"{n}: HTTP {status}"); return status, data, "billing", n
        if status in (500, 502, 503, 504, 599) and attempt < 12:
            d = min(300, 5 * 2 ** min(attempt, 6)); log("upstream", status, data[:150], "on", n, "retry in", d); time.sleep(d); continue
        return status, data, f"attempts={attempt}", n
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def send(self, status, body, ctype="application/json"):
        self.send_response(status); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith("/emb/v1/models"): return self.send(200, json.dumps({"data": [{"id": "all-MiniLM-L6-v2", "object": "model"}]}).encode())
        if self.path.startswith("/health"):
            with ML:
                pools = {}
                for n in ACTIVE: pools.setdefault(root(n), []).append(n)
                info = {"ok": True, "stopped": os.path.exists(STOP), "active_keys": ACTIVE,
                        "pools": {r: {"keys": ks, "exhausted": exhausted(r), "calls_today": sum(calls_today(k) for k in ks)} for r, ks in pools.items()}}
                info["available_pools"] = sum(1 for v in info["pools"].values() if not v["exhausted"])
            return self.send(200, json.dumps(info).encode())
        self.send(404, b"{}")
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n)
        if self.path.startswith("/emb/"):
            req = json.loads(body); inp = req["input"]; inp = [inp] if isinstance(inp, str) else inp
            with EL: v = EMB.encode(inp, batch_size=32, normalize_embeddings=True).tolist()
            out = {"object": "list", "model": "all-MiniLM-L6-v2", "data": [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(v)], "usage": {"prompt_tokens": 0, "total_tokens": 0}}
            return self.send(200, json.dumps(out).encode())
        if not self.path.startswith("/gm/"): return self.send(404, b"{}")
        tag, rest = self.path[4:].split("/", 1); rest = rest[2:] if rest.startswith("v1") else rest
        j = json.loads(body)
        if j.get("stream"): return self.send(400, b'{"error":{"message":"streaming not supported by gemini gateway"}}')
        m = str(j.get("model", ""))
        if "reasoning_effort" not in j:
            if m.startswith("gemini-2.5"): j["reasoning_effort"] = "none"
            elif m.startswith("gemini-3"): j["reasoning_effort"] = "minimal"  # 3.x rejects "none"; minimal = lowest thinking level
        for p in ("logit_bias", "logprobs", "top_logprobs", "seed", "frequency_penalty", "presence_penalty"): j.pop(p, None)
        body = json.dumps(j).encode()
        k = hashlib.sha256(rest.encode() + b"\0" + body).hexdigest()   # cache key independent of which API key serves it
        with DL: r = db.execute("select status, body from cache where k=?", (k,)).fetchone()
        if r:
            with DL: db.execute("insert into log values (?,?,?,?,?,?,?,?,?,?,?,?)", (time.time(), tag, rest, m, 1, r[0], 0, 0, 0, 0.0, "", None)); db.commit()
            return self.send(r[0], r[1])
        est = len(body) // 3 + 100
        t = time.perf_counter(); status, data, note, kn = upstream(rest, body, est); ms = (time.perf_counter() - t) * 1000; pt = ct = 0
        try: u = json.loads(data).get("usage") or {}; pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        except Exception: pass
        with DL:
            db.execute("insert into log values (?,?,?,?,?,?,?,?,?,?,?,?)", (time.time(), tag, rest, m, 0, status, ms, pt, ct, 0.0, note, kn))
            if status == 200: db.execute("insert or replace into cache values (?,?,?)", (k, status, data))
            db.commit()
        if status != 200: log("non-200 to client", status, data[:200])
        self.send(status, data)
ThreadingHTTPServer.daemon_threads = True
log("gemini gateway up; keys", NAMES, "active", ACTIVE, "rpm/pool", RPM, "tpm/pool", TPM)
ThreadingHTTPServer(("127.0.0.1", 8766), H).serve_forever()
