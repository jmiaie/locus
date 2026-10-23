"""Disk-cached OpenAI-compatible transport for locus.llm (key read from env, never written)."""
import json, hashlib, sqlite3, threading, time, urllib.request
DB = "/workspace/llmbench/llm_cache.sqlite"
_lock = threading.Lock()
_con = sqlite3.connect(DB, check_same_thread=False)
_con.execute("create table if not exists c (k text primary key, resp text, ms real, ts real)"); _con.commit()
STATS = {"hits": 0, "misses": 0, "ms": []}
def transport(url, headers, body, timeout):
    k = hashlib.sha256(url.encode() + b"\0" + body).hexdigest()
    with _lock:
        r = _con.execute("select resp, ms from c where k=?", (k,)).fetchone()
    if r:
        STATS["hits"] += 1; STATS["ms"].append(r[1]); return json.loads(r[0])
    for attempt in range(5):
        t = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read().decode())
            if "choices" not in d: raise RuntimeError(str(d)[:200])
            break
        except Exception:
            if attempt == 4: raise
            time.sleep(2 ** attempt)
    ms = (time.perf_counter() - t) * 1000
    with _lock:
        _con.execute("insert or replace into c values (?,?,?,?)", (k, json.dumps(d), ms, time.time())); _con.commit()
    STATS["misses"] += 1; STATS["ms"].append(ms); return d
