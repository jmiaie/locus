"""Re-measure the IMPLEMENTED opt-in ranking features (not the prototypes) on LongMemEval-S.
usage: remeasure.py {locus|ompa} [SHARD NSHARDS]. Same data/config as expansion-20261112/run500.py, except each session doc
carries its session date (Locus: YAML frontmatter `date:`; OMPA: write(date=...)) so date-aware ranking can use it.
Writes rm__SYSTEM.shardK.jsonl: one row per question with the top-10 sessions for every variant. Resumable."""
import sys, json, re, time, tempfile, gc
from pathlib import Path
SYSTEM = sys.argv[1]; SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0; NS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
H = Path(__file__).resolve().parent; EXP = Path("/workspace/llmbench/results/expansion-20261112")
DATA = EXP / "lme_s_500.jsonl"; TMP = Path("/workspace/rank-20261112/tmp")
OUT = H / f"rm__{SYSTEM}.shard{SHARD}.jsonl"
REPO = Path("/workspace/rank-20261112") / SYSTEM
sys.path.insert(0, str(REPO))
import torch; torch.set_num_threads(2)
def dedup(sids, k=10):
    out = []
    for s in sids:
        if s and s not in out: out.append(s)
        if len(out) == k: break
    return out
def iso(d): return d.split(" (")[0].replace("/", "-")
def sess(e):
    return [(sid, d, "\n".join(f"{t['role']}: {t['content']}" for t in s)) for sid, d, s in zip(e["haystack_session_ids"], e["haystack_dates"], e["haystack_sessions"])]
_DR = None
def run_locus(e, wd):
    from locus.core import LocusEngine
    from locus.dense import DenseReranker
    dd = Path(wd) / "docs"; dd.mkdir(); back = {}
    for sid, d, txt in sess(e):
        k = re.sub(r"[^\w-]", "_", sid); back[k] = sid
        (dd / f"{k}.md").write_text(f"---\ndate: {iso(d)}\n---\n{txt}")
    q, ref = e["question"], iso(e["question_date"])
    V = {}; ms = {}
    def go(name, eng, **kw):
        t = time.perf_counter(); ch = eng.retrieve(q, limit=100, use_cache=False, mode="bm25", **kw); ms[name] = (time.perf_counter() - t) * 1000
        V[name] = dedup([back.get(Path(c.doc_path).stem) for c in ch])
    eng = LocusEngine(store_path=Path(wd) / ".locus"); eng.index(dd)
    go("baseline", eng); go("temporal", eng, temporal=True, reference_date=ref)
    global _DR
    if _DR is None: _DR = {r: DenseReranker(route=r, threads=2) for r in (None, "advice")}
    if _DR["advice"]._st is None: _DR["advice"]._st = _DR[None]._load()  # share one loaded model (same weights)
    for route, tag in ((None, "dense"), ("advice", "dense_advice")):
        eng2 = LocusEngine(store_path=Path(wd) / ".locus", reranker=_DR[route])  # binds the reranker to this engine
        go(tag, eng2, rerank=True)
        if route is None: go("dense_temporal", eng2, rerank=True, temporal=True, reference_date=ref)
        try: eng2.close()
        except Exception: pass
    try: eng.close()
    except Exception: pass
    return V, ms
_M = None
def run_ompa(e, wd):
    global _M
    from ompa import Ompa
    if _M is None:
        from sentence_transformers import SentenceTransformer
        _M = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    ao = Ompa(vault_path=str(Path(wd) / "vault"), enable_semantic=True, embedding_backend=_M); back = {}
    for sid, d, txt in sess(e):
        k = re.sub(r"[^\w-]", "_", sid); back[k] = sid; ao.write(content=txt, file_path=f"work/docs/{k}.md", tags=["bench"], date=iso(d))
    ao.rebuild_index()
    q, ref = e["question"], iso(e["question_date"]); V = {}; ms = {}
    for name, kw in (("baseline", {}), ("temporal", {"temporal": True, "reference_date": ref})):
        t = time.perf_counter(); rs = ao.search(q, limit=10, **kw); ms[name] = (time.perf_counter() - t) * 1000
        V[name] = dedup([back.get(Path(r.path).stem) for r in rs])
    try: ao.close()
    except Exception: pass
    return V, ms
done = set()
if OUT.exists():
    for l in open(OUT):
        try: r = json.loads(l)
        except Exception: continue
        if "error" not in r: done.add(r["qid"])
fn = globals()["run_" + SYSTEM]; T0 = time.time(); n = 0
with open(DATA) as f:
    for i, line in enumerate(f):
        if i % NS != SHARD: continue
        e = json.loads(line); qid = e["question_id"]
        if qid in done or qid.endswith("_abs"): continue
        try:
            with tempfile.TemporaryDirectory(dir=TMP) as wd:
                V, ms = fn(e, wd)
            row = {"qid": qid, "qtype": e["question_type"], "answers": e["answer_session_ids"], "variants": V, "ms": ms}
        except Exception as ex:
            import traceback; traceback.print_exc(); row = {"qid": qid, "error": f"{type(ex).__name__}: {str(ex)[:300]}"}
        with open(OUT, "a") as o: o.write(json.dumps(row) + "\n")
        del e; gc.collect(); n += 1
        print(SYSTEM, SHARD, n, qid, row.get("error") or {k: round(v, 1) for k, v in row["ms"].items()}, f"{time.time()-T0:.0f}s", flush=True)
print(SYSTEM, SHARD, "DONE", n, f"wall {time.time()-T0:.0f}s", flush=True)
