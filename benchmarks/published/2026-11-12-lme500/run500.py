"""Full LongMemEval-S (500 q) run for Locus / OMPA with configs IDENTICAL to p2_run.py run_locus/run_ompa (LME30).
usage: run500.py {locus|ompa} SHARD NSHARDS   -> appends to lme500__SYSTEM.shardK.jsonl, resumes by skipping done qids.
Single long-lived process per shard (OMPA's MiniLM model is loaded once, outside the timed region, as in p2_run.py)."""
import sys, json, re, time, tempfile, os, gc
from pathlib import Path
SYSTEM, SHARD, NS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
HERE = Path(__file__).resolve().parent; DATA = HERE / "lme_s_500.jsonl"; TMP = Path("/workspace/llmbench/tmp")
OUT = HERE / f"lme500__{SYSTEM}.shard{SHARD}.jsonl"
def dedup_sessions(sids, k=10):
    out = []
    for s in sids:
        if s and s not in out: out.append(s)
        if len(out) == k: break
    return out
def _sess_docs(e):
    return [(sid, "\n".join(f"{t['role']}: {t['content']}" for t in sess)) for sid, sess in zip(e["haystack_session_ids"], e["haystack_sessions"])]
def run_locus(e, wd):
    sys.path.insert(0, "/workspace/scorecard-20261020/repos/locus")
    from locus.core import LocusEngine
    dd = Path(wd) / "docs"; dd.mkdir(); back = {}
    for sid, txt in _sess_docs(e):
        k = re.sub(r"[^\w-]", "_", sid); back[k] = sid; (dd / f"{k}.md").write_text(txt)
    t0 = time.perf_counter(); eng = LocusEngine(store_path=Path(wd) / ".locus"); eng.index(dd); ing = time.perf_counter() - t0
    t = time.perf_counter(); ch = eng.retrieve(e["question"], limit=100, use_cache=False, mode="bm25"); qms = (time.perf_counter() - t) * 1000
    # second timed query (warm) to report steady-state latency separately; ranking taken from the first call
    t = time.perf_counter(); eng.retrieve(e["question"], limit=100, use_cache=False, mode="bm25"); qms2 = (time.perf_counter() - t) * 1000
    try: eng.close()
    except Exception: pass
    return dedup_sessions([back.get(Path(c.doc_path).stem) for c in ch]), ing, qms, {"ms_warm": qms2}
_M = None
def run_ompa(e, wd):
    global _M
    sys.path.insert(0, "/workspace/scorecard-20261020/repos/ompa")
    from ompa import Ompa
    if _M is None:
        import torch; torch.set_num_threads(2)
        from sentence_transformers import SentenceTransformer
        _M = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    t0 = time.perf_counter()
    ao = Ompa(vault_path=str(Path(wd) / "vault"), enable_semantic=True, embedding_backend=_M); back = {}
    for sid, txt in _sess_docs(e):
        k = re.sub(r"[^\w-]", "_", sid); back[k] = sid; ao.write(content=txt, file_path=f"work/docs/{k}.md", tags=["bench"])
    ao.rebuild_index(); ing = time.perf_counter() - t0
    t = time.perf_counter(); rs = ao.search(e["question"], limit=10); qms = (time.perf_counter() - t) * 1000
    t = time.perf_counter(); ao.search(e["question"], limit=10); qms2 = (time.perf_counter() - t) * 1000
    try: ao.close()
    except Exception: pass
    return dedup_sessions([back.get(Path(r.path).stem) for r in rs]), ing, qms, {"ms_warm": qms2}
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
        if qid in done: continue
        try:
            with tempfile.TemporaryDirectory(dir=TMP) as wd:
                ranked, ing, qms, extra = fn(e, wd)
            row = {"qid": qid, "qtype": e["question_type"], "abstention": qid.endswith("_abs"), "ranked": ranked, "answers": e["answer_session_ids"],
                   "ms": qms, "ingest_s": round(ing, 3), "n_sessions": len(e["haystack_session_ids"]), **extra}
        except Exception as ex:
            import traceback; traceback.print_exc(); row = {"qid": qid, "error": f"{type(ex).__name__}: {str(ex)[:300]}"}
        with open(OUT, "a") as o: o.write(json.dumps(row) + "\n")
        del e; gc.collect(); n += 1
        print(SYSTEM, SHARD, n, qid, row.get("error") or (round(row["ingest_s"], 2), round(row["ms"], 1)), f"{time.time()-T0:.0f}s", flush=True)
print(SYSTEM, SHARD, "DONE", n, f"wall {time.time()-T0:.0f}s", flush=True)
