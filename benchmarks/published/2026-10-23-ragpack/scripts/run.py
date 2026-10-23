"""RAG feature-pack benchmark harness (2026-10).
usage: run.py NAME BENCH [--cfg JSON] [--splits dev,test] [--limit N]
BENCH: scifact | nfcorpus (dev and/or test, one index) | lme (LongMemEval-S, 470 q) | construction (21 q)
cfg = {"chunking": {...}, "retrieve": {... extra retrieve() kwargs ...}, "rerank": {"model":..., "depth":...}}
Writes results/<bench>_<split>__<NAME>.jsonl rows {qid, ranked(doc ids, top 10), ms} + .meta.json."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(v, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false"); os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import sys, re, json, time, csv, tempfile, logging, argparse, resource
from pathlib import Path
logging.disable(logging.WARNING)
ROOT = Path("/workspace/ragpack"); RES = ROOT / "results"; TMP = ROOT / "tmp"
DATA = Path("/workspace/ompa-locus-20261020/data"); LME = Path("/workspace/stem/data/lme_s_470.jsonl")
CONS = Path("/workspace/scorecard-20261020/repos/locus/examples/construction")
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("bench"); ap.add_argument("--cfg", default="{}")
ap.add_argument("--splits", default="dev,test"); ap.add_argument("--limit", type=int)
A = ap.parse_args(); CFG = json.loads(A.cfg)
from locus.core import LocusEngine
RET = dict(CFG.get("retrieve", {})); CHUNK_LIMIT = RET.pop("chunk_limit", 100)
safe = lambda s: re.sub(r"[^\w-]", "_", s)
reranker = None
if CFG.get("rerank"):
    from locus.rerank import CrossEncoderReranker
    reranker = CrossEncoderReranker(**CFG["rerank"])

def engine(wd):
    return LocusEngine(store_path=Path(wd) / ".locus", chunking=CFG.get("chunking"), reranker=reranker) if reranker \
        else LocusEngine(store_path=Path(wd) / ".locus", chunking=CFG.get("chunking"))

def ranked_docs(eng, back, q, k=10, **extra):
    kw = {"limit": CHUNK_LIMIT, "use_cache": False, "mode": "bm25", **RET, **extra}
    out = []
    for c in eng.retrieve(q, **kw):
        d = back(c.doc_path)
        if d not in out: out.append(d)
        if len(out) == k: break
    return out

class Out:
    def __init__(s, key):
        s.path = RES / f"{key}__{A.name}.jsonl"; s.f = open(s.path, "w"); s.meta = {"key": key, "name": A.name, "cfg": CFG}
    def row(s, **kw): s.f.write(json.dumps(kw) + "\n")
    def close(s, **m):
        s.meta.update(m); s.meta["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        json.dump(s.meta, open(str(s.path).replace(".jsonl", ".meta.json"), "w"), indent=1); s.f.close()

def dir_mb(p): return round(sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 2**20, 2)

if A.bench in ("scifact", "nfcorpus"):
    d = DATA / A.bench
    queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(d / "queries.jsonl")}
    with tempfile.TemporaryDirectory(dir=TMP) as wd:
        dd = Path(wd) / "docs"; dd.mkdir(); back = {}
        for l in open(d / "corpus.jsonl"):
            c = json.loads(l); n = safe(c["_id"]); back[n] = c["_id"]
            (dd / f"{n}.md").write_text(f"# {c.get('title', '') or ''}\n\n{c['text']}" if c.get("title") else c["text"])
        eng = engine(wd); t0 = time.perf_counter(); info = eng.index(dd); idx = time.perf_counter() - t0
        bk = lambda p: back.get(Path(p).stem)
        for split in A.splits.split(","):
            qf = ("train" if A.bench == "scifact" else "dev") if split == "dev" else "test"
            qids = set()
            with open(d / "qrels" / f"{qf}.tsv") as f:
                r = csv.reader(f, delimiter="\t"); next(r)
                for qid, did, s in r:
                    if int(s) > 0 and qid in queries: qids.add(qid)
            qids = sorted(qids)[: A.limit] if A.limit else sorted(qids)
            out = Out(f"{A.bench}_{split}")
            for w in qids[:3]: ranked_docs(eng, bk, queries[w])
            for q in qids:
                t = time.perf_counter(); r = ranked_docs(eng, bk, queries[q]); ms = (time.perf_counter() - t) * 1000
                out.row(qid=q, ranked=r, ms=ms)
            out.close(index_s=round(idx, 2), store_mb=dir_mb(Path(wd) / ".locus"), chunks=info["chunks"], n=len(qids))
            print(A.name, A.bench, split, "n", len(qids), "idx", round(idx, 1), "chunks", info["chunks"], flush=True)
elif A.bench == "lme":
    out = Out("lme"); tot = 0.0; n = 0
    for n, l in enumerate(open(LME)):
        if A.limit and n >= A.limit: break
        e = json.loads(l)
        with tempfile.TemporaryDirectory(dir=TMP) as wd:
            dd = Path(wd) / "docs"; dd.mkdir(); back = {}
            for sid, sess in zip(e["haystack_session_ids"], e["haystack_sessions"]):
                k = safe(sid); back[k] = sid
                (dd / f"{k}.md").write_text("\n".join(f"{t['role']}: {t['content']}" for t in sess))
            eng = engine(wd); t0 = time.perf_counter(); eng.index(dd); tot += time.perf_counter() - t0
            t = time.perf_counter(); r = ranked_docs(eng, lambda p: back.get(Path(p).stem), e["question"]); ms = (time.perf_counter() - t) * 1000
            out.row(qid=e["question_id"], qtype=e["question_type"], ranked=r, answers=e["answer_session_ids"], ms=ms)
        if n % 100 == 0: print(A.name, "lme", n, flush=True)
    out.close(total_index_s=round(tot, 1), n=n + 1); print(A.name, "lme done", flush=True)
elif A.bench == "construction":
    qs = json.loads((CONS / "eval_questions.json").read_text())["questions"]
    with tempfile.TemporaryDirectory(dir=TMP) as wd:
        eng = engine(wd); t0 = time.perf_counter(); eng.index(CONS / "docs"); idx = time.perf_counter() - t0
        out = Out("construction")
        for q in qs:
            extra = {"filters": {"as_of": q["as_of"]}} if q.get("as_of") else {}
            t = time.perf_counter(); r = ranked_docs(eng, lambda p: p, q["query"], k=5, **extra); ms = (time.perf_counter() - t) * 1000
            out.row(qid=q["id"], ranked=r, answers=q["expected_docs"], ms=ms)
        out.close(index_s=round(idx, 2), n=len(qs)); print(A.name, "construction done", flush=True)
