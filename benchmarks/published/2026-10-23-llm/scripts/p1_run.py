"""Phase 1: Locus LLM rewrite / multi-query / HyDE vs default BM25.
usage: p1_run.py BENCH [--gen-only]   BENCH: scifact | nfcorpus | lme
Writes results/<key>__<variant>.jsonl rows {qid, ranked, answers?, ms (retrieval), llm_ms, cost}."""
import os, sys, re, json, time, csv, tempfile, logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, "/workspace/scorecard-20261020/repos/locus"); sys.path.insert(0, "/workspace/llmbench")
logging.disable(logging.WARNING)
import cached_llm
from locus.core import LocusEngine
from locus.llm import OpenAICompatibleClient, LLMQueryTransformer
from locus.retrieval.fusion import rrf_fuse
MODEL = "openai/gpt-4o-mini"; STRATS = ("rewrite", "multi", "hyde")
RES = Path("/workspace/llmbench/results"); TMP = Path("/workspace/llmbench/tmp")
DATA = Path("/workspace/ompa-locus-20261020/data"); LME = Path("/workspace/stem/data/lme_s_470.jsonl")
bench = sys.argv[1]; gen_only = "--gen-only" in sys.argv
safe = lambda s: re.sub(r"[^\w-]", "_", s)
if bench in ("scifact", "nfcorpus"):
    d = DATA / bench
    queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(d / "queries.jsonl")}
    qids = set()
    with open(d / "qrels" / "test.tsv") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for qid, did, s in r:
            if int(s) > 0 and qid in queries: qids.add(qid)
    QS = [(q, queries[q]) for q in sorted(qids)]; key = f"{bench}_test"
else:
    LMEROWS = [json.loads(l) for l in open(LME)]
    QS = [(e["question_id"], e["question"]) for e in LMEROWS]; key = "lme"
def transform(args):
    strat, (qid, q) = args
    c = OpenAICompatibleClient.from_env(model=MODEL, timeout=45)
    captured = {}
    def tp(url, h, b, t):
        n0 = len(cached_llm.STATS["ms"]); d = cached_llm.transport(url, h, b, t)
        captured["d"] = d; captured["ms"] = cached_llm.STATS["ms"][n0] if len(cached_llm.STATS["ms"]) > n0 else None; return d
    c.transport = tp
    qs = LLMQueryTransformer(c).queries(q, strat, n=3)
    d = captured.get("d", {}); u = d.get("usage", {})
    return strat, qid, {"queries": qs, "cost": u.get("cost", 0.0), "pt": u.get("prompt_tokens", 0), "ct": u.get("completion_tokens", 0),
                        "llm_ms": captured.get("ms"), "model": d.get("model"), "provider": d.get("provider")}
T = {s: {} for s in STRATS}
with ThreadPoolExecutor(8) as ex:
    for s, qid, v in ex.map(transform, [(s, x) for s in STRATS for x in QS]): T[s][qid] = v
json.dump(T, open(RES / f"{key}__transforms.json", "w"))
print(bench, "transforms done", {s: round(sum(v["cost"] for v in T[s].values()), 4) for s in STRATS}, cached_llm.STATS["hits"], cached_llm.STATS["misses"], flush=True)
if gen_only: sys.exit()
DEPTH = 100
def docs_of(chunks, back, k):
    out = []
    for c in chunks:
        d = back(c.doc_path)
        if d not in out: out.append(d)
        if len(out) == k: break
    return out
def run_all(eng, back, qid, q, k, outs, extra):
    t = time.perf_counter(); base = eng.retrieve(q, limit=DEPTH, use_cache=False, mode="bm25")
    ms0 = (time.perf_counter() - t) * 1000
    outs["base"].write(json.dumps({"qid": qid, "ranked": docs_of(base, back, k), "ms": ms0, **extra}) + "\n")
    for s in STRATS:
        tv = T[s][qid]; t = time.perf_counter()
        lists = [base] + [eng.retrieve(x, limit=DEPTH, use_cache=False, mode="bm25") for x in tv["queries"][1:]]
        fused = rrf_fuse(lists, limit=DEPTH) if len(lists) > 1 else lists[0]
        ms = (time.perf_counter() - t) * 1000 + ms0
        outs[s].write(json.dumps({"qid": qid, "ranked": docs_of(fused, back, k), "ms": ms, "llm_ms": tv["llm_ms"], "cost": tv["cost"], **extra}) + "\n")
outs = {v: open(RES / f"{key}__{v}.jsonl", "w") for v in ("base",) + STRATS}
if bench in ("scifact", "nfcorpus"):
    with tempfile.TemporaryDirectory(dir=TMP) as wd:
        dd = Path(wd) / "docs"; dd.mkdir(); back = {}
        for l in open(d / "corpus.jsonl"):
            c = json.loads(l); n = safe(c["_id"]); back[n] = c["_id"]
            (dd / f"{n}.md").write_text(f"# {c.get('title', '') or ''}\n\n{c['text']}" if c.get("title") else c["text"])
        eng = LocusEngine(store_path=Path(wd) / ".locus"); eng.index(dd)
        bk = lambda p: back.get(Path(p).stem)
        for qid, q in QS[:3]: eng.retrieve(q, limit=DEPTH, use_cache=False, mode="bm25")
        for qid, q in QS: run_all(eng, bk, qid, q, 10, outs, {})
else:
    for i, e in enumerate(LMEROWS):
        with tempfile.TemporaryDirectory(dir=TMP) as wd:
            dd = Path(wd) / "docs"; dd.mkdir(); back = {}
            for sid, sess in zip(e["haystack_session_ids"], e["haystack_sessions"]):
                k = safe(sid); back[k] = sid
                (dd / f"{k}.md").write_text("\n".join(f"{t['role']}: {t['content']}" for t in sess))
            eng = LocusEngine(store_path=Path(wd) / ".locus"); eng.index(dd)
            run_all(eng, lambda p: back.get(Path(p).stem), e["question_id"], e["question"], 10, outs,
                    {"qtype": e["question_type"], "answers": e["answer_session_ids"]})
        if i % 100 == 0: print("lme", i, flush=True)
for f in outs.values(): f.close()
print(bench, "done", flush=True)
