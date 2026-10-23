"""Dev-split rerank-depth tuning: score the BM25 top-100 chunks once with the cross-encoder,
then simulate depth d (rerank the first d chunks, keep the rest in BM25 order) offline.
usage: rr_dev.py DATASET  -> results/<ds>_dev__rr_d{20,50,100}.jsonl"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "2")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys, json, csv, re, time, tempfile, logging
from pathlib import Path
logging.disable(logging.WARNING)
from locus.core import LocusEngine
from locus.rerank import CrossEncoderReranker
ds = sys.argv[1]; d = Path("/workspace/ompa-locus-20261020/data") / ds; RES = Path("/workspace/ragpack/results")
safe = lambda s: re.sub(r"[^\w-]", "_", s)
queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(d / "queries.jsonl")}
qf = "train" if ds == "scifact" else "dev"; qids = set()
with open(d / "qrels" / f"{qf}.tsv") as f:
    r = csv.reader(f, delimiter="\t"); next(r)
    for qid, did, s in r:
        if int(s) > 0 and qid in queries: qids.add(qid)
qids = sorted(qids)
ce = CrossEncoderReranker(depth=100, threads=2)
DEPTHS = [10, 20, 30, 50, 100]
outs = {k: open(RES / f"{ds}_dev__rr_d{k}.jsonl", "w") for k in DEPTHS}
with tempfile.TemporaryDirectory(dir="/workspace/ragpack/tmp") as wd:
    dd = Path(wd) / "docs"; dd.mkdir(); back = {}
    for l in open(d / "corpus.jsonl"):
        c = json.loads(l); n = safe(c["_id"]); back[n] = c["_id"]
        (dd / f"{n}.md").write_text(f"# {c.get('title', '') or ''}\n\n{c['text']}" if c.get("title") else c["text"])
    eng = LocusEngine(store_path=Path(wd) / ".locus"); eng.index(dd)
    for i, q in enumerate(qids):
        pool = eng.retrieve(queries[q], limit=100, use_cache=False, mode="bm25")
        t = time.perf_counter(); sc = ce.score(queries[q], [c.content for c in pool]); ms = (time.perf_counter() - t) * 1000
        for k in DEPTHS:
            head = sorted(range(min(k, len(pool))), key=lambda j: (-sc[j], j)); order = head + list(range(len(head), len(pool)))
            docs = []
            for j in order:
                x = back.get(Path(pool[j].doc_path).stem)
                if x not in docs: docs.append(x)
                if len(docs) == 10: break
            outs[k].write(json.dumps({"qid": q, "ranked": docs, "ms": ms * min(k, len(pool)) / max(1, len(pool))}) + "\n")
        if i % 200 == 0: print(ds, i, round(ms), flush=True)
for k, f in outs.items():
    f.close(); json.dump({"note": "simulated from one depth-100 scoring pass; ms = CE time scaled by depth"}, open(RES / f"{ds}_dev__rr_d{k}.meta.json", "w"))
print("done", ds)
