"""BEIR evaluation of the new Locus pipeline (FTS5 BM25 + optional hybrid re-rank).
For each dataset: index the full corpus once, then run on the DEV split (SciFact: train qrels,
NFCorpus: dev qrels) and the TEST split:
  - locus_bm25            : engine.retrieve(mode="bm25")  (new default)
  - locus_hybrid_a{A}_d{D}: engine.retrieve(mode="hybrid") for a grid of alpha x depth
  - rank_bm25             : BM25Okapi baseline, same tokenizer, doc level
Config selection uses DEV only (see analyze.py). Writes per-query rankings + latencies.
usage: beir_eval.py DATASET"""
import os
BEIR = os.environ.get("LOCUS_BEIR_DIR", "data/beir")
OUT = os.environ.get("LOCUS_BENCH_OUT", "benchmarks/results/repro")
TMP = os.environ.get("LOCUS_BENCH_TMP") or None
os.makedirs(OUT, exist_ok=True)
import json, sys, time, tempfile, csv, heapq, resource, shutil, logging
from pathlib import Path
logging.disable(logging.WARNING)
from locus.core import LocusEngine
from locus.memory.corpus import tokenize
from rank_bm25 import BM25Okapi

ds = sys.argv[1]
d = Path(BEIR) / ds
out = Path(OUT)
ALPHAS = [0.1, 0.25, 0.5, 1.0]
DEPTHS = [40, 100, 200]
corpus = [json.loads(l) for l in open(d / "corpus.jsonl")]
queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(d / "queries.jsonl")}
def load_qrels(split):
    q = {}
    with open(d / "qrels" / f"{split}.tsv") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for qid, did, s in r:
            if int(s) > 0: q.setdefault(qid, {})[did] = int(s)
    return {k: v for k, v in q.items() if k in queries}
splits = {"dev": load_qrels("train" if ds == "scifact" else "dev"), "test": load_qrels("test")}
print(ds, len(corpus), "docs;", {k: len(v) for k, v in splits.items()}, "queries", flush=True)

tmp = Path(tempfile.mkdtemp(prefix=f"beir_{ds}_", dir=TMP))
docs = tmp / "docs"; docs.mkdir()
for doc in corpus:
    (docs / f"{doc['_id'].replace('/', '_')}.md").write_text(f"# {doc.get('title','')}\n\n{doc['text']}\n")
eng = LocusEngine(store_path=tmp / ".locus")
t0 = time.perf_counter(); info = eng.index(docs); t_index = time.perf_counter() - t0
store_mb = sum(p.stat().st_size for p in (tmp / ".locus").rglob("*") if p.is_file()) / 2**20
print(f"indexed {info} in {t_index:.1f}s, store {store_mb:.1f} MB", flush=True)

def to_docs(chunks, k=10):
    return list(dict.fromkeys(Path(c.doc_path).stem for c in chunks))[:k]

def run(fn, qids):
    res, lat = {}, {}
    for q in qids:
        t = time.perf_counter(); r = fn(queries[q]); lat[q] = (time.perf_counter() - t) * 1000
        res[q] = r
    return {"ranked": res, "latency_ms": lat}

ids = [x["_id"] for x in corpus]
bm = BM25Okapi([tokenize(f"{x.get('title','')} {x['text']}") for x in corpus])
systems = {
    "locus_bm25": lambda q: to_docs(eng.retrieve(q, limit=40, use_cache=False)),
    "rank_bm25": lambda q: [ids[i] for i in heapq.nlargest(10, range(len(ids)), key=bm.get_scores(tokenize(q)).__getitem__)],
}
def hybrid(alpha, depth):
    def f(q):
        eng.rerank_alpha = alpha; eng.rerank_depth = depth
        return to_docs(eng.retrieve(q, limit=40, use_cache=False, mode="hybrid"))
    return f
for a in ALPHAS:
    for dp in DEPTHS:
        systems[f"locus_hybrid_a{a}_d{dp}"] = hybrid(a, dp)
result = {"meta": {"dataset": ds, "n_docs": len(corpus), "index_s": t_index, "index_info": info,
                   "store_mb": store_mb, "backend": eng.corpus.backend, "alphas": ALPHAS, "depths": DEPTHS},
          "qrels": splits, "runs": {}}
for split, qr in splits.items():
    qids = sorted(qr)
    for name, fn in systems.items():
        result["runs"][f"{split}:{name}"] = run(fn, qids)
        print(split, name, "done", flush=True)
# determinism: repeat the default system on test
rep = run(systems["locus_bm25"], sorted(splits["test"]))
base = result["runs"]["test:locus_bm25"]["ranked"]
result["meta"]["determinism_bm25_test_identical"] = sum(rep["ranked"][q] == base[q] for q in base)
result["meta"]["determinism_n"] = len(base)
result["meta"]["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
json.dump(result, open(out / f"beir_{ds}_new.json", "w"))
print(json.dumps(result["meta"]), flush=True)
shutil.rmtree(tmp, ignore_errors=True)
