"""SYNTHETIC scaling benchmark (same generator as the 2026-10-20 assessment harness).
Docs: 150-250 tokens drawn Zipf(1.1) from a 20k pseudo-word vocab + one unique 2-token needle.
Queries: needle tokens + 3 Zipf distractors. Known-item: relevant doc = the needle doc.
usage: scale.py N_DOCS N_QUERIES SEED [--rank-bm25] [--hybrid-queries K] [--tag TAG]"""
import os
BEIR = os.environ.get("LOCUS_BEIR_DIR", "data/beir")
OUT = os.environ.get("LOCUS_BENCH_OUT", "benchmarks/results/repro")
TMP = os.environ.get("LOCUS_BENCH_TMP") or None
os.makedirs(OUT, exist_ok=True)
import time, json, random, tempfile, resource, statistics, heapq, shutil, argparse, gc
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("n", type=int); ap.add_argument("nq", type=int); ap.add_argument("seed", type=int)
ap.add_argument("--rank-bm25", action="store_true"); ap.add_argument("--hybrid-queries", type=int, default=None)
ap.add_argument("--tag", default="new"); ap.add_argument("--out", default=OUT)
ap.add_argument("--keep", action="store_true")
a = ap.parse_args()
from locus.core import LocusEngine
from locus.memory.corpus import tokenize
N, NQ, SEED = a.n, a.nq, a.seed
rng = random.Random(SEED); nrng = np.random.default_rng(SEED)
syll = ["ka","lo","mi","ra","te","zu","po","ne","shi","va","do","gri","bel","tor","fen","qua"]
def word(i):
    s = []; x = i + 7
    while x: s.append(syll[x % 16]); x //= 16
    return "".join(s) + "x"
V = 20000; vocab = [word(i) for i in range(V)]
ranks = np.arange(1, V+1); p = 1/ranks**1.1; p /= p.sum()
tmp = Path(tempfile.mkdtemp(prefix=f"locus_scale_{N}_", dir=TMP)); dd = tmp/"docs"; dd.mkdir()
needles = []; texts = [] if a.rank_bm25 else None
for i in range(N):
    L = int(nrng.integers(150, 251)); toks = list(vocab[j] for j in nrng.choice(V, L, p=p))
    nd = [f"needle{i}a", f"needle{i}b"]; needles.append(nd)
    for t in nd: toks.insert(int(nrng.integers(0, len(toks))), t)
    txt = " ".join(toks)
    if texts is not None: texts.append(txt)
    (dd/f"d{i:06d}.md").write_text(f"# doc {i}\n\n{txt}\n")
qidx = rng.sample(range(N), min(NQ, N))
queries = [(i, " ".join(needles[i] + [vocab[j] for j in nrng.choice(V, 3, p=p)])) for i in qidx]
eng = LocusEngine(store_path=tmp/".locus")
t0 = time.perf_counter(); info = eng.index(dd); t_idx = time.perf_counter() - t0
store_bytes = sum(f.stat().st_size for f in (tmp/".locus").rglob("*") if f.is_file())
corpus_db = (tmp/".locus"/"corpus"/"corpus.sqlite3").stat().st_size
kg_db = (tmp/".locus"/"kg.sqlite3").stat().st_size
out = {"synthetic": True, "tag": a.tag, "n_docs": N, "n_queries": len(queries), "seed": SEED,
       "index_backend": eng.corpus.backend, "locus_index_s": t_idx, "locus_store_mb": store_bytes/2**20,
       "corpus_db_mb": corpus_db/2**20, "kg_db_mb": kg_db/2**20, "index_info": info}
print(json.dumps({k: v for k, v in out.items()}), flush=True)
def evalrun(fn, qs):
    lats = []; rr = []; hit10 = []; ranked = {}
    for i, q in qs:
        t = time.perf_counter(); docs = fn(q); lats.append((time.perf_counter()-t)*1000)
        ranked[str(i)] = docs[:10]
        tgt = f"d{i:06d}"; r = next((k+1 for k, d in enumerate(docs[:10]) if d == tgt), None)
        rr.append(1/r if r else 0); hit10.append(1 if r else 0)
    res = {"lat_ms": lats, "mrr@10": statistics.mean(rr), "recall@10": statistics.mean(hit10), "ranked": ranked}
    res.update({f"p{q}_ms": float(np.percentile(lats, q)) for q in (50, 95, 99)})
    res["mean_ms"] = float(np.mean(lats))
    return res
stem = lambda chunks: list(dict.fromkeys(Path(c.doc_path).stem for c in chunks))
# warm-up (page cache / sqlite) — excluded from timings
for _, q in queries[:3]: eng.retrieve(q, limit=10, use_cache=False)
out["locus_bm25"] = evalrun(lambda q: stem(eng.retrieve(q, limit=10, use_cache=False)), queries)
rep = evalrun(lambda q: stem(eng.retrieve(q, limit=10, use_cache=False)), queries)
out["determinism_bm25_identical_rankings"] = sum(out["locus_bm25"]["ranked"][k] == rep["ranked"][k] for k in rep["ranked"])
hq = queries[: (a.hybrid_queries or len(queries))]
out["locus_hybrid"] = evalrun(lambda q: stem(eng.retrieve(q, limit=10, use_cache=False, mode="hybrid")), hq)
print(json.dumps({k: {kk: vv for kk, vv in out[k].items() if kk not in ("lat_ms", "ranked")} for k in ("locus_bm25", "locus_hybrid")}), flush=True)
if a.rank_bm25:
    from rank_bm25 import BM25Okapi
    t0 = time.perf_counter(); bm = BM25Okapi([tokenize(t) for t in texts]); out["rank_bm25_index_s"] = time.perf_counter()-t0
    out["rank_bm25"] = evalrun(lambda q: [f"d{j:06d}" for j in heapq.nlargest(10, range(N), key=bm.get_scores(tokenize(q)).__getitem__)], queries)
    del bm; gc.collect()
out["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
json.dump(out, open(f"{a.out}/scale_{a.tag}_N{N}_s{SEED}.json", "w"))
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("lat_ms","ranked")}) for k, v in out.items()}), flush=True)
if not a.keep: shutil.rmtree(tmp, ignore_errors=True)
