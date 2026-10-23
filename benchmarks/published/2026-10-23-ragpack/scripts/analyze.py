"""Metrics + seeded (paired) bootstrap.  usage: analyze.py KEY BASE NAME [NAME...]
KEY: scifact_dev scifact_test nfcorpus_dev nfcorpus_test lme construction"""
import json, sys, csv, math
import numpy as np
from pathlib import Path
RES = Path("/workspace/ragpack/results"); DATA = Path("/workspace/ompa-locus-20261020/data")
B, SEED = 10000, 20261020
def ndcg10(ranked, rel):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
def qrels(key):
    ds, split = key.split("_"); f = ("train" if ds == "scifact" else "dev") if split == "dev" else "test"
    q = {}
    with open(DATA / ds / "qrels" / f"{f}.tsv") as fh:
        r = csv.reader(fh, delimiter="\t"); next(r)
        for qid, did, s in r:
            if int(s) > 0: q.setdefault(qid, {})[did] = int(s)
    return q
def load(key, name):
    p = RES / f"{key}__{name}.jsonl"
    if not p.exists(): return None
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p)}
def per_query(key, rows):
    if key.startswith(("scifact", "nfcorpus")):
        qr = qrels(key); return {q: ndcg10(r["ranked"], qr[q]) for q, r in rows.items()}
    k = 5
    return {q: float(bool(set(r["ranked"][:k]) & set(r["answers"]))) for q, r in rows.items()}
def mrr(rows):
    out = {}
    for q, r in rows.items():
        rk = [i + 1 for i, d in enumerate(r["ranked"]) if d in set(r["answers"])]
        out[q] = 1 / rk[0] if rk else 0.0
    return out
def boot(x):
    x = np.asarray(x, float); m = x[np.random.default_rng(SEED).integers(0, len(x), (B, len(x)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]
def summarize(key, base, name):
    rb, rn = load(key, base), load(key, name)
    if rn is None: return {"status": "missing"}
    qs = sorted(set(rb) & set(rn)) if rb else sorted(rn)
    pn = per_query(key, rn); v = [pn[q] for q in qs]
    ms = [rn[q]["ms"] for q in qs]
    meta = json.load(open(RES / f"{key}__{name}.meta.json"))
    res = {"metric": "nDCG@10" if key.startswith(("scifact", "nfcorpus")) else "R@5", "n": len(qs),
           "value": round(float(np.mean(v)), 4), "ci": boot(v),
           "p50_ms": round(float(np.percentile(ms, 50)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2),
           "index_s": meta.get("index_s", meta.get("total_index_s")), "store_mb": meta.get("store_mb"), "chunks": meta.get("chunks")}
    if key in ("construction", "lme"): res["MRR"] = round(float(np.mean([mrr(rn)[q] for q in qs])), 4)
    if rb and name != base:
        pb = per_query(key, rb); d = np.array([pn[q] - pb[q] for q in qs])
        m = d[np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))].mean(1)
        lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
        bms = [rb[q]["ms"] for q in qs]
        res["vs_base"] = {"diff": round(float(d.mean()), 4), "ci": [round(lo, 4), round(hi, 4)], "sig": bool(lo > 0 or hi < 0),
                          "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
                          "latency_ratio_p50": round(float(np.percentile(ms, 50) / np.percentile(bms, 50)), 2)}
    return res
if __name__ == "__main__":
    key, base, *names = sys.argv[1:]
    for n in [base] + names:
        r = summarize(key, base, n)
        if "value" not in r: print(f"{n:28s} missing"); continue
        s = f"{n:28s} {r['value']:.4f} [{r['ci'][0]:.4f},{r['ci'][1]:.4f}] p50 {r['p50_ms']}ms"
        if "vs_base" in r:
            v = r["vs_base"]; s += f"  d={v['diff']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] {'SIG' if v['sig'] else 'ns'} {v['better']}/{v['worse']} lat x{v['latency_ratio_p50']}"
        print(s)
