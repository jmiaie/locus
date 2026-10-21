"""Test-split evaluation of Locus analyzers vs Locus main, rank_bm25, LlamaIndex BM25.
Seeded bootstrap (B=10000, seed 20261020), paired bootstrap on per-query scores."""
import json, sys, os, math
import numpy as np
sys.path.insert(0, "/workspace/stem/harness")
from common import load_beir, RES
B, SEED = 10000, 20261020
def ndcg10(ranked, rel):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
def ci(x):
    x = np.asarray(x, float); m = x[np.random.default_rng(SEED).integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    m = d[np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    return {"diff": round(float(d.mean()), 4), "ci": [round(lo, 4), round(hi, 4)], "sig": bool(lo > 0 or hi < 0),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "tie": int((d == 0).sum())}
SYS = {"locus_main": "main_bm25", "branch_plain": "locus_bm25_plain", "branch_porter": "locus_bm25_porter",
       "branch_porter_english": "locus_bm25_porter_english", "rank_bm25": "main_rank_bm25", "llamaindex_bm25": "main_llamaindex_bm25"}
REFS = ["locus_main", "llamaindex_bm25", "rank_bm25"]
def rows(bench, f):
    p = RES / f"{bench}__{f}.jsonl"
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p)} if p.exists() else None
out = {"B": B, "seed": SEED}
for bench in ("scifact", "nfcorpus", "lme"):
    if bench == "lme":
        R = {k: rows(bench, f) for k, f in SYS.items()}
        qids = sorted(R["locus_main"]); score = lambda r, q: float(bool(set(r[q]["ranked"][:5]) & set(r[q]["answers"])))
    else:
        _, _, qrels, qids = load_beir(bench); R = {k: rows(bench, f) for k, f in SYS.items()}
        score = lambda r, q: ndcg10(r[q]["ranked"], qrels[q])
    res = {}; per = {}
    for k, r in R.items():
        if r is None or len(r) != len(qids): res[k] = {"status": "missing/incomplete"}; continue
        per[k] = [score(r, q) for q in qids]; m, lo, hi = ci(per[k]); ms = [r[q]["ms"] for q in qids]
        meta = json.load(open(RES / f"{bench}__{SYS[k]}.meta.json")) if (RES / f"{bench}__{SYS[k]}.meta.json").exists() else {}
        res[k] = {"value": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "p50_ms": round(float(np.percentile(ms, 50)), 2),
                  "p95_ms": round(float(np.percentile(ms, 95)), 2), "index_s": meta.get("index_s", meta.get("total_index_s")), "peak_rss_mb": meta.get("peak_rss_mb")}
    for k in ("branch_plain", "branch_porter", "branch_porter_english"):
        if k in per:
            res[k]["vs"] = {ref: paired(per[k], per[ref]) for ref in REFS if ref in per}
            if "branch_plain" in per and k != "branch_plain": res[k]["vs"]["branch_plain"] = paired(per[k], per["branch_plain"])
    if "branch_plain" in per and "locus_main" in per:
        res["branch_plain"]["identical_rankings_to_main"] = all(R["branch_plain"][q]["ranked"] == R["locus_main"][q]["ranked"] for q in qids)
    out[bench] = {"metric": "R_any@5" if bench == "lme" else "nDCG@10", "n": len(qids), "systems": res}
json.dump(out, open("/workspace/stem/compare.json", "w"), indent=1)
for bench in ("scifact", "nfcorpus", "lme"):
    print(f"== {bench} {out[bench]['metric']} n={out[bench]['n']}")
    for k, v in out[bench]["systems"].items():
        if "value" not in v: print(f"  {k:22s} {v['status']}"); continue
        line = f"  {k:22s} {v['value']:.4f} [{v['ci'][0]:.4f},{v['ci'][1]:.4f}] p50 {v['p50_ms']} p95 {v['p95_ms']} idx {v['index_s']}"
        if "identical_rankings_to_main" in v: line += f" identical_to_main={v['identical_rankings_to_main']}"
        print(line)
        for ref, p in v.get("vs", {}).items():
            print(f"      vs {ref:16s} {p['diff']:+.4f} [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] {'SIG' if p['sig'] else 'ns'} {p['better']}/{p['worse']}/{p['tie']}")
