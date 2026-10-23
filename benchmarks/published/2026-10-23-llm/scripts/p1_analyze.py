"""Phase 1 metrics + paired bootstrap (B=10000, seed 20261020). Writes results/p1_summary.json"""
import json, sys, csv, math
import numpy as np
from pathlib import Path
RES = Path("/workspace/llmbench/results"); DATA = Path("/workspace/ompa-locus-20261020/data")
B, SEED = 10000, 20261020
def ndcg10(ranked, rel):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
def qrels(ds):
    q = {}
    with open(DATA / ds / "qrels" / "test.tsv") as fh:
        r = csv.reader(fh, delimiter="\t"); next(r)
        for qid, did, s in r:
            if int(s) > 0: q.setdefault(qid, {})[did] = int(s)
    return q
def load(key, v):
    p = RES / f"{key}__{v}.jsonl"
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p)} if p.exists() else None
def per_q(key, rows):
    if key != "lme":
        qr = qrels(key.split("_")[0]); return {q: ndcg10(r["ranked"], qr[q]) for q, r in rows.items()}
    return {q: float(bool(set(r["ranked"][:5]) & set(r["answers"]))) for q, r in rows.items()}
def boot_mean(x, rng_seed=SEED):
    x = np.asarray(x, float); m = x[np.random.default_rng(rng_seed).integers(0, len(x), (B, len(x)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]
out = {}
for key in ("scifact_test", "nfcorpus_test", "lme"):
    base = load(key, "base")
    if not base: continue
    pb = per_q(key, base); qs = sorted(base); out[key] = {}
    for v in ("base", "rewrite", "multi", "hyde"):
        rows = load(key, v)
        if not rows or len(rows) != len(base): out[key][v] = {"status": "missing/incomplete"}; continue
        pv = per_q(key, rows); x = [pv[q] for q in qs]
        ret_ms = [rows[q]["ms"] for q in qs]
        r = {"metric": "nDCG@10" if key != "lme" else "R@5", "n": len(qs), "value": round(float(np.mean(x)), 4), "ci": boot_mean(x),
             "retrieval_p50_ms": round(float(np.percentile(ret_ms, 50)), 2), "retrieval_p95_ms": round(float(np.percentile(ret_ms, 95)), 2)}
        if v != "base":
            llm = [rows[q]["llm_ms"] for q in qs if rows[q].get("llm_ms") is not None]
            e2e = [rows[q]["ms"] + (rows[q].get("llm_ms") or 0) for q in qs]
            cost = [rows[q].get("cost") or 0 for q in qs]
            d = np.array([pv[q] - pb[q] for q in qs])
            m = d[np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))].mean(1)
            lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
            r.update({"llm_p50_ms": round(float(np.percentile(llm, 50)), 0), "llm_p95_ms": round(float(np.percentile(llm, 95)), 0),
                      "e2e_p50_ms": round(float(np.percentile(e2e, 50)), 0), "cost_total_usd": round(sum(cost), 4),
                      "usd_per_1k_queries": round(1000 * sum(cost) / len(qs), 4),
                      "vs_base": {"diff": round(float(d.mean()), 4), "ci": [round(lo, 4), round(hi, 4)], "sig": bool(lo > 0 or hi < 0),
                                  "better": int((d > 0).sum()), "worse": int((d < 0).sum())}})
        if key == "lme":
            bt = {}
            for q in qs: bt.setdefault(rows[q]["qtype"], []).append(pv[q])
            r["by_type"] = {t: [round(float(np.mean(a)), 3), len(a)] for t, a in sorted(bt.items())}
        out[key][v] = r
json.dump(out, open(RES / "p1_summary.json", "w"), indent=1)
for key, vs in out.items():
    print("##", key)
    for v, r in vs.items():
        if "value" not in r: print(v, r); continue
        s = f"{v:8s} {r['metric']} {r['value']:.4f} [{r['ci'][0]:.4f},{r['ci'][1]:.4f}] ret p50 {r['retrieval_p50_ms']}ms"
        if "vs_base" in r:
            b = r["vs_base"]; s += f" | d={b['diff']:+.4f} [{b['ci'][0]:+.4f},{b['ci'][1]:+.4f}] {'SIG' if b['sig'] else 'ns'} {b['better']}/{b['worse']} | llm p50 {r['llm_p50_ms']}ms e2e p50 {r['e2e_p50_ms']}ms | ${r['usd_per_1k_queries']}/1k q"
        print(s)
