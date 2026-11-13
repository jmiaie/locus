"""Analyze full LongMemEval-S runs (Locus/OMPA + optional variants). Primary set = 470 non-abstention q (standard for
retrieval; abstention q's 'answer' sessions are the _abs evidence sessions); all-500 reported as supplementary.
Bootstrap: paired, B=10000, seed 20261020 (percentile CIs)."""
import json, sys, glob, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; B, SEED = 10000, 20261020
def load(name):
    rows = {}
    for f in sorted(glob.glob(str(H / f"lme500__{name}.shard*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if "error" not in r: rows[r["qid"]] = r
    return rows
def hit(r, k=5): return float(bool(set(r["ranked"][:k]) & set(r["answers"])))
def hit_all(r, k=5): return float(set(r["answers"]) <= set(r["ranked"][:k]))
def rr(r):
    for i, d in enumerate(r["ranked"][:10]):
        if d in set(r["answers"]): return 1 / (i + 1)
    return 0.0
def ndcg(r, k=10):
    a = set(r["answers"]); dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(r["ranked"][:k]) if d in a)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(a)))); return dcg / idcg
def boot(x):
    x = np.asarray(x, float); m = x[np.random.default_rng(SEED).integers(0, len(x), (B, len(x)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]
def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); m = d[np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)); p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return {"diff": round(float(d.mean()), 4), "ci": [round(lo, 4), round(hi, 4)], "sig": bool(lo > 0 or hi < 0), "p_boot_2s": round(float(min(1.0, p)), 4), "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
def summarize(rows, qs):
    o = {"n": len(qs)}
    for nm, f in [("R@5", hit), ("R@10", lambda r: hit(r, 10)), ("Rall@5", hit_all), ("MRR@10", rr), ("NDCG@10", ndcg)]:
        v = [f(rows[q]) for q in qs]; o[nm] = round(float(np.mean(v)), 4); o[nm + "_ci"] = boot(v)
    ms = [rows[q]["ms"] for q in qs]; o["query_ms_p50"] = round(float(np.median(ms)), 2); o["query_ms_p95"] = round(float(np.percentile(ms, 95)), 2)
    if "ms_warm" in rows[qs[0]]:
        w = [rows[q]["ms_warm"] for q in qs]; o["query_warm_ms_p50"] = round(float(np.median(w)), 2); o["query_warm_ms_p95"] = round(float(np.percentile(w, 95)), 2)
    ing = [rows[q]["ingest_s"] for q in qs]; o["ingest_s_p50"] = round(float(np.median(ing)), 3); o["ingest_s_p95"] = round(float(np.percentile(ing, 95)), 3); o["ingest_s_total"] = round(float(np.sum(ing)), 1)
    return o
if __name__ == "__main__":
    names = sys.argv[1:] or ["locus", "ompa"]
    D = {n: load(n) for n in names}
    allq = json.load(open(H / "lme500_qids.json"))
    out = {}
    for setname, filt in [("non_abstention_470", lambda q: not q.endswith("_abs")), ("all_500", lambda q: True)]:
        qs_all = [q for q in allq if filt(q)]
        S = out[setname] = {}
        for n, rows in D.items():
            qs = [q for q in qs_all if q in rows]
            if not qs: continue
            s = summarize(rows, qs); s["complete"] = len(qs) == len(qs_all)
            s["by_type"] = {}
            for t in sorted({rows[q]["qtype"] for q in qs}):
                tq = [q for q in qs if rows[q]["qtype"] == t]
                s["by_type"][t] = {"n": len(tq), "R@5": round(float(np.mean([hit(rows[q]) for q in tq])), 4), "R@5_ci": boot([hit(rows[q]) for q in tq]),
                                   "Rall@5": round(float(np.mean([hit_all(rows[q]) for q in tq])), 4), "MRR@10": round(float(np.mean([rr(rows[q]) for q in tq])), 4), "MRR_ci": boot([rr(rows[q]) for q in tq]),
                                   "misses@5": [q for q in tq if not hit(rows[q])]}
            S[n] = s
        # paired tests vs first system (and locus vs ompa)
        S["paired"] = {}
        base = names[0]
        for n in names[1:]:
            common = [q for q in qs_all if q in D[n] and q in D[base]]
            if not common: continue
            pr = {"n": len(common)}
            for nm, f in [("MRR@10", rr), ("R@5", hit), ("Rall@5", hit_all), ("NDCG@10", ndcg)]:
                pr[nm] = paired([f(D[n][q]) for q in common], [f(D[base][q]) for q in common])
            pr["by_type_MRR"] = {}
            for t in sorted({D[base][q]["qtype"] for q in common}):
                tq = [q for q in common if D[base][q]["qtype"] == t]
                pr["by_type_MRR"][t] = paired([rr(D[n][q]) for q in tq], [rr(D[base][q]) for q in tq])
            S["paired"][f"{n}_minus_{base}"] = pr
    tag = "_".join(names)
    json.dump(out, open(H / f"summary500__{tag}.json", "w"), indent=1)
    for setname, S in out.items():
        print("==", setname)
        for n, s in S.items():
            if n == "paired": continue
            print(f"{n:28s} n={s['n']} R@5 {s['R@5']:.4f} {s['R@5_ci']} Rall@5 {s['Rall@5']:.4f} MRR {s['MRR@10']:.4f} {s['MRR@10_ci']} NDCG {s['NDCG@10']:.4f} q50/95 {s['query_ms_p50']}/{s['query_ms_p95']}ms ingest50 {s['ingest_s_p50']}s")
            print("   " + "; ".join(f"{t} n={v['n']} R@5 {v['R@5']:.3f} Rall {v['Rall@5']:.3f} MRR {v['MRR@10']:.3f}" for t, v in s["by_type"].items()))
        for k, v in S["paired"].items():
            print(f"  {k}: dMRR {v['MRR@10']['diff']:+.4f} {v['MRR@10']['ci']} {'SIG' if v['MRR@10']['sig'] else 'ns'} p={v['MRR@10']['p_boot_2s']} b/w {v['MRR@10']['better']}/{v['MRR@10']['worse']} | dR@5 {v['R@5']['diff']:+.4f} {v['R@5']['ci']} {'SIG' if v['R@5']['sig'] else 'ns'} | dRall@5 {v['Rall@5']['diff']:+.4f} {v['Rall@5']['ci']} {'SIG' if v['Rall@5']['sig'] else 'ns'}")
            print("     by type dMRR: " + "; ".join(f"{t} {x['diff']:+.3f}{'*' if x['sig'] else ''}" for t, x in v["by_type_MRR"].items()))
