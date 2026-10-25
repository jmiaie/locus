"""Phase 2 metrics on the fixed 30-q LongMemEval-S subset. R@5 (any answer session in top-5 distinct sessions), R@10,
paired bootstrap on MRR@10 (B=10000, seed 20261020) vs Locus and vs OMPA (R@5 saturates). Cost from the local gateway log (per-system tag)."""
import json, sqlite3, numpy as np
from pathlib import Path
R = Path("/workspace/llmbench/results"); B, SEED = 10000, 20261020
SUB = json.load(open("/workspace/llmbench/lme_subset30.json"))
SYSTEMS = ["locus", "ompa", "mem0", "langmem", "graphiti", "graphiti_relaxed", "letta", "letta_direct"]
db = sqlite3.connect("/workspace/llmbench/gateway.sqlite")
def load(s):
    f = R / f"lme30__{ {'letta_direct': 'letta', 'graphiti_relaxed': 'graphiti'}.get(s, s) }.jsonl"
    if not f.exists(): return {}
    rows = {}
    for l in open(f):
        r = json.loads(l)
        if "error" in r: continue
        if s == "letta_direct": r = {**r, "ranked": r["direct_ranked"], "ms": r["direct_ms"]}
        if s == "graphiti_relaxed":
            if "relaxed_ranked" not in r: continue
            r = {**r, "ranked": r["relaxed_ranked"], "ms": r["relaxed_ms"]}
        rows[r["qid"]] = r
    return rows
def hit(r, k): return float(bool(set(r["ranked"][:k]) & set(r["answers"])))
def hit_all(r, k): return float(set(r["answers"]) <= set(r["ranked"][:k]))
def rr(r):
    for i, d in enumerate(r["ranked"][:10]):
        if d in set(r["answers"]): return 1 / (i + 1)
    return 0.0
def ci(x):
    x = np.asarray(x, float); m = x[np.random.default_rng(SEED).integers(0, len(x), (B, len(x)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]
D = {s: load(s) for s in SYSTEMS}; out = {}
for s, rows in D.items():
    if not rows: out[s] = {"status": "not run"}; continue
    qs = [q for q in SUB if q in rows]; r5 = [hit(rows[q], 5) for q in qs]
    tag = s.split("_")[0]
    cost = db.execute("select coalesce(sum(cost),0), coalesce(sum(pt),0), coalesce(sum(ct),0), count(*) from log where tag=? and status=200", (tag,)).fetchone()
    o = {"n": len(qs), "R@5": round(float(np.mean(r5)), 3), "R@5_ci": ci(r5), "R@10": round(float(np.mean([hit(rows[q], 10) for q in qs])), 3), "Rall@5": round(float(np.mean([hit_all(rows[q], 5) for q in qs])), 3), "MRR@10": round(float(np.mean([rr(rows[q]) for q in qs])), 3),
         "query_p50_ms": round(float(np.median([rows[q]["ms"] for q in qs])), 1),
         "ingest_p50_s_per_haystack": round(float(np.median([rows[q]["ingest_s"] for q in qs])), 1) if "ingest_s" in rows[qs[0]] else None,
         "llm_usd_total_incl_pilots": round(cost[0], 4), "llm_calls": cost[3], "prompt_tokens": cost[1], "completion_tokens": cost[2]}
    if s not in ("locus", "ompa", "letta_direct", "graphiti_relaxed") and len(qs): o["usd_per_question"] = round(cost[0] / len(qs), 4)
    for ref in ("locus", "ompa"):
        if s == ref or not D[ref]: continue
        common = [q for q in qs if q in D[ref]]
        d = np.array([rr(rows[q]) - rr(D[ref][q]) for q in common])
        m = d[np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))].mean(1)
        lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
        o[f"vs_{ref}"] = {"ref_R@5_same_q": round(float(np.mean([hit(D[ref][q], 5) for q in common])), 3), "ref_MRR_same_q": round(float(np.mean([rr(D[ref][q]) for q in common])), 3), "n": len(common), "diff": round(float(d.mean()), 3), "ci": [round(lo, 3), round(hi, 3)], "sig": bool(lo > 0 or hi < 0), "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
    bt = {}
    for q in qs: bt.setdefault(rows[q]["qtype"], []).append(hit(rows[q], 5))
    o["by_type_R@5"] = {t: round(float(np.mean(v)), 2) for t, v in sorted(bt.items())}
    extra = [k for k in ("add_errors", "n_memories", "episodes", "archival_search_calls", "passages") if k in rows[qs[0]]]
    for k in extra:
        vals = [rows[q].get(k) for q in qs if rows[q].get(k) is not None]
        if vals: o[f"mean_{k}"] = round(float(np.mean(vals)), 2)
    out[s] = o
json.dump(out, open(R / "p2_summary.json", "w"), indent=1)
for s, o in out.items():
    if "n" not in o: print(f"{s:13s} not run"); continue
    line = f"{s:13s} n={o['n']:2d} R@5 {o['R@5']:.3f} {o['R@5_ci']} R@10 {o['R@10']:.3f} Rall@5 {o['Rall@5']:.3f} MRR {o['MRR@10']:.3f} q50 {o['query_p50_ms']}ms ingest50 {o['ingest_p50_s_per_haystack']}s ${o['llm_usd_total_incl_pilots']}"
    for ref in ("locus", "ompa"):
        if f"vs_{ref}" in o: v = o[f"vs_{ref}"]; line += f" | vs {ref} {v['diff']:+.3f} {v['ci']} {'SIG' if v['sig'] else 'ns'} {v['better']}/{v['worse']}"
    print(line)
