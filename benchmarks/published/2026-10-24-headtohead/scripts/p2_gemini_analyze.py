"""Graphiti-on-Gemini side-by-side (free tier, gemini-3.5-flash-lite). SEPARATE output: results/p2_gemini_summary.json.
Does not touch p2_summary.json. Same metric definitions and paired bootstrap as p2_analyze.py (MRR@10 diff, B=10000, seed 20261020).
Extra (not in p2_analyze.py): paired-bootstrap R@5 diff and a two-sided bootstrap p (2*min(P(m<=0), P(m>=0)))."""
import json, sqlite3, numpy as np
from pathlib import Path
import importlib.util
R = Path("/workspace/llmbench/results"); B, SEED = 10000, 20261020
SUB = json.load(open("/workspace/llmbench/lme_subset30.json"))
def load(fname, variant=None):
    rows = {}
    for l in open(R / fname):
        r = json.loads(l)
        if "error" in r: continue
        if variant == "relaxed":
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
def paired(d):
    d = np.asarray(d, float); m = d[np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    p = min(1.0, 2 * min(float((m <= 0).mean()), float((m >= 0).mean())))
    return {"diff": round(float(d.mean()), 3), "ci": [round(lo, 3), round(hi, 3)], "sig": bool(lo > 0 or hi < 0), "p_boot_2s": round(p, 4),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
D = {"graphiti_gemini": load("lme30__graphiti_gemini.jsonl"), "graphiti_gemini_relaxed": load("lme30__graphiti_gemini.jsonl", "relaxed"),
     "graphiti": load("lme30__graphiti.jsonl"), "graphiti_relaxed": load("lme30__graphiti.jsonl", "relaxed"),
     "locus": load("lme30__locus.jsonl"), "ompa": load("lme30__ompa.jsonl")}
gdb = sqlite3.connect("file:/workspace/llmbench/gateway_gemini.sqlite?mode=ro", uri=True)
cost = gdb.execute("select coalesce(sum(cost),0), coalesce(sum(pt),0), coalesce(sum(ct),0), count(*), sum(cached) from log where tag='graphiti' and status=200").fetchone()
models = [m for (m,) in gdb.execute("select distinct model from log")]
REFS = {"graphiti_gemini": ["graphiti", "locus", "ompa"], "graphiti_gemini_relaxed": ["graphiti_relaxed", "graphiti", "locus", "ompa"]}
out = {"_meta": {"llm": "gemini-3.5-flash-lite (free tier, reasoning_effort=minimal)", "embedder": "all-MiniLM-L6-v2 via local gateway (same as gpt-4o-mini run)",
                 "note": "different LLM, not directly comparable to the gpt-4o-mini rows", "gateway_models_seen": models,
                 "gateway_llm_calls_200": cost[3], "gateway_cached_hits": cost[4], "prompt_tokens": cost[1], "completion_tokens": cost[2], "usd_total": round(cost[0], 4),
                 "ingest_note": "ingest_s includes free-tier rate-limit waits / daily-quota holds / 503 retries -> not a throughput measure"}}
raw = [json.loads(l) for l in open(R / "lme30__graphiti_gemini.jsonl")]
out["_meta"].update({"rows": len(raw), "error_rows": sum("error" in r for r in raw), "sum_add_errors": int(sum(r.get("add_errors") or 0 for r in raw)),
                     "row_llm_models": sorted({r.get("llm_model") for r in raw}), "max_cum_usd_system": max(r.get("cum_usd_system", 0) for r in raw),
                     "qids_match_subset": sorted(r["qid"] for r in raw) == sorted(SUB)})
for s in ("graphiti_gemini", "graphiti_gemini_relaxed"):
    rows = D[s]; qs = [q for q in SUB if q in rows]; r5 = [hit(rows[q], 5) for q in qs]
    o = {"n": len(qs), "R@5": round(float(np.mean(r5)), 3), "R@5_ci": ci(r5), "R@10": round(float(np.mean([hit(rows[q], 10) for q in qs])), 3),
         "Rall@5": round(float(np.mean([hit_all(rows[q], 5) for q in qs])), 3), "MRR@10": round(float(np.mean([rr(rows[q]) for q in qs])), 3),
         "query_p50_ms": round(float(np.median([rows[q]["ms"] for q in qs])), 1), "ingest_p50_s_per_haystack": round(float(np.median([rows[q]["ingest_s"] for q in qs])), 1),
         "mean_add_errors": round(float(np.mean([rows[q]["add_errors"] for q in qs])), 2), "mean_episodes": round(float(np.mean([rows[q]["episodes"] for q in qs])), 2)}
    for ref in REFS[s]:
        common = [q for q in qs if q in D[ref]]
        v = paired([rr(rows[q]) - rr(D[ref][q]) for q in common])
        v.update({"n": len(common), "ref_R@5_same_q": round(float(np.mean([hit(D[ref][q], 5) for q in common])), 3), "ref_MRR_same_q": round(float(np.mean([rr(D[ref][q]) for q in common])), 3),
                  "ref_query_p50_ms": round(float(np.median([D[ref][q]["ms"] for q in common])), 1),
                  "R@5_diff": paired([hit(rows[q], 5) - hit(D[ref][q], 5) for q in common])})
        o[f"vs_{ref}"] = v
    o["median_n_edges_hit"] = float(np.median([rows[q]["n_edges_hit"] for q in qs])); o["empty_default_search"] = int(sum(len(D["graphiti_gemini"][q]["ranked"]) == 0 for q in qs))
    o["median_relaxed_edges"] = float(np.median([rows[q]["relaxed_edges"] for q in qs]))
    bt = {}
    for q in qs: bt.setdefault(rows[q]["qtype"], []).append(hit(rows[q], 5))
    o["by_type_R@5"] = {t: round(float(np.mean(v)), 2) for t, v in sorted(bt.items())}
    out[s] = o
g4 = [D["graphiti"][q] for q in SUB if q in D["graphiti"]]
out["_meta"]["gpt4omini_ref"] = {"median_n_edges_hit": float(np.median([r["n_edges_hit"] for r in g4])), "empty_default_search": int(sum(len(r["ranked"]) == 0 for r in g4)),
    "median_relaxed_edges": float(np.median([r["relaxed_edges"] for r in g4])), "mean_episodes": round(float(np.mean([r["episodes"] for r in g4])), 2)}
json.dump(out, open(R / "p2_gemini_summary.json", "w"), indent=1)
print(json.dumps(out["_meta"], indent=1))
for s in ("graphiti_gemini", "graphiti_gemini_relaxed"):
    o = out[s]
    print(f"{s}: n={o['n']} R@5 {o['R@5']:.3f} {o['R@5_ci']} R@10 {o['R@10']:.3f} Rall@5 {o['Rall@5']:.3f} MRR {o['MRR@10']:.3f} q50 {o['query_p50_ms']}ms ingest50 {o['ingest_p50_s_per_haystack']}s")
    for ref in REFS[s]:
        v = o[f"vs_{ref}"]; w = v["R@5_diff"]
        print(f"   vs {ref:16s} dMRR {v['diff']:+.3f} {v['ci']} {'SIG' if v['sig'] else 'ns'} p={v['p_boot_2s']} {v['better']}/{v['worse']} | dR@5 {w['diff']:+.3f} {w['ci']} {'SIG' if w['sig'] else 'ns'} p={w['p_boot_2s']} | ref MRR {v['ref_MRR_same_q']} R@5 {v['ref_R@5_same_q']} q50 {v['ref_query_p50_ms']}ms")
