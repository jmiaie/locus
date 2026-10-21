"""Summarise 5x wiki / simple-100 runs (each a fresh process, different PYTHONHASHSEED)."""
import os
BEIR = os.environ.get("LOCUS_BEIR_DIR", "data/beir")
OUT = os.environ.get("LOCUS_BENCH_OUT", "benchmarks/results/repro")
TMP = os.environ.get("LOCUS_BENCH_TMP") or None
os.makedirs(OUT, exist_ok=True)
import json, glob
out = {}
for tag in ("before", "after"):
    for kind in ("wiki", "simple100"):
        runs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/repeat_{tag}/{kind}_run*.json"))]
        tab = {}
        for r in runs:
            for a in r.get("ablation", []):
                tab.setdefault("ablation:" + a["config"], []).append((a["recall@5"], a["mrr"]))
            for b in r.get("baseline", []):
                if b.get("num_queries", 0):
                    tab.setdefault("baseline:" + b["system"], []).append((b["recall@5"], b["mrr"]))
        out[f"{tag}:{kind}"] = {k: {"n_runs": len(v), "r@5": [x[0] for x in v], "mrr": [x[1] for x in v],
                                    "deterministic": len(set(v)) == 1} for k, v in tab.items()}
json.dump(out, open(f"{OUT}/repeat_summary.json", "w"), indent=1)
for key, tab in out.items():
    print("==", key)
    for k, v in tab.items():
        rs = sorted(set(v["r@5"])); ms = sorted(set(v["mrr"]))
        print(f"  {k:28s} runs={v['n_runs']} R@5={rs} MRR={ms} {'DETERMINISTIC' if v['deterministic'] else 'VARIES'}")
