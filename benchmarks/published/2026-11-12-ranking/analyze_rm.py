"""Paired-bootstrap comparison of implemented variants vs in-harness baseline (470 non-abstention q), plus identity check
of the in-harness baseline vs the original LME-500 run (expansion-20261112/lme500__SYSTEM.shard*.jsonl)."""
import sys, json, glob, numpy as np
from pathlib import Path
sys.path.insert(0, "/workspace/llmbench/results/expansion-20261112")
from analyze500 import hit, hit_all, rr, boot, paired
H = Path(__file__).resolve().parent; EXP = Path("/workspace/llmbench/results/expansion-20261112")
SYSTEM = sys.argv[1]
rows = {}
for f in glob.glob(str(H / f"rm__{SYSTEM}.shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if "error" not in r: rows[r["qid"]] = r
orig = {}
for f in glob.glob(str(EXP / f"lme500__{SYSTEM}.shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if "error" not in r: orig[r["qid"]] = r["ranked"]
qs = sorted(q for q in rows if q in orig)
def R(v, q): return {"ranked": rows[q]["variants"][v], "answers": rows[q]["answers"]}
out = {"system": SYSTEM, "n": len(qs), "baseline_identical_to_lme500": sum(rows[q]["variants"]["baseline"] == orig[q] for q in qs), "variants": {}}
types = sorted({rows[q]["qtype"] for q in qs})
for v in rows[qs[0]]["variants"]:
    o = {"R@5": round(float(np.mean([hit(R(v, q)) for q in qs])), 4), "Rall@5": round(float(np.mean([hit_all(R(v, q)) for q in qs])), 4),
         "MRR@10": round(float(np.mean([rr(R(v, q)) for q in qs])), 4), "MRR_ci": boot([rr(R(v, q)) for q in qs]),
         "changed_q": sum(rows[q]["variants"][v] != rows[q]["variants"]["baseline"] for q in qs),
         "ms_p50": round(float(np.median([rows[q]["ms"][v] for q in qs])), 2), "ms_p95": round(float(np.percentile([rows[q]["ms"][v] for q in qs], 95)), 2)}
    if v != "baseline":
        o["dMRR"] = paired([rr(R(v, q)) for q in qs], [rr(R("baseline", q)) for q in qs])
        o["dR@5"] = paired([hit(R(v, q)) for q in qs], [hit(R("baseline", q)) for q in qs])
        o["dRall@5"] = paired([hit_all(R(v, q)) for q in qs], [hit_all(R("baseline", q)) for q in qs])
    o["by_type"] = {}
    for t in types:
        tq = [q for q in qs if rows[q]["qtype"] == t]
        x = {"n": len(tq), "MRR@10": round(float(np.mean([rr(R(v, q)) for q in tq])), 4), "Rall@5": round(float(np.mean([hit_all(R(v, q)) for q in tq])), 4)}
        if v != "baseline": x["dMRR"] = paired([rr(R(v, q)) for q in tq], [rr(R("baseline", q)) for q in tq])
        o["by_type"][t] = x
    out["variants"][v] = o
json.dump(out, open(H / f"rm__{SYSTEM}.summary.json", "w"), indent=1)
print(SYSTEM, "n", out["n"], "baseline identical to LME-500 run:", out["baseline_identical_to_lme500"])
for v, o in out["variants"].items():
    s = f"{v:16s} R@5 {o['R@5']:.4f} Rall@5 {o['Rall@5']:.4f} MRR {o['MRR@10']:.4f} {o['MRR_ci']} changed {o['changed_q']} ms p50/p95 {o['ms_p50']}/{o['ms_p95']}"
    if "dMRR" in o: d = o["dMRR"]; s += f" | dMRR {d['diff']:+.4f} {d['ci']} {'SIG' if d['sig'] else 'ns'} p={d['p_boot_2s']} b/w {d['better']}/{d['worse']} | dRall {o['dRall@5']['diff']:+.4f} {o['dRall@5']['ci']} {'SIG' if o['dRall@5']['sig'] else 'ns'} | dR@5 {o['dR@5']['diff']:+.4f}"
    print(s); print("    " + "; ".join(f"{t} {x['MRR@10']:.3f}" + (f" ({x['dMRR']['diff']:+.3f}{'*' if x['dMRR']['sig'] else ''})" if 'dMRR' in x else "") for t, x in o["by_type"].items()))
