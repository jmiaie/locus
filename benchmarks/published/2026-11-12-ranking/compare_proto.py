"""Per-question agreement between implemented variants (rm__SYSTEM) and the prototype variants (proto_eval.py)."""
import sys, json, glob
sys.path.insert(0, "/workspace/llmbench/results/expansion-20261112")
from proto_eval import variants
from analyze500 import rr
from pathlib import Path
EXP = Path("/workspace/llmbench/results/expansion-20261112"); H = Path(__file__).resolve().parent
SYSTEM = sys.argv[1]
cache = {}
for f in ("proto_cache.jsonl", "proto_cache_rev.jsonl"):
    for l in open(EXP / f): r = json.loads(l); cache[r["qid"]] = r
base500 = {}
for f in glob.glob(str(EXP / f"lme500__{SYSTEM}.shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if "error" not in r: base500[r["qid"]] = r["ranked"]
rows = {}
for f in glob.glob(str(H / f"rm__{SYSTEM}.shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if "error" not in r: rows[r["qid"]] = r
MAP = {"temporal": "+ time-window", "dense": "+ MiniLM (RRF)", "dense_temporal": "+ MiniLM + time-window", "dense_advice": "+ advice-query router -> MiniLM RRF (post-hoc)"}
out = {}
qs = [q for q in rows if q in cache and q in base500]
for v, pv in MAP.items():
    if v not in rows[qs[0]]["variants"]: continue
    same = top1 = rrsame = 0
    for q in qs:
        P, _, _ = variants(cache[q], base500[q]); a, b = rows[q]["variants"][v], P[pv]
        same += a == b; top1 += a[:1] == b[:1]
        rrsame += rr({"ranked": a, "answers": cache[q]["answers"]}) == rr({"ranked": b, "answers": cache[q]["answers"]})
    out[v] = {"prototype_variant": pv, "n": len(qs), "identical_top10": same, "same_top1": top1, "same_reciprocal_rank": rrsame}
print(json.dumps(out, indent=1)); json.dump(out, open(H / f"rm__{SYSTEM}.vs_proto.json", "w"), indent=1)
