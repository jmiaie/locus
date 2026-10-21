"""Interleaved latency check: same process, same queries, analyzers alternated per round (5 rounds)."""
import json, sys, time, tempfile, statistics, os
sys.path.insert(0, "/workspace/stem/harness")
from pathlib import Path
from common import load_beir
from locus.core import LocusEngine
import numpy as np
out = {}
for bench in ("scifact", "nfcorpus"):
    corpus, queries, qrels, qids = load_beir(bench)
    with tempfile.TemporaryDirectory(dir="/workspace/stem") as wd:
        dd = Path(wd) / "docs"; dd.mkdir()
        for i, t, x in corpus: (dd / f"{i.replace('/', '_')}.md").write_text(f"# {t}\n\n{x}" if t else x)
        eng = {}
        for a in ("plain", "porter", "porter+english"):
            e = LocusEngine(store_path=Path(wd) / f".l_{a}", analyzer=a); t0 = time.perf_counter(); e.index(dd)
            eng[a] = (e, time.perf_counter() - t0)
        times = {a: {q: [] for q in qids} for a in eng}
        for rnd in range(5):
            order = list(eng) if rnd % 2 == 0 else list(eng)[::-1]
            for a in order:
                e = eng[a][0]
                for q in qids:
                    t = time.perf_counter(); e.retrieve(queries[q], limit=30, mode="bm25", use_cache=False); times[a][q].append((time.perf_counter() - t) * 1000)
        res = {}
        for a in eng:
            med = [statistics.median(v) for v in times[a].values()]
            res[a] = {"index_s": round(eng[a][1], 2), "p50_ms": round(float(np.percentile(med, 50)), 3), "p95_ms": round(float(np.percentile(med, 95)), 3)}
        for a in res:
            res[a]["p50_ratio_vs_plain"] = round(res[a]["p50_ms"] / res["plain"]["p50_ms"], 2)
            res[a]["p95_ratio_vs_plain"] = round(res[a]["p95_ms"] / res["plain"]["p95_ms"], 2)
        out[bench] = res; print(bench, json.dumps(res))
json.dump(out, open("/workspace/stem/latency.json", "w"), indent=1)
