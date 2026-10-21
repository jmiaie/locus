"""Metrics + seeded bootstrap CIs for the new BEIR runs, dev-only config selection, paired tests.
Before-numbers come from the 2026-10-20 PM rerun on Locus main 62156b5 (same data, same metric code)."""
import os
BEIR = os.environ.get("LOCUS_BEIR_DIR", "data/beir")
OUT = os.environ.get("LOCUS_BENCH_OUT", "benchmarks/results/repro")
TMP = os.environ.get("LOCUS_BENCH_TMP") or None
os.makedirs(OUT, exist_ok=True)
import json, math, csv
import numpy as np
B = 10000; SEED = 20261020
R = OUT
OLD = os.environ.get("LOCUS_BENCH_BEFORE")  # optional dir with beir_<ds>_lexical.json from the old pipeline

def per_query(ranked, rel, k=10):
    r = ranked[:k]
    dcg = sum((2**rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(r))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    rr = next((1 / (i + 1) for i, d in enumerate(r) if d in rel), 0.0)
    return {"ndcg@10": dcg / idcg if idcg else 0, "mrr@10": rr,
            "recall@10": len(set(ranked[:10]) & set(rel)) / len(rel)}

def boot_mean(x):
    x = np.asarray(x, float); rng = np.random.default_rng(SEED)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m <= 0).mean())

def boot_pct(L, p):
    L = np.asarray(L, float); rng = np.random.default_rng(SEED)
    bs = np.percentile(L[rng.integers(0, len(L), (B, len(L)))], p, axis=1)
    return float(np.percentile(L, p)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

out = {"bootstrap_B": B, "seed": SEED, "datasets": {}}
data = {ds: json.load(open(f"{R}/beir_{ds}_new.json")) for ds in ("scifact", "nfcorpus")}
# ---- dev grid (selection uses DEV ONLY)
grid = {}
for ds, j in data.items():
    qr = j["qrels"]["dev"]; qids = sorted(qr)
    for key, run in j["runs"].items():
        split, name = key.split(":", 1)
        if split != "dev": continue
        grid.setdefault(name, {})[ds] = float(np.mean([per_query(run["ranked"][q], qr[q])["ndcg@10"] for q in qids]))
hyb = {n: v for n, v in grid.items() if n.startswith("locus_hybrid")}
best = max(sorted(hyb), key=lambda n: (np.mean(list(hyb[n].values()))))
out["dev_grid_ndcg@10"] = grid
out["selected_hybrid_config_by_dev"] = best
print("DEV grid (mean nDCG@10 scifact-train / nfcorpus-dev):")
for n in sorted(grid, key=lambda n: -np.mean(list(grid[n].values()))):
    print(f"  {n:28s} scifact={grid[n]['scifact']:.4f} nfcorpus={grid[n]['nfcorpus']:.4f}")
print("selected:", best)

rows = []; diffs = []
for ds, j in data.items():
    qr = j["qrels"]["test"]; qids = sorted(qr)
    systems = {}
    if OLD:
        old = json.load(open(f"{OLD}/beir_{ds}_lexical.json"))
        assert sorted(old["qrels"]) == qids, "query sets differ"
        systems["before: locus full_6signal (main 62156b5)"] = (old["runs"]["locus_full_6signal"]["ranked"], old["runs"]["locus_full_6signal"]["latency_ms"])
        systems["before: locus bm25_only (main 62156b5)"] = (old["runs"]["locus_bm25_only"]["ranked"], old["runs"]["locus_bm25_only"]["latency_ms"])
    NEW_BM25 = "after: locus default (bm25, FTS5)"
    NEW_HYB = f"after: locus hybrid ({best.replace('locus_hybrid_', '')})"
    RB = "rank_bm25 (BM25Okapi baseline)"
    systems[NEW_BM25] = (j["runs"]["test:locus_bm25"]["ranked"], j["runs"]["test:locus_bm25"]["latency_ms"])
    systems[NEW_HYB] = (j["runs"][f"test:{best}"]["ranked"], j["runs"][f"test:{best}"]["latency_ms"])
    systems[RB] = (j["runs"]["test:rank_bm25"]["ranked"], j["runs"]["test:rank_bm25"]["latency_ms"])
    per = {n: {q: per_query(r[q], qr[q]) for q in qids} for n, (r, _) in systems.items()}
    for n, (r, lat) in systems.items():
        row = {"dataset": ds, "system": n, "n_queries": len(qids)}
        for m in ("ndcg@10", "recall@10", "mrr@10"):
            mean, lo, hi, _ = boot_mean([per[n][q][m] for q in qids])
            row[m] = round(mean, 4); row[m + "_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        for p in (50, 95):
            v, lo, hi = boot_pct([lat[q] for q in qids], p)
            row[f"p{p}_ms"] = round(v, 2); row[f"p{p}_ms_ci95"] = f"[{lo:.2f}, {hi:.2f}]"
        rows.append(row)
    pairs = [(NEW_BM25, RB), (NEW_HYB, NEW_BM25), (NEW_HYB, RB)]
    pairs += [(NEW_BM25, n) for n in systems if n.startswith("before:")]
    for a, b in pairs:
        dvec = [per[a][q]["ndcg@10"] - per[b][q]["ndcg@10"] for q in qids]
        mean, lo, hi, p_le0 = boot_mean(dvec)
        diffs.append({"dataset": ds, "a": a, "b": b, "metric": "ndcg@10", "mean_diff": round(mean, 4),
                      "ci95": f"[{lo:.4f}, {hi:.4f}]", "p_boot(diff<=0)": round(p_le0, 4),
                      "identical_rankings": sum(systems[a][0][q][:10] == systems[b][0][q][:10] for q in qids)})
    out["datasets"][ds] = {"meta": j["meta"]}
out["summary"] = rows; out["paired"] = diffs
json.dump(out, open(f"{R}/beir_metrics_new.json", "w"), indent=1)
with open(f"{R}/beir_metrics_new.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print()
for r in rows: print(r["dataset"], "|", r["system"], "| nDCG", r["ndcg@10"], r["ndcg@10_ci95"], "| p50", r["p50_ms"], "p95", r["p95_ms"])
print()
for d_ in diffs: print(d_)
