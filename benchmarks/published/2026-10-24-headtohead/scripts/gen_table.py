import json
S = json.load(open("/workspace/llmbench/results/p2_summary.json"))
NAMES = {"locus": "Locus (FTS5 BM25, default)", "ompa": "OMPA (MiniLM + keyword)", "mem0": "mem0 2.2.1 (LLM extraction, infer=True)",
         "langmem": "LangMem 0.0.30 (memory manager)", "graphiti": "Graphiti 0.31.2 / Zep OSS (default `search`)",
         "graphiti_relaxed": "Graphiti, same graph, hybrid RRF with cosine floor removed", "letta": "Letta 0.16.8 agent (archival_memory_search tool calls)",
         "letta_direct": "Letta 0.16.8 archival search API (no LLM at query)"}
rows = ["| System | n | R@5 [95% CI] | R_all@5 | MRR@10 | ΔMRR vs Locus, same q [95% CI] | ΔMRR vs OMPA, same q | Query p50 | Ingest p50 / haystack | LLM $ / question |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for k, nm in NAMES.items():
    o = S.get(k, {})
    if "n" not in o: rows.append(f"| {nm} | 0 | not run | | | | | | | |"); continue
    def dv(ref):
        v = o.get(f"vs_{ref}")
        if not v: return "–"
        return f"{v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] {'**sig**' if v['sig'] else 'ns'}"
    usd = o.get("usd_per_question")
    ing = o.get("ingest_p50_s_per_haystack")
    rows.append(f"| {nm} | {o['n']} | {o['R@5']:.3f} [{o['R@5_ci'][0]:.2f}, {o['R@5_ci'][1]:.2f}] | {o['Rall@5']:.3f} | {o['MRR@10']:.3f} | {dv('locus')} | {dv('ompa')} | "
                f"{o['query_p50_ms']:,.0f} ms | {ing:,.1f} s | {'$%.3f' % usd if usd is not None else ('$0' if k in ('locus','ompa') else 'incl. above')} |")
open("/workspace/llmbench/p2_table.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
