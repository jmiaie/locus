import json
S = json.load(open("/workspace/llmbench/results/p2_gemini_summary.json"))
NM = {"graphiti_gemini": "Graphiti 0.31.2, `gemini-3.5-flash-lite` (default `search`)", "graphiti_gemini_relaxed": "Graphiti, `gemini-3.5-flash-lite`, same graph, cosine floor removed"}
REFN = {"graphiti": "Graphiti gpt-4o-mini (default)", "graphiti_relaxed": "Graphiti gpt-4o-mini (floor removed)", "locus": "Locus", "ompa": "OMPA"}
def f(v): return f"{v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] {'**sig**' if v['sig'] else 'ns'}, p{"<0.0001" if v['p_boot_2s'] == 0 else "=" + str(v["p_boot_2s"])}"
L = ["| System (different LLM, not directly comparable to the gpt-4o-mini rows) | n | R@5 [95% CI] | R_all@5 | MRR@10 | Query p50 | Ingest p50 / haystack | LLM $ |", "|---|---|---|---|---|---|---|---|"]
for k, nm in NM.items():
    o = S[k]; L.append(f"| {nm} | {o['n']} | {o['R@5']:.3f} [{o['R@5_ci'][0]:.2f}, {o['R@5_ci'][1]:.2f}] | {o['Rall@5']:.3f} | {o['MRR@10']:.3f} | {o['query_p50_ms']:,.0f} ms | {o['ingest_p50_s_per_haystack']:,.1f} s* | $0 (free tier) |")
L += ["", "| Gemini row | vs (same 30 q) | ref R@5 / MRR | ΔMRR@10 [95% CI], bootstrap p | ΔR@5 [95% CI], bootstrap p | better / worse (MRR) |", "|---|---|---|---|---|---|"]
for k, nm in NM.items():
    o = S[k]
    for ref in [r[3:] for r in o if r.startswith("vs_")]:
        v = o[f"vs_{ref}"]; L.append(f"| {'default' if k == 'graphiti_gemini' else 'floor removed'} | {REFN[ref]} | {v['ref_R@5_same_q']:.3f} / {v['ref_MRR_same_q']:.3f} | {f(v)} | {f(v['R@5_diff'])} | {v['better']} / {v['worse']} |")
print("\n".join(L)); open("/workspace/llmbench/p2_gemini_table.md", "w").write("\n".join(L) + "\n")
