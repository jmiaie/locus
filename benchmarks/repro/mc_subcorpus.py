"""Monte Carlo over randomized sub-corpora of BEIR SciFact (same design as the 2026-10-20 harness):
each seed samples 100 test queries; corpus = their relevant docs + random distractors up to 2000 docs.
Systems: locus_bm25 (new default), locus_hybrid (new, default params), rank_bm25. Seeds 1..10."""
import os
BEIR = os.environ.get("LOCUS_BEIR_DIR", "data/beir")
OUT = os.environ.get("LOCUS_BENCH_OUT", "benchmarks/results/repro")
TMP = os.environ.get("LOCUS_BENCH_TMP") or None
os.makedirs(OUT, exist_ok=True)
import json, csv, random, tempfile, math, heapq, logging, shutil
from pathlib import Path
logging.disable(logging.WARNING)
from locus.core import LocusEngine
from locus.memory.corpus import tokenize
from rank_bm25 import BM25Okapi
d = Path(BEIR) / "scifact"
corpus = {json.loads(l)["_id"]: json.loads(l) for l in open(d/"corpus.jsonl")}
queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(d/"queries.jsonl")}
qrels = {}
for qid, did, s in list(csv.reader(open(d/"qrels/test.tsv"), delimiter="\t"))[1:]:
    if int(s) > 0: qrels.setdefault(qid, {})[did] = int(s)
def ndcg(r, rel):
    dcg = sum(1/math.log2(i+2) for i, x in enumerate(r[:10]) if x in rel); idcg = sum(1/math.log2(i+2) for i in range(min(10, len(rel)))); return dcg/idcg
out = []
for seed in range(1, 11):
    rng = random.Random(seed); qs = rng.sample(sorted(qrels), 100)
    keep = {x for q in qs for x in qrels[q]}; others = [x for x in corpus if x not in keep]
    keep |= set(rng.sample(others, 2000 - len(keep))); ids = sorted(keep)
    tmp = Path(tempfile.mkdtemp(dir=TMP)); (tmp/"docs").mkdir()
    for i in ids: (tmp/"docs"/f"{i}.md").write_text(f"# {corpus[i]['title']}\n\n{corpus[i]['text']}\n")
    eng = LocusEngine(store_path=tmp/".locus"); eng.index(tmp/"docs")
    stem = lambda ch: list(dict.fromkeys(Path(c.doc_path).stem for c in ch))[:10]
    bm = BM25Okapi([tokenize(corpus[i]["title"] + " " + corpus[i]["text"]) for i in ids])
    res = {"seed": seed}
    for name, fn in [("locus_bm25", lambda q: stem(eng.retrieve(q, limit=40, use_cache=False))),
                     ("locus_hybrid", lambda q: stem(eng.retrieve(q, limit=40, use_cache=False, mode="hybrid"))),
                     ("rank_bm25", lambda q: [ids[j] for j in heapq.nlargest(10, range(len(ids)), key=bm.get_scores(tokenize(q)).__getitem__)])]:
        nd = [ndcg(fn(queries[q]), qrels[q]) for q in qs]
        res[name] = {"ndcg@10": sum(nd)/len(nd), "per_query": nd}
    out.append(res); print(json.dumps({k: (v if not isinstance(v, dict) else round(v["ndcg@10"], 4)) for k, v in res.items()}), flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
json.dump(out, open(f"{OUT}/mc_subcorpus_scifact_new.json", "w"))
