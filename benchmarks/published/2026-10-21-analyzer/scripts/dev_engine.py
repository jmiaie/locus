"""Dev-split check with the real engine (SciFact train, NFCorpus dev) for the three analyzers, BM25 mode."""
import json, csv, sys, math, tempfile
from pathlib import Path
from locus.core import LocusEngine
D = Path('/workspace/ompa-locus-20261020/data')
def ndcg10(ranked, rel):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
out = {}
for ds, split in (("scifact", "train"), ("nfcorpus", "dev")):
    corpus = [json.loads(l) for l in open(D / ds / "corpus.jsonl")]
    qs = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(D / ds / "queries.jsonl")}
    qrels = {}
    r = csv.reader(open(D / ds / "qrels" / f"{split}.tsv"), delimiter="\t"); next(r)
    for q, d, s in r:
        if int(s) > 0: qrels.setdefault(q, {})[d] = int(s)
    qids = sorted(q for q in qrels if q in qs)
    with tempfile.TemporaryDirectory(dir="/workspace/stem") as wd:
        dd = Path(wd) / "docs"; dd.mkdir(); back = {}
        for c in corpus:
            n = c["_id"].replace("/", "_"); back[n] = c["_id"]
            t = c.get("title") or ""; (dd / f"{n}.md").write_text(f"# {t}\n\n{c['text']}" if t else c["text"])
        for a in ("plain", "porter", "porter+english"):
            e = LocusEngine(store_path=Path(wd) / f".l_{a}", analyzer=a); e.index(dd)
            sc = []
            for q in qids:
                ch = e.retrieve(qs[q], limit=30, mode="bm25", use_cache=False)
                ranked = list(dict.fromkeys(back.get(Path(c.doc_path).stem) for c in ch))[:10]
                sc.append(ndcg10(ranked, qrels[q]))
            out[f"{ds}-{split}|{a}"] = round(sum(sc) / len(sc), 4); print(ds, split, len(qids), a, out[f"{ds}-{split}|{a}"], flush=True)
json.dump(out, open("/workspace/stem/dev_engine.json", "w"), indent=1)
