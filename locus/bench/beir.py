"""
BEIR-format retrieval benchmark (ROADMAP: "Now — public benchmark harness").

Reads a BEIR dataset that is already on disk — nothing is downloaded:

    <dataset>/corpus.jsonl        {"_id", "title", "text"} per line
    <dataset>/queries.jsonl       {"_id", "text"} per line
    <dataset>/qrels/<split>.tsv   query-id  corpus-id  score   (header row)

and scores Locus, rank_bm25 (optional extra ``bench``) and, once implemented,
a small embedding baseline with the same metrics: nDCG@10, Recall@k and MRR.

Usage:
    python -m locus.bench.beir /data/beir/scifact --split test --max-queries 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class BeirDataset:
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


@dataclass
class BeirResult:
    system: str
    ndcg_at_10: float
    recall: dict[int, float]
    mrr: float
    avg_query_ms: float
    num_queries: int
    notes: str = ""
    per_query: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "ndcg@10": round(self.ndcg_at_10, 4),
            **{f"recall@{k}": round(v, 4) for k, v in sorted(self.recall.items())},
            "mrr": round(self.mrr, 4),
            "avg_query_ms": round(self.avg_query_ms, 2),
            "num_queries": self.num_queries,
            "notes": self.notes,
        }


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_beir(
    dataset_dir: str | Path,
    split: str = "test",
    max_queries: int | None = None,
) -> BeirDataset:
    """Load a local BEIR dataset.  Queries without qrels are dropped."""
    root = Path(dataset_dir)
    corpus = {
        str(d["_id"]): (f"{d.get('title', '')}\n\n{d.get('text', '')}").strip()
        for d in _read_jsonl(root / "corpus.jsonl")
    }
    qrels: dict[str, dict[str, int]] = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if i == 0 and not parts[-1].lstrip("-").isdigit():
                continue  # header
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0 and did in corpus:
                qrels.setdefault(qid, {})[did] = score
    queries = {
        str(q["_id"]): q["text"]
        for q in _read_jsonl(root / "queries.jsonl")
        if str(q["_id"]) in qrels
    }
    qids = sorted(queries)
    if max_queries is not None:
        qids = qids[:max_queries]
    return BeirDataset(
        corpus=corpus,
        queries={q: queries[q] for q in qids},
        qrels={q: qrels[q] for q in qids},
    )


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = sum(
        (2 ** rels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k])
    )
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def score_run(
    system: str,
    dataset: BeirDataset,
    retrieve_fn: Callable[[str], list[str]],
    k_values: tuple[int, ...] = (1, 5, 10, 100),
    notes: str = "",
) -> BeirResult:
    ndcg_sum = rr_sum = 0.0
    recall_sum = {k: 0.0 for k in k_values}
    times: list[float] = []
    per_query: dict[str, list[str]] = {}
    for qid, text in dataset.queries.items():
        t0 = time.perf_counter()
        ranked = retrieve_fn(text)
        times.append((time.perf_counter() - t0) * 1000)
        per_query[qid] = ranked
        rels = dataset.qrels[qid]
        ndcg_sum += ndcg_at_k(ranked, rels, 10)
        for k in k_values:
            recall_sum[k] += len(set(ranked[:k]) & rels.keys()) / len(rels)
        rr_sum += next((1.0 / r for r, d in enumerate(ranked, 1) if d in rels), 0.0)
    n = len(dataset.queries) or 1
    return BeirResult(
        system=system,
        ndcg_at_10=ndcg_sum / n,
        recall={k: v / n for k, v in recall_sum.items()},
        mrr=rr_sum / n,
        avg_query_ms=sum(times) / len(times) if times else 0.0,
        num_queries=len(dataset.queries),
        notes=notes,
        per_query=per_query,
    )


def _file_name(doc_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", doc_id)[:60]
    return f"{safe}-{hashlib.sha1(doc_id.encode()).hexdigest()[:8]}.md"


def run_locus(
    dataset: BeirDataset, store_dir: str | Path, limit: int = 100, mode: str = "bm25",
) -> BeirResult:
    """Index the corpus as one markdown file per document and score Locus."""
    from ..core import LocusEngine

    store_dir = Path(store_dir)
    doc_dir = store_dir / "docs"
    doc_dir.mkdir(parents=True, exist_ok=True)
    name_to_id: dict[str, str] = {}
    for doc_id, text in dataset.corpus.items():
        name = _file_name(doc_id)
        name_to_id[name] = doc_id
        (doc_dir / name).write_text(text, encoding="utf-8")

    engine = LocusEngine(store_path=store_dir / ".locus")
    engine.index(str(doc_dir))

    def retrieve(query: str) -> list[str]:
        seen: list[str] = []
        for chunk in engine.retrieve(query, limit=limit, use_cache=False, mode=mode):
            doc_id = name_to_id.get(Path(chunk.doc_path).name)
            if doc_id and doc_id not in seen:
                seen.append(doc_id)
        return seen

    return score_run(
        f"locus_{mode}", dataset, retrieve,
        notes=f"LocusEngine.retrieve(mode={mode!r})",
    )


def run_rank_bm25(dataset: BeirDataset, limit: int = 100) -> BeirResult:
    """rank_bm25 BM25Okapi baseline.  Requires ``pip install "locus-rag[bench]"``."""
    from rank_bm25 import BM25Okapi

    from .baseline import BaselineBench

    ids = list(dataset.corpus)
    bm25 = BM25Okapi([BaselineBench._tokenize(dataset.corpus[i]) for i in ids])

    def retrieve(query: str) -> list[str]:
        scores = bm25.get_scores(BaselineBench._tokenize(query))
        top = sorted(range(len(ids)), key=lambda i: (-scores[i], i))[:limit]
        return [ids[i] for i in top]

    return score_run("rank_bm25", dataset, retrieve, notes="BM25Okapi k1=1.5 b=0.75")


def run_embedding_baseline(dataset: BeirDataset, model: str = "all-MiniLM-L6-v2") -> BeirResult:
    raise NotImplementedError(
        "Embedding baseline is not implemented yet; see docs/ROADMAP.md, "
        "'Now: benchmark harness', item N3 (small embedding baseline)."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score Locus on a local BEIR dataset")
    parser.add_argument("dataset_dir")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--out", default=None, help="Write JSON results here")
    args = parser.parse_args(argv)

    dataset = load_beir(args.dataset_dir, split=args.split, max_queries=args.max_queries)
    results: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        results.append(run_locus(dataset, tmp, mode="bm25").to_dict())
    with tempfile.TemporaryDirectory() as tmp:
        results.append(run_locus(dataset, tmp, mode="hybrid").to_dict())
    try:
        results.append(run_rank_bm25(dataset).to_dict())
    except ImportError:
        results.append({"system": "rank_bm25", "notes": "skipped: pip install 'locus-rag[bench]'"})

    report = {
        "dataset": str(Path(args.dataset_dir).name),
        "split": args.split,
        "corpus_docs": len(dataset.corpus),
        "queries": len(dataset.queries),
        "results": results,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
