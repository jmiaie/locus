"""
RM3 pseudo-relevance feedback: LLM-free query expansion.

1. Run the query with BM25 and take the top *fb_docs* chunks as
   pseudo-relevant.
2. Build a relevance model from them. Each term's weight is
   ``p(w|R) = sum_D w_D * tf(w, D) / |D|``, where ``w_D`` is the chunk's BM25
   score normalised over the feedback set.
3. Keep the top *fb_terms* terms and interpolate them with the original query
   model: ``p'(w) = orig_weight * p(w|Q) + (1 - orig_weight) * p(w|R)``.
4. Score candidates with the weighted query
   ``score(D) = sum_w p'(w) * bm25_w(D)``. The candidates are the union of
   the top-*depth* chunks for the original query and for the expanded term
   set.

FTS5 cannot weight query terms, so step 4 rescans per-term BM25
contributions (:meth:`Corpus.score_terms`) restricted to the candidate chunks.
The terms and statistics are the same as for plain BM25. Everything is
deterministic: the ordering among equal scores is by term and then by index
order.

Reference: Abdul-Jaleel et al., "UMass at TREC 2004" (RM3); Lv & Zhai 2009.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass

from ..memory.corpus import Corpus


@dataclass(frozen=True)
class RM3Config:
    fb_docs: int = 10
    fb_terms: int = 20
    orig_weight: float = 0.5
    depth: int = 200

    def __post_init__(self):
        if self.fb_docs < 1 or self.fb_terms < 1 or self.depth < 1:
            raise ValueError("rm3 fb_docs, fb_terms and depth must be >= 1")
        if not 0.0 <= self.orig_weight <= 1.0:
            raise ValueError("rm3 orig_weight must be in [0, 1]")

    @classmethod
    def from_value(cls, v: "RM3Config | dict | None") -> "RM3Config":
        if v is None:
            return cls()
        if isinstance(v, cls):
            return v
        if isinstance(v, dict):
            unknown = set(v) - set(cls.__dataclass_fields__)
            if unknown:
                raise ValueError(f"Unknown rm3 option(s) {sorted(unknown)}")
            return cls(**v)
        raise TypeError("rm3 config must be a dict or RM3Config")

    def key(self) -> str:
        d = asdict(self)
        return ",".join(f"{k}={d[k]}" for k in sorted(d))


def _feedback_terms(corpus: Corpus, chunk_ids: list[str]) -> dict[str, list[str]]:
    """Analysed term stream of each chunk (stored FTS5 body when available)."""
    out: dict[str, list[str]] = {}
    if corpus.backend == "fts5":
        order = corpus.chunk_order(chunk_ids)
        if order:
            ph = ",".join("?" * len(order))
            try:
                with corpus._conn() as conn:
                    rows = conn.execute(
                        f"SELECT rowid, body FROM chunk_fts WHERE rowid IN ({ph})", list(order.values())
                    ).fetchall()
                by_row = {r: b for r, b in rows}
                for cid, r in order.items():
                    if r in by_row:
                        out[cid] = by_row[r].split()
                return out
            except sqlite3.OperationalError:
                pass
    for cid, ch in corpus.get_chunks_batch(chunk_ids).items():
        out[cid] = corpus.analyze(ch.content)
    return out


def expansion_model(corpus: Corpus, query: str, cfg: RM3Config) -> dict[str, float]:
    """The interpolated RM3 query model ``{term: weight}`` (weights sum to 1)."""
    q = corpus.analyze(query)
    if not q:
        return {}
    qm = {t: c / len(q) for t, c in Counter(q).items()}
    fb = corpus.search_bm25(q, limit=cfg.fb_docs)
    if not fb:
        return qm
    total = sum(max(s, 0.0) for _, s in fb) or 1.0
    streams = _feedback_terms(corpus, [cid for cid, _ in fb])
    rel: dict[str, float] = {}
    for cid, s in fb:
        terms = streams.get(cid)
        if not terms:
            continue
        w = max(s, 0.0) / total
        n = len(terms)
        for t, c in Counter(terms).items():
            rel[t] = rel.get(t, 0.0) + w * c / n
    top = sorted(rel.items(), key=lambda kv: (-kv[1], kv[0]))[: cfg.fb_terms]
    z = sum(v for _, v in top) or 1.0
    relm = {t: v / z for t, v in top}
    lam = cfg.orig_weight
    model = {t: lam * qm.get(t, 0.0) + (1 - lam) * relm.get(t, 0.0) for t in set(qm) | set(relm)}
    return {t: w for t, w in model.items() if w > 0}


def rm3_search(corpus: Corpus, query: str, limit: int, cfg: RM3Config | dict | None = None) -> list[tuple[str, float]]:
    """``[(chunk_id, score)]`` ranked by the RM3-expanded query (see module doc)."""
    cfg = RM3Config.from_value(cfg)
    if limit <= 0:
        return []
    model = expansion_model(corpus, query, cfg)
    if not model:
        return []
    q = corpus.analyze(query)
    depth = max(cfg.depth, limit)
    cands = [cid for cid, _ in corpus.search_bm25(q, limit=depth)]
    terms = sorted(model, key=lambda t: (-model[t], t))
    cands += [cid for cid, _ in corpus.search_bm25(terms, limit=depth)]
    cands = list(dict.fromkeys(cands))
    if not cands:
        return []
    per = corpus.score_terms(terms, cands)
    scores = {cid: 0.0 for cid in cands}
    for t in terms:
        w = model[t]
        for cid, s in per.get(t, {}).items():
            scores[cid] += w * s
    order = corpus.chunk_order(cands)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], order.get(kv[0], 0)))
    return [(cid, s) for cid, s in ranked if s > 0][:limit]
