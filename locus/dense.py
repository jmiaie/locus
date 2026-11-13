"""
Optional dense re-rank (``pip install "locus-rag[dense]"``). Off by default.

A local sentence-embedding model (default ``all-MiniLM-L6-v2``, 22M
parameters, CPU) scores each candidate *document* by the best cosine
similarity between the query and ~200-word windows of the document.  That
dense ranking is fused with the BM25 document ranking by Reciprocal Rank
Fusion (k=60, equal weights).  It helps most when a query shares few words
with the passage that answers it (e.g. "Can you suggest a hotel for my trip?"
vs. a conversation where the user stated their travel preferences).

    from locus.dense import DenseReranker
    engine = LocusEngine(".locus", reranker=DenseReranker())
    engine.retrieve("any tips for my trip to Miami?")             # BM25 + dense
    engine.retrieve("any tips for my trip to Miami?", rerank=False)  # BM25 only

How it works:
- The engine hands the reranker its first-stage pool (``depth`` chunks,
  default 300).  Candidate documents are the distinct documents of that pool,
  in pool order (the BM25 document ranking).
- Each candidate's full indexed text (its sections, from the corpus) is split
  into windows of ``window_words`` words with stride ``stride_words``; the
  document's dense score is the max cosine over its windows.  Window
  embeddings are cached per (document, text hash) in a bounded LRU.
- Output: the pool's chunks grouped by fused document rank (stable within a
  document), ``score`` = the document's fused RRF score, provenance
  ``+dense``.

``route="advice"`` (POST-HOC, not a default): fuse only when the query looks
like an advice/recommendation request (recommend, suggest, tips, ideas,
advice, any thoughts); every other query keeps the BM25 ranking and never
loads the model.  This rule was written after looking at per-question-type
results on LongMemEval-S, where it fires on 25 of 30 single-session-preference
questions and on no other question type, i.e. it is close to a question-type
oracle on that dataset.  It has not been validated on other data.

Only ``sentence-transformers`` (and so PyTorch) is imported, and only when
the model is first used; the Locus core never imports this module.  Model
weights download from the Hugging Face Hub on first use (set
``HF_HUB_OFFLINE=1`` once cached); telemetry is disabled unless you set it.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .retrieval.temporal import reorder_by_docs, rrf_doc_order

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ROUTES = (None, "advice")
#: Advice/recommendation-style query detector used by ``route="advice"`` (post-hoc, see module docstring).
ADVICE_RE = re.compile(r"\b(recommend|recommendations?|suggest|suggestions?|tips|ideas|advice|any thoughts)\b", re.I)


def text_windows(text: str, window_words: int = 200, stride_words: int = 150) -> list[str]:
    """Overlapping word windows of *text* (one window when it is short)."""
    w = text.split()
    if len(w) <= window_words:
        return [" ".join(w)]
    return [" ".join(w[i:i + window_words]) for i in range(0, max(1, len(w) - window_words + stride_words), stride_words)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _normalize(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [float(x) / n for x in v]


@dataclass
class DenseReranker:
    model: str = DEFAULT_MODEL
    depth: int = 300
    window_words: int = 200
    stride_words: int = 150
    batch_size: int = 64
    device: str = "cpu"
    threads: int | None = None
    rrf_k: int = 60
    route: str | None = None
    #: optional ``texts -> vectors`` callable replacing the model (tests, custom encoders); vectors are L2-normalised here.
    encoder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None
    cache_docs: int = 4096
    _st: object = field(default=None, init=False, repr=False)
    _lock: object = field(default_factory=threading.Lock, init=False, repr=False)
    _cache: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _engine: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {self.route!r}")
        if not 0 < self.stride_words <= self.window_words:
            raise ValueError("need 0 < stride_words <= window_words")

    def key(self) -> str:
        enc = "custom" if self.encoder is not None else self.model
        return f"dense|{enc}|{self.depth}|{self.window_words}|{self.stride_words}|{self.rrf_k}|{self.route}"

    def bind(self, engine) -> None:
        """Called by :class:`~locus.core.LocusEngine`; gives access to full document text."""
        self._engine = engine

    def applies(self, query: str) -> bool:
        return self.route is None or bool(ADVICE_RE.search(query or ""))

    # -- encoding -------------------------------------------------------
    def _load(self):
        if self._st is None:
            with self._lock:
                if self._st is None:
                    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
                    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as e:  # pragma: no cover - exercised without the extra
                        raise ImportError('Dense re-ranking needs the optional extra: pip install "locus-rag[dense]"') from e
                    if self.threads:
                        import torch
                        torch.set_num_threads(int(self.threads))
                    self._st = SentenceTransformer(self.model, device=self.device)
        return self._st

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.encoder is not None:
            return [_normalize(v) for v in self.encoder(list(texts))]
        vecs = self._load().encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                   show_progress_bar=False, convert_to_numpy=True)
        return [list(map(float, v)) for v in vecs]

    # -- document text --------------------------------------------------
    def _doc_texts(self, docs: list[str], pool: list) -> dict[str, str]:
        out: dict[str, str] = {}
        corpus = getattr(self._engine, "corpus", None)
        if corpus is not None:
            try:
                secs = corpus.doc_sections(docs)
                for d in docs:
                    if secs.get(d):
                        out[d] = "\n\n".join(s.content for s in secs[d])
            except Exception:  # pragma: no cover - older stores without sections
                pass
            for d in docs:
                if d not in out:
                    try:
                        chs = corpus.get_chunks_for_doc(d)
                    except Exception:  # pragma: no cover
                        chs = []
                    if chs:
                        out[d] = _join_chunks(chs)
        for d in docs:  # unbound / missing: the pool's own chunks
            if d not in out:
                out[d] = "\n\n".join(c.content for c in pool if c.doc_path == d)
        return out

    def doc_scores(self, query: str, docs: list[str], pool: list) -> dict[str, float]:
        """Max-cosine dense score of each document in *docs*."""
        texts = self._doc_texts(docs, pool)
        keys = {d: (d, hashlib.sha256(texts[d].encode("utf-8", "replace")).hexdigest()) for d in docs}
        vecs_of: dict[str, list] = {}
        for d in docs:  # cache hits (refreshed in the LRU)
            if keys[d] in self._cache:
                self._cache.move_to_end(keys[d])
                vecs_of[d] = self._cache[keys[d]]
        todo = [d for d in docs if d not in vecs_of]
        wins, owner = [], []
        for d in todo:
            for w in text_windows(texts[d], self.window_words, self.stride_words):
                wins.append(w)
                owner.append(d)
        vecs = self.encode([query] + wins)
        qv, wv = vecs[0], vecs[1:]
        for d in todo:
            vecs_of[d] = []
        for d, v in zip(owner, wv):
            vecs_of[d].append(v)
        for d in todo:
            self._cache[keys[d]] = vecs_of[d]
        while len(self._cache) > self.cache_docs:
            self._cache.popitem(last=False)
        return {d: max((_dot(qv, v) for v in vecs_of[d]), default=-1.0) for d in docs}

    def rerank(self, query: str, chunks: list, limit: int | None = None) -> list:
        """Fuse the pool's BM25 document order with the dense order (RRF); see module docstring."""
        chunks = list(chunks)
        if not chunks or not self.applies(query):
            return chunks[:limit] if limit is not None else chunks
        bm = list(dict.fromkeys(c.doc_path for c in chunks))
        sc = self.doc_scores(query, bm, chunks)
        pos = {d: i for i, d in enumerate(bm)}
        dn = sorted(bm, key=lambda d: (-sc[d], pos[d]))
        order, fused = rrf_doc_order([bm, dn], k=self.rrf_k)
        return reorder_by_docs(chunks, order, fused, "+dense", limit)


def _join_chunks(chs: list) -> str:
    """Reconstruct document text from position-ordered chunks, dropping word overlap between neighbours."""
    words: list[str] = []
    end = 0
    for c in chs:
        w = c.content.split()
        start = int(getattr(c, "start_word", 0) or 0)
        skip = max(0, end - start) if start or end else 0
        words.extend(w[skip:])
        end = max(end, start + len(w))
    return " ".join(words)
