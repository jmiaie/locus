"""
Optional cross-encoder reranking (``pip install "locus[rerank]"``).

A cross-encoder reads the query and each candidate chunk together and
outputs a relevance score.
- More accurate than BM25 alone.
- Far slower, so it only reorders the top *depth* BM25 candidates.
- Runs locally on CPU.
- Only ``sentence-transformers`` (and so PyTorch) is imported, and only when
  the reranker is first used. The Locus core never imports this module.

    from locus.rerank import CrossEncoderReranker
    engine = LocusEngine(".locus", reranker=CrossEncoderReranker(depth=50))
    engine.retrieve("why is the L2 slab thicker?")          # reranked
    engine.retrieve("why is the L2 slab thicker?", rerank=False)

Defaults:
- Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (22M parameters).
- Scoring: inputs are truncated to ``max_length`` tokens and scored in
  batches, with ties broken by the BM25 order.
- Determinism: results are deterministic for a given model, library version
  and thread count.

Model weights:
- They are downloaded from the Hugging Face Hub on first use. Set
  ``HF_HUB_OFFLINE=1`` once they are cached.
- Telemetry is disabled (``HF_HUB_DISABLE_TELEMETRY=1``) unless you set it
  yourself.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, replace

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class CrossEncoderReranker:
    model: str = DEFAULT_MODEL
    depth: int = 50
    batch_size: int = 32
    max_length: int = 256
    device: str = "cpu"
    threads: int | None = None
    _ce: object = field(default=None, init=False, repr=False)
    _lock: object = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        if self.depth < 1:
            raise ValueError("depth must be >= 1")

    def key(self) -> str:
        return f"{self.model}|{self.depth}|{self.max_length}"

    def _load(self):
        if self._ce is None:
            with self._lock:
                if self._ce is None:
                    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
                    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                    try:
                        from sentence_transformers import CrossEncoder
                    except ImportError as e:  # pragma: no cover - exercised without the extra
                        raise ImportError(
                            'Cross-encoder reranking needs the optional extra: pip install "locus[rerank]"'
                        ) from e
                    if self.threads:
                        import torch
                        torch.set_num_threads(int(self.threads))
                    self._ce = CrossEncoder(self.model, device=self.device, max_length=self.max_length)
        return self._ce

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        ce = self._load()
        scores = ce.predict([(query, t) for t in texts], batch_size=self.batch_size,
                            show_progress_bar=False, convert_to_numpy=True)
        return [float(s) for s in scores]

    def rerank(self, query: str, chunks: list, limit: int | None = None) -> list:
        """Reorder the first *depth* chunks by cross-encoder score; the rest keep their order."""
        head, tail = list(chunks[: self.depth]), list(chunks[self.depth:])
        scores = self.score(query, [c.content for c in head])
        order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
        out = [replace(head[i], score=scores[i], provenance=head[i].provenance + "+ce") for i in order] + tail
        return out[:limit] if limit is not None else out
