"""
LocusEngine — main orchestrator for Locus vectorless RAG.

Retrieval is two-stage, no embeddings required:

  Stage 1 — BM25 (SQLite FTS5 inverted index; pure-Python fallback).
  Stage 2 — optional re-rank of the BM25 candidate pool with five signals:
      KG             — entity expansion via the temporal knowledge graph
      LinkWalk       — wikilink graph traversal from the top BM25 hits
      Structural     — frontmatter date / tag / type matching
      Recency        — frontmatter date (newest first)
      LinkPopularity — inbound wikilink count
    fused with the BM25 ranking by weighted Reciprocal Rank Fusion.

``mode="bm25"`` (the default) returns stage 1 only.  On the public BEIR
SciFact and NFCorpus test sets the stage-2 re-rank did not significantly
beat BM25 (see ``benchmarks/README.md``), so it is opt-in:
``retrieve(..., mode="hybrid")`` or ``LocusEngine(retrieval_mode="hybrid")``
or ``LOCUS_RETRIEVAL_MODE=hybrid``.  Passing ``as_of`` without a mode
selects ``hybrid``, because the temporal filter only acts on KG signals.

Retrieval is deterministic (ties broken by chunk id / doc path) and
read-only unless ``track=True`` is passed, which records hits in the
hot-context bulletin and the token budget.
"""

import logging
import os
import time
from pathlib import Path
from typing import Iterator

from .memory.corpus import Corpus
from .memory.knowledge_graph import TemporalKG
from .memory.entity_resolver import EntityResolver
from .retrieval.bm25 import BM25Retriever, ScoredChunk
from .retrieval.kg_retrieval import KGRetriever, kg_source_scores
from .retrieval.link_walker import LinkWalker
from .retrieval.structural import StructuralRetriever, score_frontmatter
from .retrieval.recency import RecencyRetriever
from .retrieval.link_popularity import LinkPopularityRetriever
from .retrieval.reranker import LocusReranker, RerankerWeights
from .context.packer import ContextPacker, PackedContext
from .retrieval.fusion import rrf_fuse
from .retrieval.classifier import classify_query, INTENT_WEIGHTS, QueryIntent
from .retrieval.streaming import StreamingConfig, StreamingRetriever
from .context.bulletin import ContextBulletin
from .context.budget import ContextBudget
from .hooks import LocusHooks
from .reasoning import LocusReasoner
from .query_expander import QueryExpander
from .similarity import DocSimilarity
from .diff import CorpusDiff
from .annotations import AnnotationStore
from .feedback import RelevanceFeedback

logger = logging.getLogger(__name__)

__version__ = "1.0.0"

# Fixed weights appended after intent weights: [structural, recency]
_STRUCTURAL_WEIGHT  = 1.0
_RECENCY_WEIGHT     = 0.3
_LINK_POP_WEIGHT    = 0.2

_CACHE_MAX_SIZE     = 256

#: Retrieval modes.  "bm25" = first stage only; "hybrid" = BM25 + re-rank.
RETRIEVAL_MODES = ("bm25", "hybrid")
DEFAULT_RETRIEVAL_MODE = "bm25"
#: Stage-2 signals, in fusion order.
RERANK_SIGNALS = ("kg", "links", "structural", "recency", "link_pop")
#: BM25 candidates re-ranked in hybrid mode (at least ``limit``).
#: Selected on BEIR dev splits (SciFact train, NFCorpus dev) from {40, 100, 200}.
RERANK_DEPTH = 100
#: Multiplier on all stage-2 signal weights relative to BM25.  Selected on the
#: same dev splits from {0.1, 0.25, 0.5, 1.0}; larger values hurt nDCG there.
#: Corpora rich in wikilinks/frontmatter may prefer a larger value.
RERANK_ALPHA = 0.1


def _default_mode() -> str:
    env = os.environ.get("LOCUS_RETRIEVAL_MODE", "").strip().lower()
    return env if env in RETRIEVAL_MODES else DEFAULT_RETRIEVAL_MODE


class LocusEngine:
    """
    Vectorless RAG engine.

    Usage:
        engine = LocusEngine(store_path=".locus")
        engine.index("./my-docs")
        chunks = engine.retrieve("how does the auth system work?")
        context = engine.format_context(chunks)
    """

    def __init__(
        self,
        store_path: str | Path = ".locus",
        retrieval_mode: str | None = None,
        index_backend: str | None = None,
        analyzer: str | None = None,
        rerank_depth: int = RERANK_DEPTH,
        rerank_alpha: float = RERANK_ALPHA,
    ):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        mode = retrieval_mode or _default_mode()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {mode!r}")
        self.retrieval_mode = mode
        self.rerank_depth = int(rerank_depth)
        self.rerank_alpha = float(rerank_alpha)

        # Entity resolver must be initialised before KG (transparent alias lookups)
        self.resolver = EntityResolver(self.store_path / "resolver.sqlite3")

        self.corpus = Corpus(
            self.store_path / "corpus", section_aware=True, backend=index_backend, analyzer=analyzer,
        )
        self.kg = TemporalKG(str(self.store_path / "kg.sqlite3"), resolver=self.resolver)
        self.bulletin = ContextBulletin(
            archive_path=self.store_path / "archive",
            db_path=self.store_path / "bulletin.sqlite3",
        )
        self.budget = ContextBudget()

        # Retrieval signals
        self._bm25 = BM25Retriever(self.corpus)
        self._kg_ret = KGRetriever(self.kg, self.corpus)
        self._walker = LinkWalker(self.corpus)
        self._structural = StructuralRetriever(self.corpus)
        self._recency = RecencyRetriever(self.corpus)
        self._link_pop = LinkPopularityRetriever(self.corpus)
        self._reranker = LocusReranker(self.corpus, kg=self.kg)
        self._packer = ContextPacker()
        self._hooks = LocusHooks()
        self._reasoner = LocusReasoner(self.kg)
        self._expander = QueryExpander(self.kg, self.resolver)
        self._similarity = DocSimilarity(self.corpus)
        self._diff = CorpusDiff(self.corpus)
        self.annotations = AnnotationStore(self.store_path / "annotations.sqlite3")
        self.feedback = RelevanceFeedback(self.store_path / "feedback.sqlite3")
        self._reranker.feedback = self.feedback

        # Query result cache — invalidated on corpus changes
        self._query_cache: dict[str, list[ScoredChunk]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def set_hooks(self, hooks: "LocusHooks") -> None:
        """Replace the engine's hook registry."""
        self._hooks = hooks

    def _fire(self, event: str, **data) -> None:
        self._hooks.fire(event, self, **data)

    def index(self, path: str | Path, pattern: str = "**/*.md") -> dict:
        """
        Index a file or directory (incremental).

        Unchanged files are skipped (checksum dedup).  For new or changed
        files the chunks, the FTS5 postings and the KG triples extracted from
        that file are replaced in place; facts added with ``add_fact`` are
        never touched.  Returns counts of files, chunks added and KG triples
        extracted in this call.
        """
        path = Path(path)
        self._fire("pre_index", path=str(path), pattern=pattern)
        if path.is_file():
            chunks = self.corpus.add_file(path)
            triples = self.kg.replace_from_files([path]) if chunks else 0
            self._invalidate_cache()
            result = {"files": 1, "chunks": chunks, "triples": triples}
            self._fire("post_index", result=result)
            return result

        chunks, changed = self.corpus.add_directory_changed(path, pattern=pattern)
        triples = self.kg.replace_from_files(changed, base_path=path) if changed else 0
        self._invalidate_cache()
        result = {
            "files": self.corpus.doc_count(),
            "chunks": chunks,
            "triples": triples,
        }
        self._fire("post_index", result=result)
        return result

    def forget(self, doc_path: str) -> dict:
        """Remove a document's chunks, postings and its auto-extracted KG triples."""
        self._fire("pre_forget", doc_path=doc_path)
        removed = self.corpus.remove_file(doc_path)
        triples = self.kg.remove_extracted(doc_path)
        self._invalidate_cache()
        result = {"removed_chunks": removed, "removed_triples": triples, "doc": doc_path}
        self._fire("post_forget", result=result)
        return result

    def sync(self, path: str | Path, pattern: str = "**/*.md") -> dict:
        """Drop every indexed document and re-index *path* from scratch."""
        for doc in self.corpus.list_docs():
            self.corpus.remove_file(doc)
            self.kg.remove_extracted(doc)
        self._invalidate_cache()
        return self.index(path, pattern=pattern)

    def reindex(self, backend: str | None = None, analyzer: str | None = None) -> dict:
        """Rebuild the lexical index from stored chunks (no file reads).

        Clean re-index / migration path, e.g. ``reindex(backend="fts5")``
        after upgrading a store created by Locus <= 1.0.0.
        """
        result = self.corpus.rebuild_index(backend=backend, analyzer=analyzer)
        self._invalidate_cache()
        return result

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        as_of: str = None,
        use_links: bool = True,
        intent: QueryIntent = None,
        use_cache: bool = True,
        mode: str | None = None,
        track: bool = False,
        signals: tuple[str, ...] | list[str] | None = None,
    ) -> list[ScoredChunk]:
        """
        Retrieve the top *limit* chunks for *query*.

        mode:
          - ``"bm25"``   — BM25 only (default; see module docstring).
          - ``"hybrid"`` — BM25 candidate pool re-ranked with KG, link walk,
            structural, recency and link-popularity signals via weighted RRF.
          - ``None``     — the engine default, or ``"hybrid"`` when *as_of*
            is given.

        intent (hybrid only) is auto-classified unless overridden:
          - KG-first:   entity/factual queries (e.g. "who is Alice?")
          - BM25-first: procedural/narrative (e.g. "how does auth work?")
          - Balanced:   default

        track: when True, record the returned chunks in the hot-context
        bulletin and the token budget (the only writes retrieval performs).

        signals (hybrid only): subset of ``RERANK_SIGNALS`` to use; all by default.

        Results are cached in memory by all arguments that affect ranking and
        invalidated on index/forget/sync.  Ordering is deterministic.
        """
        if mode is None:
            mode = "hybrid" if as_of else self.retrieval_mode
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}, got {mode!r}")
        if mode == "hybrid" and intent is None:
            intent = classify_query(query)
        sig = tuple(RERANK_SIGNALS if signals is None else [s for s in RERANK_SIGNALS if s in signals])
        if not use_links:
            sig = tuple(s for s in sig if s != "links")

        key = None
        fused: list[ScoredChunk] | None = None
        if use_cache:
            key = self._cache_key(
                query, limit, as_of, use_links, intent.value if intent else "-", mode, sig,
            )
            if key in self._query_cache:
                self._cache_hits += 1
                fused = self._query_cache[key]
            else:
                self._cache_misses += 1

        if fused is None:
            self._fire("pre_retrieve", query=query, limit=limit)
            if mode == "bm25":
                fused = self._bm25.search(query, limit=limit)
            else:
                fused = self._hybrid(query, limit=limit, as_of=as_of, intent=intent, signals=sig)
            if use_cache:
                if len(self._query_cache) >= _CACHE_MAX_SIZE:
                    oldest = next(iter(self._query_cache))
                    del self._query_cache[oldest]
                self._query_cache[key] = fused
            self._fire("post_retrieve", query=query, result_count=len(fused))

        if track:
            self._track(fused, label=mode)
        return list(fused)

    def _track(self, chunks: list[ScoredChunk], label: str = "") -> None:
        """Record retrieved chunks in the bulletin and token budget (writes)."""
        total_tokens = 0
        for chunk in chunks:
            self.bulletin.record_hit(
                chunk.chunk_id,
                content=chunk.content,
                doc_path=chunk.doc_path,
                base_score=chunk.score,
                provenance=chunk.provenance,
            )
            total_tokens += self.budget.estimate_tokens(chunk.content)
        if total_tokens:
            check = self.budget.record(total_tokens)
            if check.status.value in ("critical", "warning", "trend"):
                logger.warning("Budget [%s]: %s", label, check.message)

    def _hybrid(
        self,
        query: str,
        limit: int,
        as_of: str | None,
        intent: QueryIntent,
        signals: tuple[str, ...] = RERANK_SIGNALS,
    ) -> list[ScoredChunk]:
        """BM25 candidate pool re-ranked by the stage-2 signals (read-only)."""
        pool = self._bm25.search(query, limit=max(limit, self.rerank_depth))
        if not pool:
            return []
        # Best-ranked pool chunk per document: doc-level signals vote for it.
        doc_chunk: dict[str, ScoredChunk] = {}
        for c in pool:
            doc_chunk.setdefault(c.doc_path, c)

        def as_signal(doc_scores: list[tuple[str, float]], provenance: str) -> list[ScoredChunk]:
            out = []
            for doc, score in doc_scores:
                c = doc_chunk[doc]
                out.append(ScoredChunk(c.chunk_id, c.doc_path, score, c.content, provenance, c.entities))
            return out

        lists: list[list[ScoredChunk]] = [pool]
        iw = INTENT_WEIGHTS[intent]  # [bm25, kg, link]
        weights: list[float] = [iw[0]]
        a = self.rerank_alpha
        meta: dict | None = None

        def pool_meta() -> dict:
            nonlocal meta
            if meta is None:
                meta = self.corpus.get_chunks_batch([c.chunk_id for c in doc_chunk.values()])
            return meta

        if "kg" in signals:
            ks = kg_source_scores(self.kg, query, as_of=as_of)
            ranked = sorted(((d, ks[d]) for d in doc_chunk if d in ks), key=lambda x: (-x[1], x[0]))
            lists.append(as_signal(ranked, "kg"))
            weights.append(a * iw[1])
        if "links" in signals:
            walked = self._walker.walk(pool[:3], depth=2, limit=len(doc_chunk))
            ranked = []
            seen: set[str] = set()
            for w in walked:
                if w.doc_path in doc_chunk and w.doc_path not in seen:
                    seen.add(w.doc_path)
                    ranked.append((w.doc_path, w.score))
            hops = {w.doc_path: w.provenance for w in walked}
            sig = as_signal(ranked, "link")
            for c in sig:
                c.provenance = hops.get(c.doc_path, "link:hop1")
            lists.append(sig)
            weights.append(a * iw[2])
        if "structural" in signals:
            m = pool_meta()
            ranked = []
            for doc, c in doc_chunk.items():
                ch = m.get(c.chunk_id)
                sc = score_frontmatter(query, ch.metadata.get("frontmatter", {}) if ch else {})
                if sc > 0:
                    ranked.append((doc, sc))
            ranked.sort(key=lambda x: (-x[1], x[0]))
            lists.append(as_signal(ranked, "structural"))
            weights.append(a * _STRUCTURAL_WEIGHT)
        if "recency" in signals:
            m = pool_meta()
            dated = []
            for doc, c in doc_chunk.items():
                ch = m.get(c.chunk_id)
                d = str((ch.metadata.get("frontmatter", {}) if ch else {}).get("date", "") or "")[:10]
                if d:
                    dated.append((doc, d))
            dated.sort(key=lambda x: x[0])
            dated.sort(key=lambda x: x[1], reverse=True)  # newest first; stable -> ties by doc_path
            n = len(dated)
            lists.append(as_signal([(doc, 1.0 - i / max(n, 1)) for i, (doc, _) in enumerate(dated)], "recency"))
            weights.append(a * _RECENCY_WEIGHT)
        if "link_pop" in signals:
            pop = self._link_pop.popularity_map()
            mx = max(pop.values()) if pop else 1
            ranked = sorted(
                ((d, pop[d] / mx) for d in doc_chunk if pop.get(d, 0) > 0),
                key=lambda x: (-x[1], x[0]),
            )
            lists.append(as_signal(ranked, "link_pop"))
            weights.append(a * _LINK_POP_WEIGHT)

        return rrf_fuse(lists, weights=weights, limit=limit)

    def stream_retrieve(
        self,
        query: str,
        limit: int = 5,
        as_of: str = None,
        use_links: bool = True,
        config: StreamingConfig | None = None,
        track: bool = False,
    ) -> Iterator[ScoredChunk]:
        """Yield chunks as each retrieval signal finishes.

        Same six signals as :meth:`retrieve` (BM25, KG, link walk, structural,
        recency, link popularity), run one at a time. Each new ``chunk_id`` is
        yielded when its signal returns, until ``limit`` unique chunks have
        been produced.

        This is not RRF fusion. Order and membership can differ from
        ``retrieve()``. :class:`StreamingRetriever` adaptive dropout can skip
        signals that would start after the time budget is inside
        ``signal_timeout_ms``. With ``track=True`` bulletin hits and the token
        budget are recorded for chunks actually yielded (otherwise streaming is
        read-only). Results are not stored in the retrieve cache.
        """
        retriever = StreamingRetriever(config)
        retriever._start_time = time.perf_counter()
        seen: set[str] = set()
        yielded = 0
        total_tokens = 0
        bm25_hits: list[ScoredChunk] = []

        def _signals():
            yield "bm25", lambda: self._bm25.search(query, limit=limit * 2)
            yield "kg", lambda: self._kg_ret.search(query, limit=limit * 2, as_of=as_of)
            if use_links:
                yield "link", lambda: (
                    self._walker.walk(bm25_hits[:3], depth=2, limit=limit) if bm25_hits else []
                )
            yield "structural", lambda: self._structural.search(query, limit=limit * 2)
            yield "recency", lambda: self._recency.search(limit=limit * 2)
            yield "link_pop", lambda: self._link_pop.search(limit=limit * 2)

        try:
            self._fire("pre_retrieve", query=query, limit=limit, stream=True)
            for name, run in _signals():
                if yielded >= limit:
                    break
                if not retriever.should_run_signal(name):
                    break
                started = time.perf_counter()
                hits = run() or []
                elapsed_ms = (time.perf_counter() - started) * 1000
                retriever.record_signal(name, hits, elapsed_ms)
                if name == "bm25":
                    bm25_hits = list(hits)
                for chunk in hits:
                    if yielded >= limit:
                        break
                    if chunk.chunk_id in seen:
                        continue
                    seen.add(chunk.chunk_id)
                    retriever.track_result(chunk, name)
                    if track:
                        self.bulletin.record_hit(
                            chunk.chunk_id,
                            content=chunk.content,
                            doc_path=chunk.doc_path,
                            base_score=chunk.score,
                            provenance=chunk.provenance,
                        )
                        total_tokens += self.budget.estimate_tokens(chunk.content)
                    yielded += 1
                    yield chunk
        finally:
            if total_tokens:
                check = self.budget.record(total_tokens)
                if check.status.value in ("critical", "warning", "trend"):
                    logger.warning("Budget [stream]: %s", check.message)
            self._fire("post_retrieve", query=query, result_count=yielded, stream=True)

    # ------------------------------------------------------------------
    # Reranking, context packing & confidence
    # ------------------------------------------------------------------

    def rerank(
        self,
        chunks: list[ScoredChunk],
        query: str,
        weights: RerankerWeights | None = None,
    ) -> list[ScoredChunk]:
        """Apply heuristic re-ranking (title, entity density, freshness) after RRF."""
        return self._reranker.rerank(chunks, query, weights=weights)

    def pack_context(
        self,
        chunks: list[ScoredChunk],
        token_budget: int = 4000,
    ) -> PackedContext:
        """Pack chunks into a token budget, grouped by document."""
        return self._packer.pack(chunks, token_budget=token_budget)

    def assess_confidence(self, chunks: list[ScoredChunk]) -> dict:
        """
        Assess how confident Locus is in these retrieval results.
        Returns level ('ok' | 'low' | 'empty') and top RRF score.
        """
        if not chunks:
            return {"level": "empty", "top_score": 0.0,
                    "note": "No results — try indexing more documents"}
        top = chunks[0].score
        if top < 0.005:
            return {"level": "low", "top_score": round(top, 6),
                    "note": "Weak signal — results may not be relevant"}
        return {"level": "ok", "top_score": round(top, 6)}

    def prepare_context(
        self,
        query: str,
        limit: int = 5,
        token_budget: int = 4000,
        rerank: bool = True,
        as_of: str = None,
        mode: str | None = None,
        track: bool = False,
    ) -> dict:
        """
        All-in-one context preparation for LLM agents:
          retrieve → rerank → pack → assess_confidence → KG context

        Returns everything needed to answer the query in a single call.
        """
        from .retrieval.kg_retrieval import extract_query_entities

        chunks = self.retrieve(query, limit=limit, as_of=as_of, mode=mode, track=track)
        if rerank and chunks:
            chunks = self.rerank(chunks, query)

        packed = self.pack_context(chunks, token_budget=token_budget)
        confidence = self.assess_confidence(chunks)

        # KG facts for entities detected in the query
        entities = extract_query_entities(query)
        kg_context: dict[str, list[str]] = {}
        for entity in entities[:5]:
            triples = self.kg.query_entity(entity, as_of=as_of)
            if triples:
                kg_context[entity] = [
                    f"{t.subject} --{t.predicate}--> {t.object}"
                    for t in triples[:10]
                ]

        return {
            "query": query,
            "confidence": confidence,
            "packed_context": packed.text,
            "tokens_used": packed.tokens_used,
            "token_budget": packed.token_budget,
            "chunks_included": packed.chunks_included,
            "truncated": packed.truncated,
            "sources": packed.sources,
            "kg_context": kg_context,
        }

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_key(self, query: str, limit: int, as_of: str | None,
                   use_links: bool, intent_val: str, mode: str = "",
                   signals: tuple = ()) -> str:
        import hashlib
        raw = f"{query}|{limit}|{as_of}|{use_links}|{intent_val}|{mode}|{','.join(signals)}|{self.rerank_depth}|{self.rerank_alpha}"
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def _invalidate_cache(self) -> None:
        self._query_cache.clear()

    def cache_stats(self) -> dict:
        total = self._cache_hits + self._cache_misses
        return {
            "size": len(self._query_cache),
            "max_size": _CACHE_MAX_SIZE,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(self._cache_hits / total, 3) if total else 0.0,
        }

    def clear_cache(self) -> dict:
        size = len(self._query_cache)
        self._invalidate_cache()
        self._cache_hits = 0
        self._cache_misses = 0
        return {"cleared": size}

    def format_context(
        self, chunks: list[ScoredChunk], include_hot: bool = True
    ) -> str:
        parts: list[str] = []
        if include_hot:
            hot = self.bulletin.inject(token_limit=800)
            if hot:
                parts.append(f"## Locus Hot Context\n{hot}")
        if chunks:
            parts.append("## Retrieved Context")
            for i, chunk in enumerate(chunks, 1):
                section = chunk.content.splitlines()[0][:60] if chunk.content else ""
                parts.append(
                    f"### [{i}] {chunk.doc_path}  (via {chunk.provenance})\n{chunk.content}"
                )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Knowledge Graph
    # ------------------------------------------------------------------

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        valid_from: str = None,
        valid_to: str = None,
        source: str = None,
    ) -> dict:
        self._fire("pre_add_fact", subject=subject, predicate=predicate, object_=object_)
        self.kg.add_triple(
            subject, predicate, object_,
            valid_from=valid_from, valid_to=valid_to, source=source,
        )
        result = {"added": f"{subject} --{predicate}--> {object_}"}
        self._fire("post_add_fact", result=result)
        return result

    def query_entity(self, entity: str, as_of: str = None) -> dict:
        triples = self.kg.query_entity(entity, as_of=as_of)
        return {
            "entity": entity,
            "facts": [
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "valid_from": t.valid_from,
                    "valid_to": t.valid_to,
                    "source": t.source,
                }
                for t in triples
            ],
        }

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    def add_alias(self, alias: str, canonical: str) -> dict:
        """Register alias → canonical so both resolve to the same KG entity."""
        self.resolver.add_alias(alias, canonical)
        return {"alias": alias, "canonical": canonical}

    def suggest_aliases(self, threshold: float = 0.75) -> dict:
        """Suggest entity name pairs that may refer to the same entity."""
        entities = self.kg.all_entities()
        suggestions = self.resolver.suggest_aliases(entities, threshold=threshold)
        return {
            "entity_count": len(entities),
            "suggestions": suggestions,
            "threshold": threshold,
        }

    def list_aliases(self) -> dict:
        return {"aliases": self.resolver.list_aliases()}

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    def find_contradictions(self, entity: str = None) -> dict:
        """
        Find KG triples that contradict each other:
        same subject+predicate, different objects, overlapping validity windows.
        """
        contradictions = self.kg.find_contradictions(entity)
        return {
            "entity": entity,
            "contradiction_count": len(contradictions),
            "contradictions": contradictions,
        }

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain(self, chunk_id: str, query: str = None) -> dict:
        """
        Explain why a chunk was (or would be) retrieved for a query.

        For each active retrieval signal (BM25, KG, Structural) the method
        checks whether this chunk would have been returned and reports the
        specific terms, entities, or metadata that caused it.  A plain-
        English narrative is assembled and returned alongside the raw data.

        Works with any chunk_id regardless of whether it came from a prior
        retrieve() call.
        """
        from .memory.corpus import tokenize
        from .retrieval.kg_retrieval import extract_query_entities
        from .retrieval.structural import (
            _extract_date_range, _extract_tags, _extract_type,
            _date_in_range, _tag_overlap,
        )

        chunk = self.corpus.get_chunk(chunk_id)
        if not chunk:
            return {"error": f"Chunk not found: {chunk_id}"}

        fm = chunk.metadata.get("frontmatter", {})
        result: dict = {
            "chunk_id": chunk_id,
            "doc_path": chunk.doc_path,
            "section": chunk.metadata.get("section", ""),
            "content_preview": chunk.content[:300],
            "entities_in_chunk": chunk.metadata.get("entities", []),
            "links_in_chunk": chunk.metadata.get("links", []),
            "frontmatter": fm,
        }

        signals: list[str] = []

        if query:
            # BM25 — which query terms appear in this chunk?
            terms = self.corpus.analyze(query)
            matched = [t for t in terms if chunk_id in self.corpus._posting_list(t)]
            if matched:
                result["bm25_matched_terms"] = matched
                signals.append(f"BM25: terms [{', '.join(matched)}] appear in this chunk")

            # KG — which query entities link to this document?
            query_entities = extract_query_entities(query)
            kg_hits: list[str] = []
            for entity in query_entities[:8]:
                if chunk.doc_path in self.kg.sources_for_entity(entity):
                    kg_hits.append(entity)
            if kg_hits:
                result["kg_matched_entities"] = kg_hits
                signals.append(
                    f"KG: entities [{', '.join(kg_hits)}] link to this document"
                )

            # Structural — does frontmatter match date/tag/type signals?
            structural: list[str] = []
            dr = _extract_date_range(query)
            if dr and _date_in_range(fm.get("date", ""), *dr):
                structural.append(f"date {fm['date']} in range {dr[0]}–{dr[1]}")
            qt = _extract_tags(query)
            if qt and _tag_overlap(fm.get("tags", ""), qt) > 0:
                structural.append(f"tags matched: {qt}")
            qtype = _extract_type(query)
            if qtype:
                doc_type = fm.get("type", fm.get("category", "")).lower()
                if qtype in doc_type or doc_type in qtype:
                    structural.append(f"type '{doc_type}' matches query type '{qtype}'")
            if structural:
                result["structural_matches"] = structural
                signals.append(f"Structural: {'; '.join(structural)}")

        # KG context — facts about entities mentioned in this chunk
        kg_context: list[dict] = []
        for entity in chunk.metadata.get("entities", [])[:5]:
            triples = self.kg.query_entity(entity)
            if triples:
                t0 = triples[0]
                kg_context.append({
                    "entity": entity,
                    "fact_count": len(triples),
                    "sample_fact": f"{t0.subject} --{t0.predicate}--> {t0.object}",
                })
        result["kg_context"] = kg_context

        # Plain-English narrative
        head = f"Chunk from '{chunk.doc_path}'"
        if chunk.metadata.get("section"):
            head += f" (section: '{chunk.metadata['section']}')"
        narrative_parts = [head]
        if signals:
            narrative_parts.append("Retrieved because: " + "; ".join(signals))
        else:
            narrative_parts.append(
                "No query provided — cannot explain retrieval signals"
                if not query
                else "No retrieval signals matched for this query"
            )
        if kg_context:
            names = [k["entity"] for k in kg_context[:3]]
            narrative_parts.append(
                f"Contains {len(kg_context)} entity/entities with KG facts: {', '.join(names)}"
            )
        result["narrative"] = ". ".join(narrative_parts) + "."
        return result

    # ------------------------------------------------------------------
    # KG traversal & export
    # ------------------------------------------------------------------

    def kg_traverse(
        self,
        start: str,
        max_depth: int = 2,
        predicate_filter: list[str] | None = None,
        direction: str = "both",
    ) -> dict:
        """BFS traversal from a starting entity. Returns entity→facts map."""
        result = self.kg.traverse(start, max_depth=max_depth,
                                   predicate_filter=predicate_filter, direction=direction)
        return {
            entity: [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object,
                 "valid_from": t.valid_from, "valid_to": t.valid_to, "source": t.source}
                for t in triples
            ]
            for entity, triples in result.items()
        }

    def kg_match(
        self,
        subject: str = "*",
        predicate: str = "*",
        obj: str = "*",
        as_of: str = None,
    ) -> dict:
        """Pattern match over the KG. '*' is a wildcard."""
        triples = self.kg.match(subject=subject, predicate=predicate, obj=obj, as_of=as_of)
        return {
            "pattern": {"subject": subject, "predicate": predicate, "object": obj},
            "count": len(triples),
            "triples": [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object,
                 "valid_from": t.valid_from, "valid_to": t.valid_to, "source": t.source}
                for t in triples
            ],
        }

    def doctor(self) -> dict:
        """Run health checks on this store. Returns structured report."""
        from .doctor import LocusDoctor
        return LocusDoctor(self).to_dict()

    def export_kg(self, path: str, fmt: str = None) -> dict:
        """Export the KG to a file. fmt: graphml | jsonl | dot (auto from extension)."""
        from .export import KGExporter
        count = KGExporter(self.kg).export(path, fmt=fmt)
        return {"path": path, "triples_exported": count, "format": fmt or "auto"}

    # ------------------------------------------------------------------
    # Reasoning & corpus inspection
    # ------------------------------------------------------------------

    def reason(self, question: str, max_depth: int = 3) -> dict:
        """Multi-hop KG reasoning over detected entities in *question*."""
        return self._reasoner.reason(question, max_depth=max_depth)

    def find_paths(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int = 3,
        predicate_filter: list[str] | None = None,
    ) -> list[dict]:
        """Find all KG paths between two entities (BFS up to *max_depth* hops)."""
        paths = self._reasoner.find_paths(
            entity_a, entity_b,
            max_depth=max_depth,
            predicate_filter=predicate_filter,
        )
        return [p.to_dict() for p in paths]

    def inspect_doc(self, doc_path: str, limit: int = 20) -> dict:
        """Return corpus statistics and top terms for a single document."""
        chunks = self.corpus.get_chunks_for_doc(doc_path)
        if not chunks:
            return {"error": f"Document not found: {doc_path}"}
        terms = self.corpus.inspect_doc_terms(doc_path, limit=limit)
        entities: list[str] = []
        links: list[str] = []
        for c in chunks:
            entities.extend(c.metadata.get("entities", []))
            links.extend(c.metadata.get("links", []))
        return {
            "doc_path": doc_path,
            "chunk_count": len(chunks),
            "word_count": sum(len(c.content.split()) for c in chunks),
            "top_terms": terms,
            "entities": list(dict.fromkeys(entities))[:20],
            "links": list(dict.fromkeys(links))[:20],
        }

    def top_terms(self, limit: int = 20) -> list[dict]:
        """Return the corpus-wide top terms by document frequency."""
        return self.corpus.top_terms(limit=limit)

    # ------------------------------------------------------------------
    # Document intelligence
    # ------------------------------------------------------------------

    def related_docs(self, doc_path: str, limit: int = 5) -> list[dict]:
        """Return the most similar documents to *doc_path* by TF-IDF cosine."""
        return self._similarity.similar_to(doc_path, limit=limit)

    def similarity_matrix(self) -> list[dict]:
        """Pairwise TF-IDF cosine similarities for all indexed documents."""
        return self._similarity.similarity_matrix()

    def diff(self, path: str, pattern: str = "**/*.md") -> dict:
        """Preview what would change if *path* were re-indexed now.

        Returns counts/lists of new, changed, deleted, and unchanged files
        without modifying the corpus.
        """
        return self._diff.diff(path, pattern=pattern)

    # ------------------------------------------------------------------
    # Annotations & feedback
    # ------------------------------------------------------------------

    def annotate(self, chunk_id: str, label: str, note: str | None = None) -> dict:
        """Attach a label (and optional note) to a chunk."""
        self.annotations.annotate(chunk_id, label, note)
        return {"chunk_id": chunk_id, "label": label, "note": note}

    def get_annotations(self, chunk_id: str) -> dict:
        """Return all annotations for a chunk."""
        return {"chunk_id": chunk_id, "annotations": self.annotations.get(chunk_id)}

    def chunks_with_label(self, label: str) -> dict:
        """Return all chunk IDs carrying *label*."""
        return {"label": label, "chunk_ids": self.annotations.chunks_with_label(label)}

    def mark_relevant(self, chunk_id: str, query: str) -> dict:
        """Mark *chunk_id* as relevant for *query*; boosts future reranking."""
        self.feedback.mark(chunk_id, query, relevant=True)
        return {"chunk_id": chunk_id, "query": query, "signal": "relevant"}

    def mark_irrelevant(self, chunk_id: str, query: str) -> dict:
        """Mark *chunk_id* as irrelevant for *query*; penalises future reranking."""
        self.feedback.mark(chunk_id, query, relevant=False)
        return {"chunk_id": chunk_id, "query": query, "signal": "irrelevant"}

    # ------------------------------------------------------------------
    # Query intelligence & snapshot
    # ------------------------------------------------------------------

    def expand_query(self, query: str, max_expansions: int = 5) -> dict:
        """Expand *query* using KG alias resolution and first-hop neighbours.

        Returns the enriched query string plus metadata about what was added.
        """
        return self._expander.expand(query, max_expansions=max_expansions).to_dict()

    def multi_retrieve(
        self,
        queries: list[str],
        limit: int = 5,
        as_of: str = None,
        use_links: bool = True,
        mode: str | None = None,
        track: bool = False,
    ) -> list[ScoredChunk]:
        """Run each query independently and fuse all result lists via RRF.

        Useful for query reformulation, typo variants, or multi-intent queries.
        """
        if not queries:
            return []
        all_lists = [
            self.retrieve(q, limit=limit * 2, as_of=as_of, use_links=use_links, mode=mode)
            for q in queries
        ]
        fused = rrf_fuse(all_lists, limit=limit)
        if track:
            self._track(fused, label="multi")
        return fused

    def timeline(self, entity: str, as_of: str = None) -> dict:
        """Return all KG facts for *entity* sorted chronologically by valid_from.

        Facts without a valid_from date are listed last.
        """
        triples = self.kg.query_entity(entity, as_of=as_of)
        sorted_triples = sorted(
            triples,
            key=lambda t: (t.valid_from or "9999-99-99", t.predicate),
        )
        return {
            "entity": entity,
            "event_count": len(sorted_triples),
            "timeline": [
                {
                    "date": t.valid_from,
                    "valid_to": t.valid_to,
                    "predicate": t.predicate,
                    "object": t.object,
                    "source": t.source,
                }
                for t in sorted_triples
            ],
        }

    def snapshot(self, output_path: str) -> dict:
        """Archive this store to a portable .tar.gz file."""
        from .snapshot import LocusSnapshot
        return LocusSnapshot.save(self, output_path)

    def session_start(self) -> dict:
        return {
            "corpus": self.corpus.stats(),
            "kg": self.kg.stats(),
            "bulletin": self.bulletin.stats(),
            "resolver": self.resolver.stats(),
            "hot_context": self.bulletin.inject(token_limit=800) or "(empty — run locus_index first)",
        }

    def wrap_up(self) -> dict:
        archived = self.bulletin.tick()
        return {
            "archived_entries": archived,
            "bulletin": self.bulletin.stats(),
            "budget": self.budget.stats(),
        }

    def status(self) -> dict:
        return {
            "version": __version__,
            "retrieval_mode": self.retrieval_mode,
            "corpus": self.corpus.stats(),
            "kg": self.kg.stats(),
            "bulletin": self.bulletin.stats(),
            "budget": self.budget.stats(),
            "resolver": self.resolver.stats(),
            "annotations": self.annotations.stats(),
            "feedback": self.feedback.stats(),
        }
