"""
LocusEngine — main orchestrator for Locus vectorless RAG.

Five-signal retrieval pipeline, no embeddings required:
  1. BM25        — probabilistic keyword retrieval
  2. KG          — entity expansion via temporal knowledge graph
  3. LinkWalk    — wikilink/citation graph traversal from top BM25 hits
  4. Structural  — frontmatter date / tag / type matching (Phase 2)
  5. Recency     — exponential freshness prior (Phase 2)

Signals 1-3 are weighted by query intent (KG-first / BM25-first / balanced).
Signals 4-5 use fixed weights (structural=1.0, recency=0.3).
All five are fused via weighted Reciprocal Rank Fusion.

Phase 3 additions:
  - EntityResolver for transparent alias de-duplication in the KG
  - Prose triple extraction from natural language text
  - Contradiction detection across KG triples
  - alias management API (add_alias, suggest_aliases)
"""

import logging
from pathlib import Path

from .memory.corpus import Corpus
from .memory.knowledge_graph import TemporalKG
from .memory.entity_resolver import EntityResolver
from .retrieval.bm25 import BM25Retriever, ScoredChunk
from .retrieval.kg_retrieval import KGRetriever
from .retrieval.link_walker import LinkWalker
from .retrieval.structural import StructuralRetriever
from .retrieval.recency import RecencyRetriever
from .retrieval.link_popularity import LinkPopularityRetriever
from .retrieval.reranker import LocusReranker, RerankerWeights
from .context.packer import ContextPacker, PackedContext
from .retrieval.fusion import rrf_fuse
from .retrieval.classifier import classify_query, INTENT_WEIGHTS, QueryIntent
from .context.bulletin import ContextBulletin
from .context.budget import ContextBudget

logger = logging.getLogger(__name__)

__version__ = "0.7.0"

# Fixed weights appended after intent weights: [structural, recency]
_STRUCTURAL_WEIGHT  = 1.0
_RECENCY_WEIGHT     = 0.3
_LINK_POP_WEIGHT    = 0.2

_CACHE_MAX_SIZE     = 256


class LocusEngine:
    """
    Vectorless RAG engine.

    Usage:
        engine = LocusEngine(store_path=".locus")
        engine.index("./my-docs")
        chunks = engine.retrieve("how does the auth system work?")
        context = engine.format_context(chunks)
    """

    def __init__(self, store_path: str | Path = ".locus"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        # Phase 3: entity resolver (must come before KG)
        self.resolver = EntityResolver(self.store_path / "resolver.sqlite3")

        self.corpus = Corpus(self.store_path / "corpus", section_aware=True)
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
        self._reranker = LocusReranker(self.corpus, kg=self.kg)  # Phase 8
        self._packer = ContextPacker()                            # Phase 8

        # Query result cache — invalidated on corpus changes
        self._query_cache: dict[str, list[ScoredChunk]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, path: str | Path, pattern: str = "**/*.md") -> dict:
        """
        Index a file or directory.
        Skips unchanged files (checksum dedup). Extracts KG triples from
        wikilinks, tags, and prose sentences.
        Returns counts of files, chunks added, and KG triples extracted.
        """
        path = Path(path)
        if path.is_file():
            chunks = self.corpus.add_file(path)
            triples = self.kg.populate_from_file(path)
            return {"files": 1, "chunks": chunks, "triples": triples}

        chunks = self.corpus.add_directory(path, pattern=pattern)
        triples = 0
        for f in path.glob(pattern):
            if any(ex in f.parts for ex in {".locus", ".palace", ".git", ".obsidian"}):
                continue
            triples += self.kg.populate_from_file(f, base_path=path)
        self._invalidate_cache()
        return {
            "files": self.corpus.doc_count(),
            "chunks": chunks,
            "triples": triples,
        }

    def forget(self, doc_path: str) -> dict:
        removed = self.corpus.remove_file(doc_path)
        self._invalidate_cache()
        return {"removed_chunks": removed, "doc": doc_path}

    def sync(self, path: str | Path, pattern: str = "**/*.md") -> dict:
        for doc in self.corpus.list_docs():
            self.corpus.remove_file(doc)
        self._invalidate_cache()
        return self.index(path, pattern=pattern)

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
    ) -> list[ScoredChunk]:
        """
        Six-signal retrieval: BM25 + KG + link walk + structural + recency + link popularity.

        Query intent auto-classified unless overridden:
          - KG-first:   entity/factual queries (e.g. "who is Alice?")
          - BM25-first: procedural/narrative (e.g. "how does auth work?")
          - Balanced:   default

        Results are cached by (query, limit, as_of, use_links, intent) and
        invalidated automatically on index/forget/sync.
        """
        if intent is None:
            intent = classify_query(query)

        if use_cache:
            key = self._cache_key(query, limit, as_of, use_links, intent.value)
            if key in self._query_cache:
                self._cache_hits += 1
                return self._query_cache[key]
            self._cache_misses += 1

        bm25_hits = self._bm25.search(query, limit=limit * 2)
        kg_hits = self._kg_ret.search(query, limit=limit * 2, as_of=as_of)
        link_hits = self._walker.walk(bm25_hits[:3], depth=2, limit=limit) if use_links and bm25_hits else []
        structural_hits = self._structural.search(query, limit=limit * 2)
        recency_hits    = self._recency.search(limit=limit * 2)
        link_pop_hits   = self._link_pop.search(limit=limit * 2)

        intent_weights = INTENT_WEIGHTS[intent]  # [bm25, kg, link]
        all_weights = [*intent_weights, _STRUCTURAL_WEIGHT, _RECENCY_WEIGHT, _LINK_POP_WEIGHT]

        fused = rrf_fuse(
            [bm25_hits, kg_hits, link_hits, structural_hits, recency_hits, link_pop_hits],
            weights=all_weights,
            limit=limit,
        )

        total_tokens = 0
        for chunk in fused:
            self.bulletin.record_hit(
                chunk.chunk_id,
                content=chunk.content,
                doc_path=chunk.doc_path,
                base_score=chunk.score,
                provenance=chunk.provenance,
            )
            total_tokens += self.budget.estimate_tokens(chunk.content)

        check = self.budget.record(total_tokens)
        if check.status.value in ("critical", "warning", "trend"):
            logger.warning("Budget [%s]: %s", intent.value, check.message)

        if use_cache:
            if len(self._query_cache) >= _CACHE_MAX_SIZE:
                oldest = next(iter(self._query_cache))
                del self._query_cache[oldest]
            self._query_cache[key] = fused

        return fused

    # ------------------------------------------------------------------
    # Phase 8 — rerank, pack, prepare, confidence, cache helpers
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
    ) -> dict:
        """
        All-in-one context preparation for LLM agents:
          retrieve → rerank → pack → assess_confidence → KG context

        Returns everything needed to answer the query in a single call.
        """
        from .retrieval.kg_retrieval import extract_query_entities

        chunks = self.retrieve(query, limit=limit, as_of=as_of)
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
    # Cache management (Phase 8)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _cache_key(self, query: str, limit: int, as_of: str | None,
                   use_links: bool, intent_val: str) -> str:
        import hashlib
        raw = f"{query}|{limit}|{as_of}|{use_links}|{intent_val}"
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
        self.kg.add_triple(
            subject, predicate, object_,
            valid_from=valid_from, valid_to=valid_to, source=source,
        )
        return {"added": f"{subject} --{predicate}--> {object_}"}

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
    # Phase 3 — Entity resolution
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
    # Phase 3 — Contradiction detection
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
    # Session lifecycle
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase 4 — Explainability
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
            terms = tokenize(query)
            matched = [t for t in terms if chunk_id in self.corpus.get_posting_list(t)]
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
    # Phase 7 — KG traversal, doctor, export
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
            "corpus": self.corpus.stats(),
            "kg": self.kg.stats(),
            "bulletin": self.bulletin.stats(),
            "budget": self.budget.stats(),
            "resolver": self.resolver.stats(),
        }
