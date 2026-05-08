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
from .retrieval.fusion import rrf_fuse
from .retrieval.classifier import classify_query, INTENT_WEIGHTS, QueryIntent
from .context.bulletin import ContextBulletin
from .context.budget import ContextBudget

logger = logging.getLogger(__name__)

__version__ = "0.3.0"

# Fixed weights appended after intent weights: [structural, recency]
_STRUCTURAL_WEIGHT = 1.0
_RECENCY_WEIGHT = 0.3


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
        self._structural = StructuralRetriever(self.corpus)   # Phase 2
        self._recency = RecencyRetriever(self.corpus)          # Phase 2

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
        return {
            "files": self.corpus.doc_count(),
            "chunks": chunks,
            "triples": triples,
        }

    def forget(self, doc_path: str) -> dict:
        removed = self.corpus.remove_file(doc_path)
        return {"removed_chunks": removed, "doc": doc_path}

    def sync(self, path: str | Path, pattern: str = "**/*.md") -> dict:
        for doc in self.corpus.list_docs():
            self.corpus.remove_file(doc)
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
    ) -> list[ScoredChunk]:
        """
        Five-signal retrieval: BM25 + KG + link walk + structural + recency.

        Query intent auto-classified unless overridden:
          - KG-first:   entity/factual queries (e.g. "who is Alice?")
          - BM25-first: procedural/narrative (e.g. "how does auth work?")
          - Balanced:   default

        Structural signal activates only when the query contains date,
        tag, or doc-type references. Recency is always a soft prior.
        """
        if intent is None:
            intent = classify_query(query)

        bm25_hits = self._bm25.search(query, limit=limit * 2)
        kg_hits = self._kg_ret.search(query, limit=limit * 2, as_of=as_of)
        link_hits = self._walker.walk(bm25_hits[:3], depth=2, limit=limit) if use_links and bm25_hits else []
        structural_hits = self._structural.search(query, limit=limit * 2)
        recency_hits = self._recency.search(limit=limit * 2)

        intent_weights = INTENT_WEIGHTS[intent]  # [bm25, kg, link]
        all_weights = [*intent_weights, _STRUCTURAL_WEIGHT, _RECENCY_WEIGHT]

        fused = rrf_fuse(
            [bm25_hits, kg_hits, link_hits, structural_hits, recency_hits],
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

        return fused

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
