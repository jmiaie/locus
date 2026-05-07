"""
LocusEngine — main orchestrator for Locus vectorless RAG.

Three-signal retrieval pipeline, no embeddings required:
  1. BM25     — probabilistic keyword retrieval
  2. KG       — entity expansion via temporal knowledge graph
  3. LinkWalk — wikilink/citation graph traversal from top BM25 hits

Fused via Reciprocal Rank Fusion. Tiered bulletin tracks hot context
across rounds. Token budget monitors context injection cost softly.
"""

import logging
from pathlib import Path

from .memory.corpus import Corpus
from .memory.knowledge_graph import TemporalKG
from .retrieval.bm25 import BM25Retriever, ScoredChunk
from .retrieval.kg_retrieval import KGRetriever
from .retrieval.link_walker import LinkWalker
from .retrieval.fusion import rrf_fuse
from .context.bulletin import ContextBulletin
from .context.budget import ContextBudget

logger = logging.getLogger(__name__)

__version__ = "0.1.0"


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

        self.corpus = Corpus(self.store_path / "corpus")
        self.kg = TemporalKG(str(self.store_path / "kg.sqlite3"))
        self.bulletin = ContextBulletin(archive_path=self.store_path / "archive")
        self.budget = ContextBudget()

        self._bm25 = BM25Retriever(self.corpus)
        self._kg_ret = KGRetriever(self.kg, self.corpus)
        self._walker = LinkWalker(self.corpus)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, path: str | Path, pattern: str = "**/*.md") -> dict:
        """Index a file or directory. Returns counts of files, chunks, KG triples."""
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
        """Remove a document from the corpus."""
        removed = self.corpus.remove_file(doc_path)
        return {"removed_chunks": removed, "doc": doc_path}

    def sync(self, path: str | Path, pattern: str = "**/*.md") -> dict:
        """Full reindex: wipe corpus and rebuild from path."""
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
    ) -> list[ScoredChunk]:
        """
        Main retrieval: BM25 + KG entity expansion + link walk, fused via RRF.
        Each returned chunk carries a provenance tag (bm25 / kg / link:hopN).
        """
        bm25_hits = self._bm25.search(query, limit=limit * 2)
        kg_hits = self._kg_ret.search(query, limit=limit * 2, as_of=as_of)

        link_hits: list[ScoredChunk] = []
        if use_links and bm25_hits:
            link_hits = self._walker.walk(bm25_hits[:3], depth=2, limit=limit)

        fused = rrf_fuse([bm25_hits, kg_hits, link_hits], limit=limit)

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
            logger.warning("Budget alert: %s", check.message)

        return fused

    def format_context(
        self, chunks: list[ScoredChunk], include_hot: bool = True
    ) -> str:
        """Format retrieved chunks as a context block ready for prompt injection."""
        parts: list[str] = []
        if include_hot:
            hot = self.bulletin.inject(token_limit=800)
            if hot:
                parts.append(f"## Locus Hot Context\n{hot}")
        if chunks:
            parts.append("## Retrieved Context")
            for i, chunk in enumerate(chunks, 1):
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
    # Session lifecycle (mirrors OMPA's session_start / stop pattern)
    # ------------------------------------------------------------------

    def session_start(self) -> dict:
        """Warm context open: corpus stats + KG stats + hot-tier bulletin."""
        return {
            "corpus": self.corpus.stats(),
            "kg": self.kg.stats(),
            "bulletin": self.bulletin.stats(),
            "hot_context": self.bulletin.inject(token_limit=800) or "(empty — run locus_index first)",
        }

    def wrap_up(self) -> dict:
        """Session close: tick bulletin decay, return summary."""
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
        }
