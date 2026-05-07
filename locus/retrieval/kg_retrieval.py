"""
KG-guided retrieval — entity expansion via temporal knowledge graph.
Finds source documents that contain triples involving query entities,
then expands to related entities one hop out.
"""

import re
import logging

from ..memory.knowledge_graph import TemporalKG
from ..memory.corpus import Corpus, tokenize
from .bm25 import ScoredChunk

logger = logging.getLogger(__name__)


def extract_query_entities(query: str) -> list[str]:
    """Candidate entities from a query: quoted strings, capitalized words, long tokens."""
    entities: list[str] = []
    entities += re.findall(r'"([^"]+)"', query)
    entities += re.findall(r"'([^']+)'", query)
    entities += re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", query)
    entities += [t for t in tokenize(query) if len(t) > 4]
    return list(dict.fromkeys(entities))


class KGRetriever:
    def __init__(self, kg: TemporalKG, corpus: Corpus):
        self.kg = kg
        self.corpus = corpus

    def search(
        self, query: str, limit: int = 10, as_of: str = None
    ) -> list[ScoredChunk]:
        entities = extract_query_entities(query)
        if not entities:
            return []

        source_scores: dict[str, float] = {}

        for entity in entities[:8]:
            triples = self.kg.query_entity(entity, as_of=as_of)
            if not triples:
                continue
            sources = self.kg.sources_for_entity(entity, as_of=as_of)
            weight = 1.0 / (1 + len(triples))
            for src in sources:
                source_scores[src] = source_scores.get(src, 0.0) + weight
            # One-hop expansion
            for triple in triples[:5]:
                related = (
                    triple.object if triple.subject == entity else triple.subject
                )
                for src in self.kg.sources_for_entity(related, as_of=as_of):
                    source_scores[src] = source_scores.get(src, 0.0) + weight * 0.3

        ranked = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)

        results: list[ScoredChunk] = []
        for source, score in ranked:
            chunks = self.corpus.get_chunks_for_doc(source)
            if not chunks:
                continue
            c = chunks[0]
            results.append(
                ScoredChunk(
                    chunk_id=c.id,
                    doc_path=c.doc_path,
                    score=score,
                    content=c.content,
                    provenance="kg",
                    entities=c.metadata.get("entities", []),
                )
            )
            if len(results) >= limit:
                break
        return results
