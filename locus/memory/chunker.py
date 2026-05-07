"""
Document chunker — overlap-aware, frontmatter-aware, wikilink-extracting.
"""

import re
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CHUNK_WORDS = 400
OVERLAP_WORDS = 50


@dataclass
class Chunk:
    id: str
    doc_path: str
    content: str
    start_word: int
    metadata: dict = field(default_factory=dict)


def _extract_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_block = text[3:end].strip()
    body = text[end + 4:].strip()
    meta: dict = {}
    for line in fm_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body


def _extract_links(text: str) -> list[str]:
    links = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", text)
    links += re.findall(r"\[(?:[^\]]+)\]\(([^)#)]+)\)", text)
    return [lnk.strip() for lnk in links if lnk.strip()]


def _extract_entities(text: str) -> list[str]:
    entities = re.findall(r'"([^"]{3,40})"', text)
    entities += [w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text)]
    return list(dict.fromkeys(entities))[:20]


class Chunker:
    def __init__(
        self,
        chunk_words: int = CHUNK_WORDS,
        overlap_words: int = OVERLAP_WORDS,
    ):
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words

    def chunk_file(self, path: Path, base_path: Path = None) -> list[Chunk]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            return []
        doc_id = (
            str(path.relative_to(base_path)).replace("\\", "/")
            if base_path
            else str(path).replace("\\", "/")
        )
        return self.chunk_text(text, source=doc_id)

    def chunk_text(self, text: str, source: str = "") -> list[Chunk]:
        meta, body = _extract_frontmatter(text)
        words = body.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        step = max(1, self.chunk_words - self.overlap_words)
        for start in range(0, len(words), step):
            end = start + self.chunk_words
            content = " ".join(words[start:end])
            if len(content.strip()) < 10:
                continue
            chunk_id = hashlib.sha256(f"{source}:{start}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_path=source,
                    content=content,
                    start_word=start,
                    metadata={
                        "frontmatter": meta,
                        "links": _extract_links(content),
                        "entities": _extract_entities(content),
                        "date": meta.get("date"),
                        "tags": meta.get("tags", ""),
                    },
                )
            )
        return chunks
