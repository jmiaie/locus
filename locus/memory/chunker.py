"""
Document chunker — section-aware (default) or overlap word-count fallback.

Section-aware mode (default):
  - Splits at markdown heading boundaries (# / ## / ###)
  - Each heading + content becomes one chunk, with the heading prepended
  - Sections that exceed chunk_words are further split by word count
  - Falls back to word-count mode for documents with no headings

Word-count mode:
  - Rolling window of chunk_words with overlap_words overlap
  - Same behaviour as v0.1.0
"""

import hashlib
import re
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


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split body text into (heading, content) pairs at markdown headings."""
    lines = body.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in lines:
        if re.match(r"^#{1,6}\s+.", line):
            if current_lines or current_heading:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or current_heading:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections


class Chunker:
    def __init__(
        self,
        chunk_words: int = CHUNK_WORDS,
        overlap_words: int = OVERLAP_WORDS,
        section_aware: bool = True,
    ):
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.section_aware = section_aware

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
        has_headings = bool(re.search(r"^#{1,6}\s+.", body, re.MULTILINE))

        if self.section_aware and has_headings:
            return self._chunk_sections(body, source, meta)
        return self._chunk_words(body, source, meta)

    # ------------------------------------------------------------------
    # Section-aware path
    # ------------------------------------------------------------------

    def _chunk_sections(
        self, body: str, source: str, meta: dict
    ) -> list[Chunk]:
        sections = _split_sections(body)
        chunks: list[Chunk] = []

        for heading, content in sections:
            if not content.strip() and not heading:
                continue
            full = f"{heading}\n\n{content}".strip() if heading else content.strip()
            words = full.split()

            if len(words) <= self.chunk_words:
                if len(full) < 10:
                    continue
                cid = hashlib.sha256(
                    f"{source}:section:{heading}".encode()
                ).hexdigest()[:16]
                chunks.append(
                    Chunk(
                        id=cid,
                        doc_path=source,
                        content=full,
                        start_word=0,
                        metadata={
                            "frontmatter": meta,
                            "links": _extract_links(full),
                            "entities": _extract_entities(full),
                            "date": meta.get("date"),
                            "tags": meta.get("tags", ""),
                            "section": heading,
                        },
                    )
                )
            else:
                # Section too long — fall back to word-count within it
                chunks.extend(
                    self._chunk_words(full, source, meta, base_section=heading)
                )

        return chunks

    # ------------------------------------------------------------------
    # Word-count path
    # ------------------------------------------------------------------

    def _chunk_words(
        self,
        body: str,
        source: str,
        meta: dict,
        base_section: str = "",
    ) -> list[Chunk]:
        words = body.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        step = max(1, self.chunk_words - self.overlap_words)

        for start in range(0, len(words), step):
            content = " ".join(words[start : start + self.chunk_words])
            if len(content.strip()) < 10:
                continue
            cid = hashlib.sha256(
                f"{source}:{base_section}:{start}".encode()
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=cid,
                    doc_path=source,
                    content=content,
                    start_word=start,
                    metadata={
                        "frontmatter": meta,
                        "links": _extract_links(content),
                        "entities": _extract_entities(content),
                        "date": meta.get("date"),
                        "tags": meta.get("tags", ""),
                        "section": base_section,
                    },
                )
            )
        return chunks
