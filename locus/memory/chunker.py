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
    return [(h, c) for h, c, _ in _split_sections_tree(body)]


def _split_sections_tree(body: str) -> list[tuple[str, str, list[str]]]:
    """Like :func:`_split_sections`, plus each section's heading path
    (ancestor headings by level, then its own heading)."""
    lines = body.splitlines()
    sections: list[tuple[str, str, list[str]]] = []
    current_heading = ""
    current_path: list[str] = []
    current_lines: list[str] = []
    stack: list[tuple[int, str]] = []

    for line in lines:
        m = re.match(r"^(#{1,6})\s+.", line)
        if m:
            if current_lines or current_heading:
                sections.append((current_heading, "\n".join(current_lines).strip(), current_path))
            level = len(m.group(1))
            current_heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, current_heading))
            current_path = [h for _, h in stack]
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or current_heading:
        sections.append((current_heading, "\n".join(current_lines).strip(), current_path))

    return sections


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Character (start, end) of every whitespace-separated word in *text*."""
    return [m.span() for m in re.finditer(r"\S+", text)]


#: Chunking options (see :class:`Chunker`) and their defaults.
CHUNKING_DEFAULTS = {
    "chunk_words": CHUNK_WORDS,
    "overlap_words": OVERLAP_WORDS,
    "heading_context": False,
    "boundaries": "words",
}
BOUNDARIES = ("words", "lines")


def normalize_chunking(cfg: dict | None) -> dict:
    """Validate a chunking config and fill in the defaults."""
    cfg = dict(cfg or {})
    unknown = set(cfg) - set(CHUNKING_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown chunking option(s) {sorted(unknown)}; use {sorted(CHUNKING_DEFAULTS)}")
    out = {**CHUNKING_DEFAULTS, **cfg}
    out["chunk_words"] = int(out["chunk_words"])
    out["overlap_words"] = int(out["overlap_words"])
    out["heading_context"] = bool(out["heading_context"])
    if out["chunk_words"] < 1:
        raise ValueError("chunking.chunk_words must be >= 1")
    if not 0 <= out["overlap_words"] < out["chunk_words"]:
        raise ValueError("chunking.overlap_words must be >= 0 and < chunk_words")
    if out["boundaries"] not in BOUNDARIES:
        raise ValueError(f"chunking.boundaries must be one of {BOUNDARIES}")
    return out


@dataclass
class Section:
    """A parent unit: one heading section (or the whole body when it has none)."""
    id: str
    doc_path: str
    ordinal: int
    heading: str
    content: str


class Chunker:
    """
    Split documents into retrieval chunks.

    chunk_words / overlap_words: window size and overlap for sections (or
    heading-less bodies) longer than *chunk_words*.

    heading_context: prefix every chunk with its heading path
    (``"Spec > 03 30 00 > Slabs"``), so windows cut from the middle of a long
    section still carry the section's title.  Off by default (the original
    behaviour: only the first window of a section contains its heading).

    boundaries: ``"words"`` (fixed windows, the original behaviour) or
    ``"lines"`` (window ends snap back to the nearest line/paragraph break in
    the second half of the window, so chunks do not cut sentences or
    conversation turns in two when a break exists).

    Every chunk records its parent section (``metadata["parent_id"]``) and the
    character span of its text inside that section (``metadata["span"]``), so
    retrieval can return the enclosing section or document with the matched
    passages cited (``LocusEngine.retrieve(..., parent="section")``).
    """

    def __init__(
        self,
        chunk_words: int = CHUNK_WORDS,
        overlap_words: int = OVERLAP_WORDS,
        section_aware: bool = True,
        heading_context: bool = False,
        boundaries: str = "words",
    ):
        cfg = normalize_chunking({
            "chunk_words": chunk_words, "overlap_words": overlap_words,
            "heading_context": heading_context, "boundaries": boundaries,
        })
        self.chunk_words = cfg["chunk_words"]
        self.overlap_words = cfg["overlap_words"]
        self.heading_context = cfg["heading_context"]
        self.boundaries = cfg["boundaries"]
        self.section_aware = section_aware

    @classmethod
    def from_config(cls, cfg: dict | None, section_aware: bool = True) -> "Chunker":
        return cls(section_aware=section_aware, **normalize_chunking(cfg))

    def config(self) -> dict:
        return {"chunk_words": self.chunk_words, "overlap_words": self.overlap_words,
                "heading_context": self.heading_context, "boundaries": self.boundaries}

    def chunk_file(self, path: Path, base_path: Path = None) -> list[Chunk]:
        return self.chunk_file_with_sections(path, base_path)[0]

    def chunk_file_with_sections(
        self, path: Path, base_path: Path = None
    ) -> tuple[list[Chunk], list[Section]]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            return [], []
        doc_id = (
            str(path.relative_to(base_path)).replace("\\", "/")
            if base_path
            else str(path).replace("\\", "/")
        )
        return self.chunk_text_with_sections(text, source=doc_id)

    def chunk_text(self, text: str, source: str = "") -> list[Chunk]:
        return self.chunk_text_with_sections(text, source)[0]

    def chunk_text_with_sections(
        self, text: str, source: str = ""
    ) -> tuple[list[Chunk], list[Section]]:
        meta, body = _extract_frontmatter(text)
        has_headings = bool(re.search(r"^#{1,6}\s+.", body, re.MULTILINE))
        sections: list[Section] = []
        chunks: list[Chunk] = []
        if self.section_aware and has_headings:
            for heading, content, hpath in _split_sections_tree(body):
                if not content.strip() and not heading:
                    continue
                full = f"{heading}\n\n{content}".strip() if heading else content.strip()
                sec = self._section(source, len(sections), heading, full)
                n_before = len(chunks)
                if len(full.split()) <= self.chunk_words and not self.heading_context:
                    if len(full) < 10:
                        continue
                    cid = hashlib.sha256(f"{source}:section:{heading}".encode()).hexdigest()[:16]
                    chunks.append(self._chunk(cid, source, full, 0, meta, heading, sec, (0, len(full))))
                elif self.heading_context:
                    body_text = content.strip()
                    prefix = " > ".join(hpath) if hpath else heading
                    offset = len(full) - len(body_text)
                    chunks.extend(self._windows(body_text, source, meta, heading, sec,
                                                prefix=prefix, offset=offset,
                                                single_id=f"{source}:section:{heading}"))
                else:
                    chunks.extend(self._windows(full, source, meta, heading, sec))
                if len(chunks) > n_before:
                    sections.append(sec)
            return chunks, sections

        stripped = body.strip()
        sec = self._section(source, 0, "", stripped)
        chunks = self._windows(stripped, source, meta, "", sec)
        return chunks, ([sec] if chunks else [])

    # ------------------------------------------------------------------

    @staticmethod
    def _section(source: str, ordinal: int, heading: str, content: str) -> Section:
        sid = hashlib.sha256(f"{source}:parent:{ordinal}".encode()).hexdigest()[:16]
        return Section(id=sid, doc_path=source, ordinal=ordinal, heading=heading, content=content)

    @staticmethod
    def _chunk(cid, source, content, start_word, meta, section, sec: Section, span) -> Chunk:
        return Chunk(
            id=cid,
            doc_path=source,
            content=content,
            start_word=start_word,
            metadata={
                "frontmatter": meta,
                "links": _extract_links(content),
                "entities": _extract_entities(content),
                "date": meta.get("date"),
                "tags": meta.get("tags", ""),
                "section": section,
                "parent_id": sec.id,
                "span": [span[0], span[1]],
            },
        )

    def _window_bounds(self, text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Word-index windows [start, end) covering *spans*."""
        n = len(spans)
        cw, ov = self.chunk_words, self.overlap_words
        if self.boundaries == "words":
            step = max(1, cw - ov)
            return [(s, min(s + cw, n)) for s in range(0, n, step)]
        # "lines": end each window at the last line break inside its second half.
        breaks = [i for i in range(1, n) if "\n" in text[spans[i - 1][1]:spans[i][0]]]
        out: list[tuple[int, int]] = []
        start = 0
        while start < n:
            end = min(start + cw, n)
            if end < n:
                lo = start + max(1, cw // 2)
                cands = [b for b in breaks if lo <= b <= end]
                if cands:
                    end = cands[-1]
            out.append((start, end))
            if end >= n:
                break
            start = max(start + 1, end - ov)
        return out

    def _windows(
        self, text: str, source: str, meta: dict, base_section: str, sec: Section,
        prefix: str = "", offset: int = 0, single_id: str | None = None,
    ) -> list[Chunk]:
        spans = _word_spans(text)
        if not spans:
            return []
        words = [text[a:b] for a, b in spans]
        bounds = self._window_bounds(text, spans)
        chunks: list[Chunk] = []
        for start, end in bounds:
            body = " ".join(words[start:end])
            content = f"{prefix}\n\n{body}" if prefix else body
            if len(content.strip()) < 10:
                continue
            if single_id is not None and len(bounds) == 1:
                key = single_id
            else:
                key = f"{source}:{base_section}:{start}"
            cid = hashlib.sha256(key.encode()).hexdigest()[:16]
            span = (offset + spans[start][0], offset + spans[end - 1][1])
            chunks.append(self._chunk(cid, source, content, start, meta, base_section, sec, span))
        return chunks
