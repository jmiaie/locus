"""
Corpus — SQLite-backed document store with inverted term index for BM25.
"""

import hashlib
import json
import logging
import sqlite3
import re
from pathlib import Path
from typing import Optional

from .chunker import Chunker, Chunk

logger = logging.getLogger(__name__)

_EXCLUDE = {".locus", ".palace", ".git", ".obsidian", "__pycache__", "node_modules"}

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "this", "that", "these", "those",
    "it", "its", "as", "not", "no", "so", "if", "then", "than", "more",
})


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 1]


class Corpus:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS chunks (
        id         TEXT PRIMARY KEY,
        doc_path   TEXT NOT NULL,
        content    TEXT NOT NULL,
        start_word INTEGER,
        metadata   TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(doc_path);

    CREATE TABLE IF NOT EXISTS term_index (
        term     TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        tf       REAL NOT NULL,
        PRIMARY KEY (term, chunk_id)
    );
    CREATE INDEX IF NOT EXISTS idx_term ON term_index(term);

    CREATE TABLE IF NOT EXISTS doc_stats (
        doc_path   TEXT PRIMARY KEY,
        word_count INTEGER,
        checksum   TEXT,
        indexed_at TEXT DEFAULT (datetime('now'))
    );
    """

    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.store_path / "corpus.sqlite3")
        self._chunker = Chunker()
        # In-process stats cache — invalidated on add/remove
        self._N_cache: Optional[int] = None
        self._avgdl_cache: Optional[float] = None
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self._SCHEMA)
            # Migration: add checksum column if corpus pre-dates Phase 1
            cols = {row[1] for row in conn.execute("PRAGMA table_info(doc_stats)").fetchall()}
            if "checksum" not in cols:
                conn.execute("ALTER TABLE doc_stats ADD COLUMN checksum TEXT")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _invalidate_stats(self) -> None:
        self._N_cache = None
        self._avgdl_cache = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _file_checksum(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    def _stored_checksum(self, doc_path: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT checksum FROM doc_stats WHERE doc_path = ?", (doc_path,)
            ).fetchone()
        return row[0] if row else None

    def add_file(self, path: Path, base_path: Path = None, force: bool = False) -> int:
        """
        Index a file. Returns number of chunks added.
        Returns 0 (skips) if the file is unchanged since last index, unless force=True.
        """
        chunks = self._chunker.chunk_file(path, base_path=base_path)
        if not chunks:
            return 0
        doc_id = chunks[0].doc_path

        # Checksum dedup — skip unchanged files
        if not force:
            try:
                checksum = self._file_checksum(path)
                if checksum == self._stored_checksum(doc_id):
                    logger.debug("Skipping unchanged file: %s", doc_id)
                    return 0
            except Exception:
                pass  # if checksum fails, proceed with indexing

        self._remove_doc(doc_id)
        for chunk in chunks:
            self._store_chunk(chunk)

        word_count = sum(len(c.content.split()) for c in chunks)
        try:
            checksum = self._file_checksum(path)
        except Exception:
            checksum = None

        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO doc_stats (doc_path, word_count, checksum) VALUES (?, ?, ?)",
                (doc_id, word_count, checksum),
            )
        self._invalidate_stats()
        return len(chunks)

    def add_directory(self, path: Path, pattern: str = "**/*.md") -> int:
        path = Path(path)
        total = 0
        for f in path.glob(pattern):
            if any(ex in f.parts for ex in _EXCLUDE):
                continue
            total += self.add_file(f, base_path=path)
        return total

    def remove_file(self, doc_path: str) -> int:
        return self._remove_doc(doc_path)

    def _remove_doc(self, doc_path: str) -> int:
        with self._conn() as conn:
            ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM chunks WHERE doc_path = ?", (doc_path,)
                ).fetchall()
            ]
            if not ids:
                return 0
            ph = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM term_index WHERE chunk_id IN ({ph})", ids)
            conn.execute("DELETE FROM chunks WHERE doc_path = ?", (doc_path,))
            conn.execute("DELETE FROM doc_stats WHERE doc_path = ?", (doc_path,))
        self._invalidate_stats()
        return len(ids)

    def _store_chunk(self, chunk: Chunk) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO chunks (id, doc_path, content, start_word, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    chunk.id,
                    chunk.doc_path,
                    chunk.content,
                    chunk.start_word,
                    json.dumps(chunk.metadata),
                ),
            )
        terms = tokenize(chunk.content)
        if not terms:
            return
        freq: dict[str, int] = {}
        for t in terms:
            freq[t] = freq.get(t, 0) + 1
        total = len(terms)
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO term_index (term, chunk_id, tf) VALUES (?, ?, ?)",
                [(t, chunk.id, count / total) for t, count in freq.items()],
            )

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, doc_path, content, start_word, metadata "
                "FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
        if not row:
            return None
        return Chunk(
            id=row[0], doc_path=row[1], content=row[2],
            start_word=row[3], metadata=json.loads(row[4] or "{}"),
        )

    def get_chunks_batch(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """Fetch multiple chunks in a single query."""
        if not chunk_ids:
            return {}
        ph = ",".join("?" * len(chunk_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, doc_path, content, start_word, metadata "
                f"FROM chunks WHERE id IN ({ph})",
                chunk_ids,
            ).fetchall()
        return {
            r[0]: Chunk(
                id=r[0], doc_path=r[1], content=r[2],
                start_word=r[3], metadata=json.loads(r[4] or "{}"),
            )
            for r in rows
        }

    def get_chunks_for_doc(self, doc_path: str) -> list[Chunk]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, doc_path, content, start_word, metadata "
                "FROM chunks WHERE doc_path = ? ORDER BY start_word",
                (doc_path,),
            ).fetchall()
        return [
            Chunk(
                id=r[0], doc_path=r[1], content=r[2],
                start_word=r[3], metadata=json.loads(r[4] or "{}"),
            )
            for r in rows
        ]

    def get_posting_list(self, term: str) -> dict[str, float]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, tf FROM term_index WHERE term = ?", (term,)
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def doc_count(self) -> int:
        if self._N_cache is None:
            with self._conn() as conn:
                self._N_cache = conn.execute("SELECT COUNT(*) FROM doc_stats").fetchone()[0]
        return self._N_cache

    def chunk_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def avg_doc_length(self) -> float:
        if self._avgdl_cache is None:
            with self._conn() as conn:
                result = conn.execute("SELECT AVG(word_count) FROM doc_stats").fetchone()[0]
            self._avgdl_cache = result or 0.0
        return self._avgdl_cache

    def list_docs(self) -> list[str]:
        with self._conn() as conn:
            return [
                r[0]
                for r in conn.execute("SELECT doc_path FROM doc_stats").fetchall()
            ]

    def stats(self) -> dict:
        return {
            "doc_count": self.doc_count(),
            "chunk_count": self.chunk_count(),
            "avg_doc_length": round(self.avg_doc_length(), 1),
        }
