"""
Corpus — SQLite-backed document store with a lexical (BM25) index.

Two interchangeable index backends, both pure standard library:

* ``fts5`` (default) — a SQLite FTS5 inverted index ranked with FTS5's
  built-in ``bm25()``.  Query cost grows with the size of the matching
  posting lists, not with the corpus, so it stays fast at 100k+ chunks.
* ``legacy`` — the original ``term_index`` table scored in Python.  Used
  automatically when the interpreter's SQLite was built without FTS5, and
  selectable with ``LOCUS_INDEX_BACKEND=legacy``.

Both backends index the *same* term stream (:func:`analyze`; Porter-stemmed
by default, see ``ANALYZERS``) and use the
same BM25 formula (k1=1.2, b=0.75, FTS5's IDF), so they rank identically up
to floating-point rounding.  Ties are broken by index order (chunk rowid),
never by hash order, so results are reproducible run to run.

Stores written by Locus <= 1.0.0 (``term_index`` only) are migrated
in place to FTS5 the first time they are opened; see :meth:`Corpus.rebuild_index`
for an explicit clean re-index path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from .chunker import CHUNKING_DEFAULTS, Chunker, Chunk, Section, normalize_chunking
from .stemmer import stem

logger = logging.getLogger(__name__)

_EXCLUDE = {".locus", ".palace", ".git", ".obsidian", "__pycache__", "node_modules"}

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "this", "that", "these", "those",
    "it", "its", "as", "not", "no", "so", "if", "then", "than", "more",
})

# Optional larger English stop-word list (function words: pronouns,
# auxiliaries, prepositions, conjunctions), used by the "+english" analyzers.
ENGLISH_STOPWORDS = STOPWORDS | frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself they them their theirs themselves what which
who whom am about against between into through during before after above below
up down out off over under again further once here there when where why how all
any both each few most other some such nor only own same too very can just now
also onto upon via
""".split())

# Lexical analyzers (how text becomes index terms).  The analyzer is stored in
# the index; opening a store with a different analyzer rebuilds the index from
# the stored chunk text (no source files are re-read).
#   plain           lower-case, \w tokens, Locus stop-words (Locus <= 1.1)
#   porter          plain + Porter stemming (locus.memory.stemmer)
#   porter+english  porter + the larger ENGLISH_STOPWORDS list
ANALYZERS = ("plain", "porter", "porter+english")
DEFAULT_ANALYZER = "porter+english"

# BM25 parameters.  FTS5's bm25() hard-codes k1=1.2, b=0.75; the legacy
# backend uses the same values so the two backends agree.
BM25_K1 = 1.2
BM25_B = 0.75

SCHEMA_VERSION = "2"
BACKENDS = ("fts5", "legacy")

_FTS_TOKENIZER = "ascii tokenchars '_'"


def tokenize(text: str) -> list[str]:
    """Plain tokens: lower-case, \\w runs, Locus stop-words removed, len > 1.

    Unstemmed.  Index terms come from :func:`analyze`, which may stem.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 1]


def _env_analyzer() -> Optional[str]:
    env = os.environ.get("LOCUS_ANALYZER", "").strip().lower()
    if env and env not in ANALYZERS:
        logger.warning("Ignoring LOCUS_ANALYZER=%r; use one of %s", env, ANALYZERS)
        return None
    return env or None


def analyze(text: str, analyzer: str = "plain") -> list[str]:
    """Index/query terms for *text* under *analyzer* (see ``ANALYZERS``)."""
    if analyzer == "plain":
        return tokenize(text)
    if analyzer not in ANALYZERS:
        raise ValueError(f"Unknown analyzer {analyzer!r}; use one of {ANALYZERS}")
    terms = tokenize(text)
    if analyzer == "porter+english":
        terms = [t for t in terms if t not in ENGLISH_STOPWORDS]
    return [stem(t) for t in terms]


_FTS5_AVAILABLE: Optional[bool] = None


def _env_chunking() -> Optional[dict]:
    raw = os.environ.get("LOCUS_CHUNKING", "").strip()
    if not raw:
        return None
    try:
        return normalize_chunking(json.loads(raw))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid LOCUS_CHUNKING={raw!r}: {e}") from None


SYMLINK_MODES = (False, "inside", True)


def _symlink_mode(value=None):
    """Resolve the symlink policy: argument, else ``LOCUS_FOLLOW_SYMLINKS``, else False."""
    if value is None:
        value = os.environ.get("LOCUS_FOLLOW_SYMLINKS", "").strip().lower() or False
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("0", "false", "no", "off", "none", ""):
            return False
        if v in ("1", "true", "yes", "on", "all"):
            return True
        if v == "inside":
            return "inside"
        raise ValueError(f"follow_symlinks must be one of {SYMLINK_MODES}, got {value!r}")
    if value not in SYMLINK_MODES:
        raise ValueError(f"follow_symlinks must be one of {SYMLINK_MODES}, got {value!r}")
    return value


def iter_index_files(root: Path, pattern: str = "**/*.md", follow_symlinks=None) -> list[Path]:
    """
    Files under *root* matching *pattern* that should be indexed, sorted.

    Tool directories (``.locus``, ``.git``, ...) are skipped.  Symlinks (the
    file itself or any directory between *root* and it) are handled by
    *follow_symlinks* (default ``LOCUS_FOLLOW_SYMLINKS`` or False):

    * ``False`` — skip them, so indexing never reads outside the tree.
    * ``"inside"`` — follow only links that resolve inside *root*.
    * ``True`` — follow every link (the behaviour before 1.2).
    """
    root = Path(root)
    mode = _symlink_mode(follow_symlinks)
    real_root = root.resolve()
    out = []
    for f in root.glob(pattern):
        if any(ex in f.parts for ex in _EXCLUDE):
            continue
        if mode is not True:
            linked = False
            cur = root
            for part in f.relative_to(root).parts:
                cur = cur / part
                if cur.is_symlink():
                    linked = True
                    break
            if linked:
                if mode is False:
                    logger.debug("Skipping symlink %s (follow_symlinks=False)", f)
                    continue
                try:
                    f.resolve().relative_to(real_root)
                except (ValueError, OSError, RuntimeError):
                    logger.debug("Skipping symlink %s: resolves outside %s", f, root)
                    continue
        if f.is_file():
            out.append(f)
    return sorted(out)


def fts5_available() -> bool:
    """True if this Python's sqlite3 module was compiled with FTS5."""
    global _FTS5_AVAILABLE
    if _FTS5_AVAILABLE is None:
        try:
            conn = sqlite3.connect(":memory:")
            try:
                conn.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
                _FTS5_AVAILABLE = True
            finally:
                conn.close()
        except sqlite3.OperationalError:
            _FTS5_AVAILABLE = False
    return _FTS5_AVAILABLE


def _default_backend() -> str:
    env = os.environ.get("LOCUS_INDEX_BACKEND", "").strip().lower()
    if env in BACKENDS:
        if env == "fts5" and not fts5_available():
            logger.warning("LOCUS_INDEX_BACKEND=fts5 but SQLite lacks FTS5; using legacy")
            return "legacy"
        return env
    return "fts5" if fts5_available() else "legacy"


def _fts_query(terms: Iterable[str]) -> str:
    # Terms come from analyze(): only \w characters, so quoting is safe.
    return " OR ".join(f'"{t}"' for t in terms)


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

    CREATE TABLE IF NOT EXISTS sections (
        id       TEXT PRIMARY KEY,
        doc_path TEXT NOT NULL,
        ordinal  INTEGER NOT NULL,
        heading  TEXT,
        content  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_section_doc ON sections(doc_path);

    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """

    _FTS_SCHEMA = f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
        body, tokenize="{_FTS_TOKENIZER}"
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts_row USING fts5vocab(chunk_fts, row);
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts_inst USING fts5vocab(chunk_fts, instance);
    """

    def __init__(
        self,
        store_path: Path,
        section_aware: bool = True,
        backend: Optional[str] = None,
        analyzer: Optional[str] = None,
        chunking: Optional[dict] = None,
        follow_symlinks=None,
    ):
        self.store_path = Path(store_path)
        #: Symlink policy for directory indexing (see :func:`iter_index_files`).
        self.follow_symlinks = _symlink_mode(follow_symlinks)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.store_path / "corpus.sqlite3")
        self._section_aware = section_aware
        self._requested_chunking = normalize_chunking(chunking) if chunking is not None else _env_chunking()
        self.chunking: dict = dict(CHUNKING_DEFAULTS)
        self._chunker = Chunker(section_aware=section_aware)
        self._N_cache: Optional[int] = None
        self._avgdl_cache: Optional[float] = None
        self._chunk_stats_cache: Optional[tuple[int, float]] = None
        self._requested_backend = backend
        if analyzer is not None and analyzer not in ANALYZERS:
            raise ValueError(f"Unknown analyzer {analyzer!r}; use one of {ANALYZERS}")
        self._requested_analyzer = analyzer or _env_analyzer()
        self.analyzer: str = self._requested_analyzer or DEFAULT_ANALYZER
        self.backend: str = "legacy"
        #: Bumped on every write; lets in-memory caches detect stale data.
        self.generation: int = 0
        self._init_db()

    # ------------------------------------------------------------------
    # Schema, backend selection & migration
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        if self._requested_backend is not None and self._requested_backend not in BACKENDS:
            raise ValueError(f"Unknown index backend {self._requested_backend!r}; use one of {BACKENDS}")
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self._SCHEMA)
            cols = {row[1] for row in conn.execute("PRAGMA table_info(doc_stats)").fetchall()}
            if "checksum" not in cols:
                conn.execute("ALTER TABLE doc_stats ADD COLUMN checksum TEXT")
            ccols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            if "n_tokens" not in ccols:
                conn.execute("ALTER TABLE chunks ADD COLUMN n_tokens INTEGER")
            stored = self._meta_get(conn, "index_backend")
            # Stores written before analyzers existed were indexed "plain".
            meta_analyzer = self._meta_get(conn, "analyzer")
            stored_analyzer = meta_analyzer or "plain"
            has_chunks = conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None
            raw_chunking = self._meta_get(conn, "chunking")
            stored_chunking = normalize_chunking(json.loads(raw_chunking)) if raw_chunking else dict(CHUNKING_DEFAULTS)
            # Chunking: an explicit request (argument or LOCUS_CHUNKING) wins;
            # otherwise keep what the store was built with.  A change cannot be
            # applied from stored chunks, so the source files are re-chunked on
            # the next index() (their checksums are cleared here).
            self.chunking = self._requested_chunking or stored_chunking
            if self.chunking != stored_chunking and has_chunks:
                logger.warning("Chunking changed (%s -> %s); re-run index() to re-chunk documents",
                               stored_chunking, self.chunking)
                conn.execute("UPDATE doc_stats SET checksum = NULL")
            self._meta_set(conn, "chunking", json.dumps(self.chunking, sort_keys=True))
        self._chunker = Chunker.from_config(self.chunking, section_aware=self._section_aware)

        want = self._requested_backend or _default_backend()
        if want == "fts5" and not fts5_available():
            logger.warning("SQLite FTS5 not available; falling back to legacy index backend")
            want = "legacy"

        if stored is None:
            # New store, or a store written by Locus <= 1.0.0 (legacy term_index).
            current = "legacy" if has_chunks else want
        else:
            current = stored

        if current == "fts5" and not fts5_available():
            # Store was built with FTS5 on another interpreter: rebuild legacy postings.
            logger.warning("Store uses FTS5 but this SQLite lacks it; rebuilding legacy index")
            current = "fts5-unavailable"

        self.backend = want
        if want == "fts5":
            with self._conn() as conn:
                conn.executescript(self._FTS_SCHEMA)
        # Analyzer: an explicit request (argument or LOCUS_ANALYZER) wins; otherwise
        # keep what the store was built with; stores from before analyzers
        # existed (no meta) are upgraded to DEFAULT_ANALYZER.
        if self._requested_analyzer is None and meta_analyzer in ANALYZERS:
            self.analyzer = meta_analyzer
        if (current != want or stored_analyzer != self.analyzer) and has_chunks:
            logger.info("Re-indexing Locus store %s/%s -> %s/%s",
                        current, stored_analyzer, want, self.analyzer)
            self.rebuild_index(backend=want, _from=current)
        with self._conn() as conn:
            self._meta_set(conn, "index_backend", want)
            self._meta_set(conn, "analyzer", self.analyzer)
            self._meta_set(conn, "schema_version", SCHEMA_VERSION)

    @staticmethod
    def _meta_get(conn: sqlite3.Connection, key: str) -> Optional[str]:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def analyze(self, text: str) -> list[str]:
        """Terms for *text* under this store's analyzer (use for queries too)."""
        return analyze(text, self.analyzer)

    def rebuild_index(
        self,
        backend: Optional[str] = None,
        _from: Optional[str] = None,
        analyzer: Optional[str] = None,
    ) -> dict:
        """
        Rebuild the lexical index from the stored chunk text.

        Used for automatic migration of pre-1.1 stores and available as an
        explicit clean re-index path (``locus reindex``).  The chunk table is
        the source of truth, so no source files need to be re-read.
        Returns ``{"backend": ..., "chunks": n}``.
        """
        backend = backend or self.backend
        if analyzer is not None:
            if analyzer not in ANALYZERS:
                raise ValueError(f"Unknown analyzer {analyzer!r}; use one of {ANALYZERS}")
            self.analyzer = analyzer
        if backend == "fts5" and not fts5_available():
            raise RuntimeError("SQLite FTS5 is not available in this Python build")
        with self._conn() as conn:
            if backend == "fts5":
                conn.executescript(self._FTS_SCHEMA)
                conn.execute("DELETE FROM chunk_fts")
            else:
                conn.execute("DELETE FROM term_index")
            n = 0
            cur = conn.execute("SELECT rowid, id, content FROM chunks ORDER BY rowid")
            batch_fts: list[tuple[int, str]] = []
            batch_terms: list[tuple[str, str, float]] = []
            batch_len: list[tuple[int, int]] = []
            for rowid, cid, content in cur.fetchall():
                terms = self.analyze(content)
                batch_len.append((len(terms), rowid))
                if backend == "fts5":
                    batch_fts.append((rowid, " ".join(terms)))
                else:
                    batch_terms.extend(self._term_rows(cid, terms))
                n += 1
            conn.executemany("UPDATE chunks SET n_tokens = ? WHERE rowid = ?", batch_len)
            if backend == "fts5":
                conn.executemany("INSERT INTO chunk_fts(rowid, body) VALUES (?, ?)", batch_fts)
                # Legacy postings are superseded; drop them to reclaim space.
                conn.execute("DELETE FROM term_index")
            else:
                conn.executemany(
                    "INSERT OR REPLACE INTO term_index (term, chunk_id, tf) VALUES (?, ?, ?)",
                    batch_terms,
                )
                try:
                    conn.execute("DELETE FROM chunk_fts")
                except sqlite3.OperationalError:
                    pass  # no FTS5 table / module
            self._meta_set(conn, "index_backend", backend)
            self._meta_set(conn, "analyzer", self.analyzer)
        self.backend = backend
        self._invalidate_stats()
        return {"backend": backend, "analyzer": self.analyzer, "chunks": n, "migrated_from": _from}

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _invalidate_stats(self) -> None:
        self.generation += 1
        self._N_cache = None
        self._avgdl_cache = None
        self._chunk_stats_cache = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _file_checksum(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _stored_checksum(self, doc_path: str, conn: sqlite3.Connection | None = None) -> Optional[str]:
        if conn is None:
            with self._conn() as c:
                return self._stored_checksum(doc_path, c)
        row = conn.execute(
            "SELECT checksum FROM doc_stats WHERE doc_path = ?", (doc_path,)
        ).fetchone()
        return row[0] if row else None

    def add_file(self, path: Path, base_path: Path = None, force: bool = False) -> int:
        """
        Index a file. Returns number of chunks added.
        Skips unchanged files (checksum dedup) unless force=True.
        """
        with self._conn() as conn:
            added = self._add_file(conn, Path(path), base_path=base_path, force=force)
        self._invalidate_stats()
        return added

    def _add_file(
        self,
        conn: sqlite3.Connection,
        path: Path,
        base_path: Path | None = None,
        force: bool = False,
    ) -> int:
        chunks, sections = self._chunker.chunk_file_with_sections(path, base_path=base_path)
        if not chunks:
            return 0
        doc_id = chunks[0].doc_path
        try:
            checksum: Optional[str] = self._file_checksum(path)
        except Exception:
            logger.debug("Checksum failed for %s, re-indexing", path)
            checksum = None
        if not force and checksum is not None and checksum == self._stored_checksum(doc_id, conn):
            return 0

        self._remove_doc_conn(conn, doc_id)
        # Chunk ids hash (source, section); a repeated heading yields a repeated
        # id.  Keep the last occurrence (same as the old INSERT OR REPLACE).
        unique: dict[str, Chunk] = {}
        for chunk in chunks:
            unique.pop(chunk.id, None)
            unique[chunk.id] = chunk
        for chunk in unique.values():
            self._store_chunk_conn(conn, chunk)
        conn.executemany(
            "INSERT OR REPLACE INTO sections (id, doc_path, ordinal, heading, content) VALUES (?, ?, ?, ?, ?)",
            [(sec.id, sec.doc_path, sec.ordinal, sec.heading, sec.content) for sec in sections],
        )

        word_count = sum(len(c.content.split()) for c in unique.values())
        conn.execute(
            "INSERT OR REPLACE INTO doc_stats (doc_path, word_count, checksum) VALUES (?, ?, ?)",
            (doc_id, word_count, checksum),
        )
        return len(unique)

    def iter_files(self, path: Path, pattern: str = "**/*.md", follow_symlinks=None) -> list[Path]:
        """Files under *path* matching *pattern* (see :func:`iter_index_files`)."""
        return iter_index_files(Path(path), pattern,
                                self.follow_symlinks if follow_symlinks is None else follow_symlinks)

    def add_directory(self, path: Path, pattern: str = "**/*.md") -> int:
        total, _ = self.add_directory_changed(path, pattern=pattern)
        return total

    def add_directory_changed(
        self, path: Path, pattern: str = "**/*.md", batch_size: int = 500,
    ) -> tuple[int, list[Path]]:
        """Index a directory; return (chunks added, files that changed)."""
        path = Path(path)
        total = 0
        changed: list[Path] = []
        conn = self._conn()
        try:
            conn.execute("PRAGMA synchronous = NORMAL")
            pending = 0
            for f in self.iter_files(path, pattern):
                n = self._add_file(conn, f, base_path=path)
                if n:
                    total += n
                    changed.append(f)
                    pending += 1
                if pending >= batch_size:
                    conn.commit()
                    pending = 0
            conn.commit()
        finally:
            conn.close()
        self._invalidate_stats()
        return total, changed

    def remove_file(self, doc_path: str) -> int:
        with self._conn() as conn:
            n = self._remove_doc_conn(conn, doc_path)
        self._invalidate_stats()
        return n

    def _remove_doc(self, doc_path: str) -> int:  # backwards-compatible alias
        return self.remove_file(doc_path)

    def _remove_doc_conn(self, conn: sqlite3.Connection, doc_path: str) -> int:
        rows = conn.execute(
            "SELECT rowid, id FROM chunks WHERE doc_path = ?", (doc_path,)
        ).fetchall()
        conn.execute("DELETE FROM doc_stats WHERE doc_path = ?", (doc_path,))
        conn.execute("DELETE FROM sections WHERE doc_path = ?", (doc_path,))
        if not rows:
            return 0
        if self.backend == "fts5":
            conn.executemany("DELETE FROM chunk_fts WHERE rowid = ?", [(r[0],) for r in rows])
        else:
            conn.executemany("DELETE FROM term_index WHERE chunk_id = ?", [(r[1],) for r in rows])
        conn.execute("DELETE FROM chunks WHERE doc_path = ?", (doc_path,))
        return len(rows)

    @staticmethod
    def _term_rows(chunk_id: str, terms: list[str]) -> list[tuple[str, str, float]]:
        freq: dict[str, int] = {}
        for t in terms:
            freq[t] = freq.get(t, 0) + 1
        total = len(terms)
        # tf stored normalised (count / chunk length), as in Locus <= 1.0.0
        return [(t, chunk_id, c / total) for t, c in sorted(freq.items())]

    def _store_chunk_conn(self, conn: sqlite3.Connection, chunk: Chunk) -> None:
        terms = self.analyze(chunk.content)
        cur = conn.execute(
            "INSERT OR REPLACE INTO chunks (id, doc_path, content, start_word, metadata, n_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chunk.id, chunk.doc_path, chunk.content, chunk.start_word,
             json.dumps(chunk.metadata, sort_keys=True), len(terms)),
        )
        if not terms:
            return
        if self.backend == "fts5":
            conn.execute(
                "INSERT INTO chunk_fts(rowid, body) VALUES (?, ?)",
                (cur.lastrowid, " ".join(terms)),
            )
        else:
            conn.executemany(
                "INSERT OR REPLACE INTO term_index (term, chunk_id, tf) VALUES (?, ?, ?)",
                self._term_rows(chunk.id, terms),
            )

    def _store_chunk(self, chunk: Chunk) -> None:  # kept for API compatibility
        with self._conn() as conn:
            self._store_chunk_conn(conn, chunk)
        self._invalidate_stats()

    # ------------------------------------------------------------------
    # BM25 search
    # ------------------------------------------------------------------

    def search_bm25(self, terms: list[str], limit: int = 10) -> list[tuple[str, float]]:
        """
        Rank chunks for the (already analysed, see :meth:`analyze`) query
        *terms* with BM25.

        Returns ``[(chunk_id, score), ...]`` sorted by score descending, ties
        broken by index order (chunk rowid; files are indexed in sorted path
        order).  Read-only and deterministic.
        """
        if not terms or limit <= 0:
            return []
        if self.backend == "fts5":
            return self._search_fts5(terms, limit)
        return self._search_legacy(terms, limit)

    _FTS_TOPK = (
        "SELECT rowid, -bm25(chunk_fts) AS s FROM chunk_fts "
        "WHERE chunk_fts MATCH ? ORDER BY s DESC, rowid LIMIT ?"
    )

    def _search_fts5(self, terms: list[str], limit: int) -> list[tuple[str, float]]:
        """Top-*limit* chunks by FTS5 ``bm25()``; ties broken by index order (rowid)."""
        with self._conn() as conn:
            rows = conn.execute(self._FTS_TOPK, (_fts_query(terms), limit)).fetchall()
            if not rows:
                return []
            ph = ",".join("?" * len(rows))
            ids = dict(conn.execute(
                f"SELECT rowid, id FROM chunks WHERE rowid IN ({ph})", [r[0] for r in rows]
            ).fetchall())
        return [(ids[r[0]], float(r[1])) for r in rows if r[0] in ids]

    def _chunk_stats(self, conn: sqlite3.Connection) -> tuple[int, float]:
        if self._chunk_stats_cache is None:
            missing = conn.execute(
                "SELECT rowid, content FROM chunks WHERE n_tokens IS NULL"
            ).fetchall()
            if missing:
                conn.executemany(
                    "UPDATE chunks SET n_tokens = ? WHERE rowid = ?",
                    [(len(self.analyze(c)), r) for r, c in missing],
                )
                conn.commit()
            n, avg = conn.execute(
                "SELECT COUNT(*), AVG(n_tokens) FROM chunks WHERE n_tokens > 0"
            ).fetchone()
            self._chunk_stats_cache = (n or 0, float(avg or 0.0))
        return self._chunk_stats_cache

    def _search_legacy(self, terms: list[str], limit: int) -> list[tuple[str, float]]:
        with self._conn() as conn:
            N, avgdl = self._chunk_stats(conn)
            if N == 0 or avgdl == 0:
                return []
            scores: dict[str, float] = {}
            order: dict[str, int] = {}
            for term in terms:
                rows = conn.execute(
                    "SELECT t.chunk_id, t.tf, c.n_tokens, c.rowid FROM term_index t "
                    "JOIN chunks c ON c.id = t.chunk_id WHERE t.term = ?",
                    (term,),
                ).fetchall()
                if not rows:
                    continue
                n = len(rows)
                idf = math.log((N - n + 0.5) / (n + 0.5))
                if idf <= 0:
                    idf = 1e-6
                for cid, tf_norm, dl, rowid in rows:
                    order[cid] = rowid
                    dl = dl or 0
                    tf = round(tf_norm * dl) if dl else 1
                    denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
                    scores[cid] = scores.get(cid, 0.0) + idf * tf * (BM25_K1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: (-x[1], order[x[0]]))
        return ranked[:limit]

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_chunk(row: tuple) -> Chunk:
        return Chunk(
            id=row[0], doc_path=row[1], content=row[2],
            start_word=row[3], metadata=json.loads(row[4] or "{}"),
        )

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, doc_path, content, start_word, metadata "
                "FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks_batch(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """Fetch multiple chunks in a single query."""
        if not chunk_ids:
            return {}
        out: dict[str, Chunk] = {}
        ids = list(chunk_ids)
        with self._conn() as conn:
            for i in range(0, len(ids), 900):
                part = ids[i:i + 900]
                ph = ",".join("?" * len(part))
                for r in conn.execute(
                    f"SELECT id, doc_path, content, start_word, metadata "
                    f"FROM chunks WHERE id IN ({ph})",
                    part,
                ).fetchall():
                    out[r[0]] = self._row_to_chunk(r)
        return out

    def get_chunks_for_doc(self, doc_path: str) -> list[Chunk]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, doc_path, content, start_word, metadata "
                "FROM chunks WHERE doc_path = ? ORDER BY start_word, rowid",
                (doc_path,),
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def first_chunks(self, doc_paths: list[str]) -> dict[str, Chunk]:
        """First chunk (by position) of each document, in one query."""
        if not doc_paths:
            return {}
        out: dict[str, Chunk] = {}
        paths = list(dict.fromkeys(doc_paths))
        with self._conn() as conn:
            for i in range(0, len(paths), 900):
                part = paths[i:i + 900]
                ph = ",".join("?" * len(part))
                for r in conn.execute(
                    f"SELECT id, doc_path, content, start_word, metadata FROM chunks "
                    f"WHERE doc_path IN ({ph}) ORDER BY doc_path, start_word, rowid",
                    part,
                ).fetchall():
                    if r[1] not in out:
                        out[r[1]] = self._row_to_chunk(r)
        return out

    def get_sections(self, section_ids: list[str]) -> dict[str, Section]:
        """Parent sections by id (see :class:`~locus.memory.chunker.Section`)."""
        out: dict[str, Section] = {}
        ids = list(dict.fromkeys(section_ids))
        with self._conn() as conn:
            for i in range(0, len(ids), 900):
                part = ids[i:i + 900]
                ph = ",".join("?" * len(part))
                for r in conn.execute(
                    f"SELECT id, doc_path, ordinal, heading, content FROM sections WHERE id IN ({ph})", part
                ).fetchall():
                    out[r[0]] = Section(*r)
        return out

    def doc_sections(self, doc_paths: list[str]) -> dict[str, list[Section]]:
        """All sections of each document, in document order."""
        out: dict[str, list[Section]] = {}
        paths = list(dict.fromkeys(doc_paths))
        with self._conn() as conn:
            for i in range(0, len(paths), 900):
                part = paths[i:i + 900]
                ph = ",".join("?" * len(part))
                for r in conn.execute(
                    f"SELECT id, doc_path, ordinal, heading, content FROM sections "
                    f"WHERE doc_path IN ({ph}) ORDER BY doc_path, ordinal", part
                ).fetchall():
                    out.setdefault(r[1], []).append(Section(*r))
        return out

    def score_terms(self, terms: list[str], chunk_ids: list[str]) -> dict[str, dict[str, float]]:
        """
        Per-term BM25 contribution for each of *chunk_ids*:
        ``{term: {chunk_id: score}}`` (chunks without the term are absent).

        Uses the same formula and corpus statistics as :meth:`search_bm25`, so
        summing a chunk's per-term scores reproduces its query score.  Used by
        weighted (e.g. RM3-expanded) queries, which FTS5 cannot express.
        """
        terms = list(dict.fromkeys(terms))
        ids = list(dict.fromkeys(chunk_ids))
        if not terms or not ids:
            return {}
        out: dict[str, dict[str, float]] = {}
        with self._conn() as conn:
            rowid_of: dict[int, str] = {}
            for i in range(0, len(ids), 900):
                part = ids[i:i + 900]
                ph = ",".join("?" * len(part))
                for rowid, cid in conn.execute(f"SELECT rowid, id FROM chunks WHERE id IN ({ph})", part):
                    rowid_of[rowid] = cid
            if not rowid_of:
                return {}
            if self.backend == "fts5":
                rowids = sorted(rowid_of)
                ph = ",".join("?" * len(rowids))
                sql = (f"SELECT rowid, -bm25(chunk_fts) FROM chunk_fts "
                       f"WHERE chunk_fts MATCH ? AND rowid IN ({ph})")
                for t in terms:
                    rows = conn.execute(sql, [_fts_query([t]), *rowids]).fetchall()
                    if rows:
                        out[t] = {rowid_of[r]: float(s) for r, s in rows}
                return out
            N, avgdl = self._chunk_stats(conn)
            if N == 0 or avgdl == 0:
                return {}
            wanted = set(rowid_of.values())
            for t in terms:
                rows = conn.execute(
                    "SELECT t.chunk_id, t.tf, c.n_tokens FROM term_index t "
                    "JOIN chunks c ON c.id = t.chunk_id WHERE t.term = ?", (t,),
                ).fetchall()
                if not rows:
                    continue
                n = len(rows)
                idf = math.log((N - n + 0.5) / (n + 0.5))
                if idf <= 0:
                    idf = 1e-6
                scores: dict[str, float] = {}
                for cid, tf_norm, dl in rows:
                    if cid not in wanted:
                        continue
                    dl = dl or 0
                    tf = round(tf_norm * dl) if dl else 1
                    scores[cid] = idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
                if scores:
                    out[t] = scores
        return out

    def chunk_order(self, chunk_ids: list[str]) -> dict[str, int]:
        """Index order (rowid) of each chunk, for deterministic tie-breaking."""
        out: dict[str, int] = {}
        ids = list(dict.fromkeys(chunk_ids))
        with self._conn() as conn:
            for i in range(0, len(ids), 900):
                part = ids[i:i + 900]
                ph = ",".join("?" * len(part))
                out.update({cid: r for r, cid in conn.execute(
                    f"SELECT rowid, id FROM chunks WHERE id IN ({ph})", part)})
        return out

    def doc_frontmatter(self, doc_paths: list[str]) -> dict[str, dict]:
        """Frontmatter of each document (from its first chunk), cached per write generation."""
        cache = getattr(self, "_fm_cache", None)
        if cache is None or cache[0] != self.generation:
            cache = (self.generation, {})
            self._fm_cache = cache
        store = cache[1]
        missing = [p for p in dict.fromkeys(doc_paths) if p not in store]
        if missing:
            for p, ch in self.first_chunks(missing).items():
                store[p] = dict(ch.metadata.get("frontmatter") or {})
            for p in missing:
                store.setdefault(p, {})
        return {p: store[p] for p in doc_paths}

    def supersedes_index(self) -> dict[str, list[dict]]:
        """``{superseded doc_path: [frontmatter of each doc that supersedes it]}`` (cached).

        Built from every document's ``supersedes`` frontmatter field (a doc
        path, or several separated by commas).
        """
        cache = getattr(self, "_sup_cache", None)
        if cache is not None and cache[0] == self.generation:
            return cache[1]
        index: dict[str, list[dict]] = {}
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT doc_path, metadata FROM chunks ORDER BY doc_path, start_word, rowid"
            ).fetchall()
        seen: set[str] = set()
        for doc_path, meta in rows:
            if doc_path in seen:
                continue
            seen.add(doc_path)
            try:
                fm = json.loads(meta or "{}").get("frontmatter") or {}
            except ValueError:
                fm = {}
            for target in str(fm.get("supersedes") or "").split(","):
                target = target.strip().strip("'\"")
                if target:
                    index.setdefault(target, []).append(fm)
        self._sup_cache = (self.generation, index)
        return index

    def iter_chunk_links(self):
        """Yield the ``links`` metadata list of every chunk (one query)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT metadata FROM chunks ORDER BY doc_path, rowid").fetchall()
        for (meta,) in rows:
            try:
                links = json.loads(meta or "{}").get("links", [])
            except ValueError:
                links = []
            if links:
                yield links

    def index_term(self, word: str) -> str:
        """The index term for a single *word* under this store's analyzer."""
        w = word.lower()
        return w if self.analyzer == "plain" else stem(w)

    def get_posting_list(self, term: str) -> dict[str, float]:
        """``{chunk_id: normalised tf}`` for every chunk containing *term*.

        *term* is a plain word; it is mapped to its index term (e.g. stemmed).
        """
        return self._posting_list(self.index_term(term))

    def _posting_list(self, term: str) -> dict[str, float]:
        """Posting list for an already-analysed index *term*."""
        if self.backend == "fts5":
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT c.id, COUNT(*), c.n_tokens FROM chunk_fts_inst v "
                    "JOIN chunks c ON c.rowid = v.doc WHERE v.term = ? "
                    "GROUP BY v.doc ORDER BY c.id",
                    (term,),
                ).fetchall()
            return {r[0]: (r[1] / r[2] if r[2] else 0.0) for r in rows}
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, tf FROM term_index WHERE term = ? ORDER BY chunk_id", (term,)
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def term_chunk_frequency(self, term: str) -> int:
        """Number of chunks containing *term* (document frequency)."""
        term = self.index_term(term)
        with self._conn() as conn:
            if self.backend == "fts5":
                row = conn.execute("SELECT doc FROM chunk_fts_row WHERE term = ?", (term,)).fetchone()
                return int(row[0]) if row else 0
            return conn.execute(
                "SELECT COUNT(*) FROM term_index WHERE term = ?", (term,)
            ).fetchone()[0]

    def doc_count(self) -> int:
        if self._N_cache is None:
            with self._conn() as conn:
                self._N_cache = conn.execute(
                    "SELECT COUNT(*) FROM doc_stats"
                ).fetchone()[0]
        return self._N_cache

    def chunk_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def avg_doc_length(self) -> float:
        if self._avgdl_cache is None:
            with self._conn() as conn:
                result = conn.execute(
                    "SELECT AVG(word_count) FROM doc_stats"
                ).fetchone()[0]
            self._avgdl_cache = result or 0.0
        return self._avgdl_cache

    def list_docs(self) -> list[str]:
        with self._conn() as conn:
            return [
                r[0]
                for r in conn.execute("SELECT doc_path FROM doc_stats ORDER BY doc_path").fetchall()
            ]

    def doc_index_times(self, doc_paths: list[str] | None = None) -> dict[str, Optional[str]]:
        """``{doc_path: indexed_at}`` for the given docs (all docs if None)."""
        with self._conn() as conn:
            if doc_paths is None:
                rows = conn.execute("SELECT doc_path, indexed_at FROM doc_stats").fetchall()
            else:
                rows = []
                paths = list(dict.fromkeys(doc_paths))
                for i in range(0, len(paths), 900):
                    part = paths[i:i + 900]
                    ph = ",".join("?" * len(part))
                    rows += conn.execute(
                        f"SELECT doc_path, indexed_at FROM doc_stats WHERE doc_path IN ({ph})", part
                    ).fetchall()
        return {r[0]: r[1] for r in rows}

    def stats(self) -> dict:
        return {
            "doc_count": self.doc_count(),
            "chunk_count": self.chunk_count(),
            "avg_doc_length": round(self.avg_doc_length(), 1),
            "index_backend": self.backend,
            "analyzer": self.analyzer,
        }

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def top_terms(self, limit: int = 20) -> list[dict]:
        """Return the *limit* most frequent terms across the entire corpus.

        Each entry: ``{"term": str, "doc_count": int, "total_tf": float}``
        where ``doc_count`` is the number of chunks containing the term.
        """
        if self.backend == "fts5":
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT term, doc FROM chunk_fts_row ORDER BY doc DESC, cnt DESC, term LIMIT ?",
                    (limit,),
                ).fetchall()
            out = []
            for term, doc in rows:
                total_tf = sum(self._posting_list(term).values())
                out.append({"term": term, "doc_count": int(doc), "total_tf": round(total_tf, 4)})
            return out
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT term,
                       COUNT(DISTINCT chunk_id) AS doc_count,
                       SUM(tf)                  AS total_tf
                FROM   term_index
                GROUP  BY term
                ORDER  BY doc_count DESC, total_tf DESC, term
                LIMIT  ?
                """,
                (limit,),
            ).fetchall()
        return [
            {"term": r[0], "doc_count": r[1], "total_tf": round(r[2], 4)}
            for r in rows
        ]

    def inspect_doc_terms(self, doc_path: str, limit: int = 20) -> list[dict]:
        """Return the top terms for a single document.

        Each entry: ``{"term": str, "chunk_count": int, "total_tf": float}``.
        Computed from the stored chunk text, so it is backend-independent.
        """
        chunk_count: dict[str, int] = {}
        total_tf: dict[str, float] = {}
        for chunk in self.get_chunks_for_doc(doc_path):
            terms = self.analyze(chunk.content)
            if not terms:
                continue
            freq: dict[str, int] = {}
            for t in terms:
                freq[t] = freq.get(t, 0) + 1
            for t, c in freq.items():
                chunk_count[t] = chunk_count.get(t, 0) + 1
                total_tf[t] = total_tf.get(t, 0.0) + c / len(terms)
        ranked = sorted(chunk_count, key=lambda t: (-chunk_count[t], -total_tf[t], t))[:limit]
        return [
            {"term": t, "chunk_count": chunk_count[t], "total_tf": round(total_tf[t], 4)}
            for t in ranked
        ]
