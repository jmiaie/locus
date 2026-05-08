"""
LocusWatcher — poll-based file watcher that auto-reindexes changed documents.

Uses the existing checksum dedup so unchanged files cost only one SHA-256
hash. Detects additions, modifications, and deletions.  Runs either
blocking (foreground) or as a daemon thread (background).

Zero external dependencies — stdlib threading + pathlib only.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_EXCLUDE = {".locus", ".palace", ".git", ".obsidian", "__pycache__", "node_modules"}


class LocusWatcher:
    """
    Watches a directory and keeps a LocusEngine in sync.

    Usage (blocking):
        watcher = LocusWatcher(engine, "./docs")
        watcher.start()          # blocks; Ctrl-C to stop

    Usage (background):
        watcher = LocusWatcher(engine, "./docs", interval=10)
        watcher.start(background=True)
        # ... do other work ...
        watcher.stop()
    """

    def __init__(
        self,
        engine,
        watch_dir: str | Path,
        pattern: str = "**/*.md",
        interval: float = 5.0,
        on_change: Callable[[str, str], None] = None,
    ):
        """
        Args:
            engine:     LocusEngine to keep in sync.
            watch_dir:  Root directory to watch.
            pattern:    Glob pattern for files to watch.
            interval:   Poll interval in seconds.
            on_change:  Optional callback(doc_path, event) where event is
                        "added", "updated", or "deleted".
        """
        from .core import LocusEngine  # local import avoids circular at module level
        self.engine: LocusEngine = engine
        self.watch_dir = Path(watch_dir)
        self.pattern = pattern
        self.interval = interval
        self.on_change = on_change

        self._running = False
        self._thread: threading.Thread | None = None
        self._cycle_count = 0
        self._total_added = 0
        self._total_deleted = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, background: bool = False) -> None:
        """
        Start watching.

        background=False: blocks until stop() is called or KeyboardInterrupt.
        background=True:  runs as a daemon thread; returns immediately.
        """
        self._running = True
        logger.info("Locus watcher started — watching %s (interval=%ss)", self.watch_dir, self.interval)
        if background:
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="locus-watcher"
            )
            self._thread.start()
        else:
            try:
                self._loop()
            except KeyboardInterrupt:
                logger.info("Watcher stopped by user")
            finally:
                self._running = False

    def stop(self) -> None:
        """Signal the watcher to stop after the current cycle."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval + 1)

    def stats(self) -> dict:
        return {
            "watch_dir": str(self.watch_dir),
            "pattern": self.pattern,
            "interval_s": self.interval,
            "running": self._running,
            "cycles": self._cycle_count,
            "total_added_or_updated": self._total_added,
            "total_deleted": self._total_deleted,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            try:
                added, deleted = self._cycle()
                if added or deleted:
                    logger.info(
                        "Watch cycle %d: +%d chunks, -%d docs",
                        self._cycle_count, added, deleted,
                    )
            except Exception as e:
                logger.warning("Watch cycle error: %s", e)
            self._cycle_count += 1
            time.sleep(self.interval)

    def _cycle(self) -> tuple[int, int]:
        """One watch cycle. Returns (chunks_added, docs_deleted)."""
        added_total = 0

        # --- Additions and updates ---
        for f in self.watch_dir.glob(self.pattern):
            if any(ex in f.parts for ex in _EXCLUDE):
                continue
            n = self.engine.corpus.add_file(f, base_path=self.watch_dir)
            if n > 0:
                doc_id = str(f.relative_to(self.watch_dir)).replace("\\", "/")
                self.engine.kg.populate_from_file(f, base_path=self.watch_dir)
                added_total += n
                self._total_added += n
                logger.debug("Indexed/updated: %s (%d chunks)", doc_id, n)
                if self.on_change:
                    self.on_change(doc_id, "updated")

        # --- Deletions ---
        indexed = set(self.engine.corpus.list_docs())
        current = {
            str(f.relative_to(self.watch_dir)).replace("\\", "/")
            for f in self.watch_dir.glob(self.pattern)
            if not any(ex in f.parts for ex in _EXCLUDE)
        }
        deleted_total = 0
        for removed in indexed - current:
            self.engine.corpus.remove_file(removed)
            deleted_total += 1
            self._total_deleted += 1
            logger.debug("Removed deleted doc: %s", removed)
            if self.on_change:
                self.on_change(removed, "deleted")

        return added_total, deleted_total
