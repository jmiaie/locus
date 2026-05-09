"""
CorpusDiff — compare the current corpus state against the filesystem.

Reports which files are new (not yet indexed), changed (checksum mismatch),
deleted (in corpus but missing from disk), and unchanged.

Useful before a sync() to preview what would actually change.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory.corpus import Corpus

from .memory.corpus import _EXCLUDE


class CorpusDiff:
    """Preview index changes without modifying the corpus."""

    def __init__(self, corpus: "Corpus") -> None:
        self._corpus = corpus

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def diff(self, path: str | Path, pattern: str = "**/*.md") -> dict:
        """Compare *path* against the current corpus.

        Returns a dict with keys:
            new       — files on disk but not yet indexed
            changed   — files indexed but with a different checksum
            deleted   — files indexed but no longer on disk
            unchanged — count of files with matching checksums
        """
        path = Path(path)
        indexed: set[str] = set(self._corpus.list_docs())

        new: list[str] = []
        changed: list[str] = []
        scanned: set[str] = set()

        for f in path.glob(pattern):
            if not f.is_file():
                continue
            if any(ex in f.parts for ex in _EXCLUDE):
                continue

            try:
                rel = str(f.relative_to(path))
            except ValueError:
                rel = str(f)

            scanned.add(rel)

            if rel not in indexed:
                new.append(rel)
            else:
                try:
                    disk_checksum = self._corpus._file_checksum(f)
                    stored_checksum = self._corpus._stored_checksum(rel)
                    if disk_checksum != stored_checksum:
                        changed.append(rel)
                except Exception:
                    changed.append(rel)

        deleted = [d for d in indexed if d not in scanned]
        unchanged = len(indexed) - len(changed) - len(deleted)

        return {
            "new": sorted(new),
            "changed": sorted(changed),
            "deleted": sorted(deleted),
            "unchanged": max(unchanged, 0),
            "total_on_disk": len(scanned),
            "total_indexed": len(indexed),
        }

    def has_changes(self, path: str | Path, pattern: str = "**/*.md") -> bool:
        """Return True if any new, changed, or deleted files exist."""
        result = self.diff(path, pattern=pattern)
        return bool(result["new"] or result["changed"] or result["deleted"])
