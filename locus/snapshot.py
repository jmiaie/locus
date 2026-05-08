"""
LocusSnapshot — save and restore a Locus store as a portable .tar.gz archive.

Usage:
    # Save
    from locus.snapshot import LocusSnapshot
    LocusSnapshot.save(engine, "backup_2026-05-08.tar.gz")

    # Restore
    LocusSnapshot.load("backup_2026-05-08.tar.gz", store_path=".locus-restored")
    engine = LocusEngine(store_path=".locus-restored")

Zero external dependencies — uses stdlib tarfile.
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import LocusEngine


class LocusSnapshot:
    """Portable save/restore for a Locus store directory."""

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @staticmethod
    def save(engine: "LocusEngine", output_path: str | Path) -> dict:
        """Archive *engine.store_path* into a compressed tar archive.

        Parameters
        ----------
        engine:       A :class:`LocusEngine` instance.
        output_path:  Destination path for the archive (e.g. ``backup.tar.gz``).
                      The ``.tar.gz`` extension is appended if not present.

        Returns
        -------
        dict with keys: path, store_path, files_archived, size_bytes
        """
        output_path = Path(output_path)
        if not output_path.suffix == ".gz":
            if not str(output_path).endswith(".tar.gz"):
                output_path = output_path.with_suffix(".tar.gz")

        store = Path(engine.store_path)
        if not store.exists():
            return {"error": f"Store path does not exist: {store}"}

        files_archived = 0
        with tarfile.open(str(output_path), "w:gz") as tar:
            for file in store.rglob("*"):
                if file.is_file():
                    # Store paths relative to the store itself so we can
                    # extract to any destination store_path cleanly.
                    arcname = file.relative_to(store)
                    tar.add(str(file), arcname=str(arcname))
                    files_archived += 1

        size = output_path.stat().st_size
        return {
            "path": str(output_path),
            "store_path": str(store),
            "files_archived": files_archived,
            "size_bytes": size,
        }

    # ------------------------------------------------------------------
    # Load / restore
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        snapshot_path: str | Path,
        store_path: str | Path,
        overwrite: bool = False,
    ) -> dict:
        """Extract a snapshot archive to *store_path*.

        Parameters
        ----------
        snapshot_path:  Path to the ``.tar.gz`` archive.
        store_path:     Destination store directory.  Must not already exist
                        unless *overwrite* is ``True``.
        overwrite:      If ``True``, wipe *store_path* before extraction.

        Returns
        -------
        dict with keys: store_path, files_restored
        """
        import shutil

        snapshot_path = Path(snapshot_path)
        store_path = Path(store_path)

        if not snapshot_path.exists():
            return {"error": f"Snapshot not found: {snapshot_path}"}

        if store_path.exists():
            if not overwrite:
                return {"error": f"store_path already exists (use overwrite=True): {store_path}"}
            shutil.rmtree(store_path)

        store_path.parent.mkdir(parents=True, exist_ok=True)

        store_path.mkdir(parents=True, exist_ok=True)

        files_restored = 0
        with tarfile.open(str(snapshot_path), "r:gz") as tar:
            # Security: strip absolute paths and prevent path traversal
            members = []
            for m in tar.getmembers():
                m.name = Path(m.name).as_posix().lstrip("/")
                if ".." in m.name:
                    continue
                members.append(m)
                if m.isfile():
                    files_restored += 1
            # Extract directly into store_path — arcnames are relative to store root
            tar.extractall(str(store_path), members=members)

        return {
            "store_path": str(store_path),
            "files_restored": files_restored,
        }

    # ------------------------------------------------------------------
    # Metadata inspection (without loading)
    # ------------------------------------------------------------------

    @staticmethod
    def inspect(snapshot_path: str | Path) -> dict:
        """Return a manifest of the archive without extracting it."""
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.exists():
            return {"error": f"Snapshot not found: {snapshot_path}"}

        files: list[str] = []
        total_size = 0
        with tarfile.open(str(snapshot_path), "r:gz") as tar:
            for m in tar.getmembers():
                if m.isfile():
                    files.append(m.name)
                    total_size += m.size

        return {
            "snapshot": str(snapshot_path),
            "archive_size_bytes": snapshot_path.stat().st_size,
            "uncompressed_bytes": total_size,
            "file_count": len(files),
            "files": files,
        }
