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
        root: str | Path | None = None,
    ) -> dict:
        """Extract a snapshot archive to *store_path*.

        Parameters
        ----------
        snapshot_path:  Path to the ``.tar.gz`` archive.
        store_path:     Destination store directory.  Must not already exist
                        unless *overwrite* is ``True``.
        overwrite:      If ``True``, wipe *store_path* before extraction.  Only
                        an empty directory or an existing Locus store is wiped.
        root:           If given, *store_path* must be a relative path that
                        resolves strictly inside *root* (used by the MCP tool,
                        where the caller is a model, not the operator).

        Returns
        -------
        dict with keys: store_path, files_restored
        """
        import shutil

        snapshot_path = Path(snapshot_path)
        target = _resolve_target(store_path, root)
        if isinstance(target, dict):
            return target

        if not snapshot_path.exists():
            return {"error": f"Snapshot not found: {snapshot_path}"}

        if target.exists():
            if not overwrite:
                return {"error": f"store_path already exists (use overwrite=True): {store_path}"}
            if not target.is_dir() or target.is_symlink():
                return {"error": f"store_path is not a directory: {store_path}"}
            if any(target.iterdir()) and not _looks_like_store(target):
                return {"error": f"Refusing to overwrite a directory that is not a Locus store: {store_path}"}
            shutil.rmtree(target)

        target.mkdir(parents=True, exist_ok=True)

        files_restored = 0
        with tarfile.open(str(snapshot_path), "r:gz") as tar:
            members = []
            for m in tar.getmembers():
                if not _safe_member(m, target):
                    return {"error": f"Unsafe archive member rejected: {m.name!r}"}
                members.append(m)
                if m.isfile():
                    files_restored += 1
            if hasattr(tarfile, "data_filter"):
                tar.extractall(str(target), members=members, filter="data")
            else:  # Python without PEP 706 filters; members were vetted above
                tar.extractall(str(target), members=members)

        return {
            "store_path": str(target),
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


_STORE_MARKERS = ("kg.sqlite3", "corpus", "bulletin.sqlite3", "resolver.sqlite3")


def _looks_like_store(path: Path) -> bool:
    return any((path / name).exists() for name in _STORE_MARKERS)


def _resolve_target(store_path: str | Path, root: str | Path | None) -> Path | dict:
    raw = Path(store_path)
    if ".." in raw.parts:
        return {"error": f"store_path must not contain '..': {store_path}"}
    if root is None:
        target = raw.resolve()
        if target == Path(target.anchor) or target == Path.home().resolve():
            return {"error": f"Refusing to restore into {target}"}
        return target
    if raw.is_absolute():
        return {"error": f"store_path must be relative: {store_path}"}
    root_resolved = Path(root).resolve()
    target = (root_resolved / raw).resolve()
    if target == root_resolved or root_resolved not in target.parents:
        return {"error": f"store_path escapes the store root: {store_path}"}
    return target


def _safe_member(member: tarfile.TarInfo, target: Path) -> bool:
    name = Path(member.name)
    if name.is_absolute() or ".." in name.parts or member.name.startswith(("/", "\\")):
        return False
    if not (member.isfile() or member.isdir()):
        return False  # no symlinks, hardlinks or device files in a store snapshot
    dest = (target / name).resolve()
    return dest == target or target in dest.parents
