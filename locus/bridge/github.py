"""
GitHubBridge — ingest markdown files from a GitHub repository into Locus.

Uses the GitHub REST API (no external dependencies).  Fetches the file tree
for a given branch, filters by pattern, writes each matching file to a local
cache directory, then calls engine.index() on that directory.
"""

from __future__ import annotations

import fnmatch
import json
import os
import pathlib
import tempfile
import urllib.request
import urllib.error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import LocusEngine


class GitHubBridge:
    """Fetch markdown files from a GitHub repo and index them with Locus."""

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        engine: "LocusEngine",
        repo: str,
        token: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        engine:  A :class:`LocusEngine` instance.
        repo:    ``"owner/repo"`` string.
        token:   Optional GitHub personal access token for private repos or
                 higher rate limits.
        """
        self._engine = engine
        self._repo = repo
        self._token = token or os.environ.get("GITHUB_TOKEN")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        branch: str = "main",
        path: str = "",
        pattern: str = "*.md",
        cache_dir: str | None = None,
    ) -> dict:
        """Fetch and index markdown files from the repository.

        Parameters
        ----------
        branch:    Git branch to read from.
        path:      Sub-directory within the repo (empty = repo root).
        pattern:   Glob pattern to select files (e.g. ``"**/*.md"``).
        cache_dir: Local directory to write fetched files to.  If *None* a
                   temporary directory is created and cleaned up afterwards.

        Returns
        -------
        dict with keys: repo, branch, files_fetched, files_indexed, errors
        """
        own_tmpdir = cache_dir is None
        tmpdir_obj = None

        if own_tmpdir:
            tmpdir_obj = tempfile.TemporaryDirectory(prefix="locus_github_")
            cache_dir = tmpdir_obj.name

        cache_path = pathlib.Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        errors: list[str] = []
        files_fetched: list[str] = []

        try:
            tree = self._get_tree(branch)
            candidates = [
                item for item in tree
                if item.get("type") == "blob"
                and self._matches(item["path"], path, pattern)
            ]

            for item in candidates:
                try:
                    content = self._get_file_content(item["path"], branch)
                    dest = cache_path / item["path"].replace("/", os.sep)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                    files_fetched.append(item["path"])
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{item['path']}: {exc}")

            result: dict = {"indexed": 0, "errors": []}
            if files_fetched:
                result = self._engine.index(str(cache_path), pattern=pattern)

        finally:
            if own_tmpdir and tmpdir_obj is not None:
                tmpdir_obj.cleanup()

        return {
            "repo": self._repo,
            "branch": branch,
            "files_fetched": len(files_fetched),
            "files_indexed": result.get("indexed", 0),
            "errors": errors + result.get("errors", []),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tree(self, branch: str) -> list[dict]:
        """Fetch the full recursive git tree for *branch*."""
        url = f"{self.BASE_URL}/repos/{self._repo}/git/trees/{branch}?recursive=1"
        data = self._request(url)
        return data.get("tree", [])

    def _get_file_content(self, file_path: str, branch: str) -> str:
        """Fetch raw file content via the contents API."""
        url = f"{self.BASE_URL}/repos/{self._repo}/contents/{file_path}?ref={branch}"
        data = self._request(url)
        import base64
        encoded = data.get("content", "")
        return base64.b64decode(encoded).decode("utf-8", errors="replace")

    def _request(self, url: str) -> dict:
        """HTTP GET with optional Bearer auth; raises on non-200."""
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _matches(file_path: str, base_path: str, pattern: str) -> bool:
        """Return True if *file_path* is under *base_path* and matches *pattern*."""
        if base_path and not file_path.startswith(base_path.rstrip("/") + "/"):
            if file_path != base_path:
                return False
        name = pathlib.PurePosixPath(file_path).name
        full = file_path
        return fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(full, pattern)
