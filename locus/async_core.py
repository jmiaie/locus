"""
AsyncLocusEngine — async wrapper around the synchronous LocusEngine.

Uses a ThreadPoolExecutor so CPU-bound SQLite operations don't block the
event loop.  All sync methods are still available via __getattr__ pass-through.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class AsyncLocusEngine:
    """Async façade over :class:`LocusEngine`."""

    def __init__(self, store_path: str = ".locus", max_workers: int = 4) -> None:
        from .core import LocusEngine
        self._engine = LocusEngine(store_path=store_path)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="locus")

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, lambda: fn(*args, **kwargs))

    # ------------------------------------------------------------------
    # Async versions of the most-used engine methods
    # ------------------------------------------------------------------

    async def index(self, path: str, pattern: str = "**/*.md") -> dict:
        return await self._run(self._engine.index, path, pattern=pattern)

    async def retrieve(self, query: str, limit: int = 5, **kwargs):
        return await self._run(self._engine.retrieve, query, limit=limit, **kwargs)

    async def prepare_context(self, query: str, **kwargs) -> dict:
        return await self._run(self._engine.prepare_context, query, **kwargs)

    async def add_fact(self, subject: str, predicate: str, obj: str, **kwargs) -> dict:
        return await self._run(self._engine.add_fact, subject, predicate, obj, **kwargs)

    async def query_entity(self, entity: str, **kwargs) -> dict:
        return await self._run(self._engine.query_entity, entity, **kwargs)

    async def forget(self, doc_path: str) -> dict:
        return await self._run(self._engine.forget, doc_path)

    async def sync(self, path: str, pattern: str = "**/*.md") -> dict:
        return await self._run(self._engine.sync, path, pattern=pattern)

    async def status(self) -> dict:
        return await self._run(self._engine.status)

    async def explain(self, chunk_id: str, query: str | None = None) -> dict:
        return await self._run(self._engine.explain, chunk_id, query=query)

    async def reason(self, question: str, **kwargs) -> dict:
        return await self._run(self._engine.reason, question, **kwargs)

    async def find_paths(self, entity_a: str, entity_b: str, **kwargs) -> list:
        return await self._run(self._engine.find_paths, entity_a, entity_b, **kwargs)

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AsyncLocusEngine":
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Sync pass-through for anything not explicitly wrapped
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def __repr__(self) -> str:
        return f"AsyncLocusEngine(store={self._engine.store_path!r})"
