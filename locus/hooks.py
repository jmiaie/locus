"""
LocusHooks — lightweight event hook system for Locus.

Events fired by the engine:
  pre_index, post_index
  pre_retrieve, post_retrieve
  pre_forget, post_forget
  pre_add_fact, post_add_fact

Handlers are error-isolated: a failing hook logs to stderr but does not
crash the calling operation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

HookFn = Callable[["HookContext"], None]


@dataclass
class HookContext:
    event: str
    engine: Any
    data: dict = field(default_factory=dict)


class LocusHooks:
    """Registry of named event handlers."""

    def __init__(self) -> None:
        self._registry: dict[str, list[HookFn]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on(self, event: str) -> Callable[[HookFn], HookFn]:
        """Decorator to register a handler for *event*."""
        def decorator(fn: HookFn) -> HookFn:
            self.register(event, fn)
            return fn
        return decorator

    def register(self, event: str, fn: HookFn) -> None:
        """Register *fn* for *event*."""
        self._registry.setdefault(event, []).append(fn)

    def unregister(self, event: str, fn: HookFn) -> None:
        """Remove a previously registered handler (no-op if not found)."""
        handlers = self._registry.get(event, [])
        try:
            handlers.remove(fn)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Firing
    # ------------------------------------------------------------------

    def fire(self, event: str, engine: Any, **data: Any) -> None:
        """Fire *event*, invoking each registered handler in registration order.

        Handlers are error-isolated: exceptions are printed to stderr but
        do not propagate.
        """
        ctx = HookContext(event=event, engine=engine, data=data)
        for fn in self._registry.get(event, []):
            try:
                fn(ctx)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[locus.hooks] handler {fn.__name__!r} raised on event {event!r}: {exc}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_hooks(self) -> dict[str, int]:
        """Return a mapping of event → handler count."""
        return {event: len(fns) for event, fns in self._registry.items() if fns}

    def clear(self, event: str | None = None) -> None:
        """Remove all handlers for *event*, or all handlers if *event* is None."""
        if event is None:
            self._registry.clear()
        else:
            self._registry.pop(event, None)

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._registry.values())
        return f"LocusHooks(events={len(self._registry)}, handlers={total})"
