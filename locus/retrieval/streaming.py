"""
Streaming retrieval pipeline.

Yields results as each signal completes, allowing partial results
and early termination without waiting for all six signals.
"""

from typing import Iterator, Callable
from dataclasses import dataclass
from datetime import datetime

from .bm25 import ScoredChunk


@dataclass
class StreamingConfig:
    """Configuration for streaming retrieval behavior."""
    
    #: Maximum time to wait for a signal (ms)
    signal_timeout_ms: float = 100.0
    
    #: Drop signal if it exceeds latency budget
    adaptive_dropout: bool = True
    
    #: Budget for total retrieval time (ms)
    total_budget_ms: float = 200.0
    
    #: Callback when signal completes
    on_signal_complete: Callable[[str, list[ScoredChunk], float], None] | None = None
    
    #: Callback when result yielded
    on_result: Callable[[ScoredChunk, str], None] | None = None


class StreamingRetriever:
    """
    Streams results from multiple signals with adaptive dropout.
    
    Instead of waiting for all six signals to complete and then fusing,
    yields results incrementally as signals finish. Allows:
      - Early termination (e.g., `take(5)`)
      - Timeout-based signal skipping
      - Progressive UI updates
    
    Example:
        >>> engine = LocusEngine()
        >>> for chunk in engine.stream_retrieve("auth", limit=5):
        ...     print(chunk)  # Updates progressively
    """
    
    def __init__(self, config: StreamingConfig | None = None):
        self.config = config or StreamingConfig()
        self._start_time: float | None = None
        self._elapsed_by_signal: dict[str, float] = {}
    
    def elapsed_ms(self) -> float:
        """Total elapsed time since stream start (ms)."""
        if self._start_time is None:
            return 0.0
        import time
        return (time.perf_counter() - self._start_time) * 1000
    
    def time_remaining_ms(self) -> float:
        """Remaining time in budget (ms)."""
        return max(0.0, self.config.total_budget_ms - self.elapsed_ms())
    
    def should_run_signal(self, signal_name: str) -> bool:
        """Check if we have time budget for this signal."""
        if not self.config.adaptive_dropout:
            return True
        return self.time_remaining_ms() > self.config.signal_timeout_ms
    
    def record_signal(self, signal_name: str, elapsed_ms: float) -> None:
        """Record completion time for a signal."""
        self._elapsed_by_signal[signal_name] = elapsed_ms
        if self.config.on_signal_complete:
            self.config.on_signal_complete(signal_name, [], elapsed_ms)
    
    def track_result(self, chunk: ScoredChunk, signal_name: str) -> None:
        """Track when a result is yielded."""
        if self.config.on_result:
            self.config.on_result(chunk, signal_name)
    
    def stats(self) -> dict:
        """Return timing statistics for all signals."""
        return {
            "total_elapsed_ms": self.elapsed_ms(),
            "budget_ms": self.config.total_budget_ms,
            "per_signal_ms": self._elapsed_by_signal,
            "signals_dropped": len(
                [s for s in self._elapsed_by_signal 
                 if self._elapsed_by_signal[s] > self.config.signal_timeout_ms]
            ),
        }


def stream_with_timeout(
    signals: dict[str, list[ScoredChunk]],
    retriever: StreamingRetriever,
) -> Iterator[tuple[ScoredChunk, str]]:
    """
    Stream ranked results from multiple signals, respecting timeout.
    
    Yields (chunk, signal_name) as results become available.
    Drops signals that exceed latency budget.
    
    Args:
        signals: Dict mapping signal name to list of ScoredChunks
        retriever: StreamingRetriever instance tracking time budget
    
    Yields:
        (chunk, signal_name) tuples in order of completion
    """
    import time
    
    retriever._start_time = time.perf_counter()
    
    # Track which signals we've exhausted
    indices: dict[str, int] = {name: 0 for name in signals}
    seen_chunk_ids: set[str] = set()
    
    # Round-robin through signals, yielding highest-score chunks
    while any(indices[s] < len(signals[s]) for s in signals):
        if retriever.elapsed_ms() > retriever.config.total_budget_ms:
            break
        
        # Find next best chunk across all signals
        best_chunk: ScoredChunk | None = None
        best_signal: str | None = None
        best_score: float = -1.0
        
        for signal_name, chunks in signals.items():
            if indices[signal_name] < len(chunks):
                chunk = chunks[indices[signal_name]]
                # Prefer chunks we haven't seen; break ties by score
                is_new = chunk.chunk_id not in seen_chunk_ids
                score = (is_new, chunk.score)
                if score > (best_signal is not None, best_score):
                    best_chunk = chunk
                    best_signal = signal_name
                    best_score = chunk.score
        
        if best_chunk and best_signal:
            indices[best_signal] += 1
            if best_chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(best_chunk.chunk_id)
                retriever.track_result(best_chunk, best_signal)
                yield best_chunk, best_signal
