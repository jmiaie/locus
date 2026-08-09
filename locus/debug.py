"""
Debug tracing for Locus retrieval pipeline.

Enables detailed logging of each signal's contribution, scoring,
and ranking decisions. Useful for diagnosing retrieval quality issues.
"""

import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .retrieval.bm25 import ScoredChunk


logger = logging.getLogger(__name__)


@dataclass
class SignalTrace:
    """Trace data for a single retrieval signal."""
    
    signal_name: str
    query: str
    start_time: str
    end_time: str
    elapsed_ms: float
    retrieval_limit: int
    results_count: int
    top_results: list[dict[str, Any]]


@dataclass
class QueryTrace:
    """Complete trace for a single retrieve() call."""
    
    query: str
    intent: str
    timestamp: str
    total_elapsed_ms: float
    cache_hit: bool
    signals: list[SignalTrace]
    final_ranking: list[dict[str, Any]]
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)


class DebugTracer:
    """Collects and reports debug trace data."""
    
    def __init__(self, store_path: str | Path = ".locus", enabled: bool = True):
        self.store_path = Path(store_path)
        self.enabled = enabled
        self.traces: list[QueryTrace] = []
        self.max_traces = 100
        
        if self.enabled:
            self.log_dir = self.store_path / "debug_logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug tracing enabled. Logs: {self.log_dir}")
    
    def start_trace(
        self,
        query: str,
        intent: str,
    ) -> "QueryTraceContext":
        """Start a new query trace."""
        if not self.enabled:
            return QueryTraceContext(None)
        return QueryTraceContext(self, query, intent)
    
    def record_signal(
        self,
        trace: "QueryTraceContext",
        signal_name: str,
        results: list[ScoredChunk],
        elapsed_ms: float,
        limit: int = 0,
    ) -> None:
        """Record a signal's results."""
        if not self.enabled or trace.tracer is None:
            return
        
        top_3 = [
            {
                "doc_path": r.doc_path,
                "chunk_id": r.chunk_id,
                "score": round(r.score, 6),
                "provenance": r.provenance,
            }
            for r in results[:3]
        ]
        
        signal_trace = SignalTrace(
            signal_name=signal_name,
            query=trace.query,
            start_time=trace.start_time,
            end_time=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=elapsed_ms,
            retrieval_limit=limit,
            results_count=len(results),
            top_results=top_3,
        )
        trace.signals.append(signal_trace)
    
    def finish_trace(
        self,
        trace: "QueryTraceContext",
        final_results: list[ScoredChunk],
        total_elapsed_ms: float,
        cache_hit: bool = False,
    ) -> "QueryTrace | None":
        """Finalize and store a query trace."""
        if not self.enabled or trace.tracer is None:
            return None
        
        final_ranking = [
            {
                "rank": i + 1,
                "doc_path": r.doc_path,
                "chunk_id": r.chunk_id,
                "score": round(r.score, 6),
                "provenance": r.provenance,
            }
            for i, r in enumerate(final_results[:10])
        ]
        
        query_trace = QueryTrace(
            query=trace.query,
            intent=trace.intent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_elapsed_ms=total_elapsed_ms,
            cache_hit=cache_hit,
            signals=trace.signals,
            final_ranking=final_ranking,
        )
        
        self.traces.append(query_trace)
        if len(self.traces) > self.max_traces:
            self.traces.pop(0)
        
        # Write to disk
        self._write_trace(query_trace)
        
        return query_trace
    
    def _write_trace(self, trace: QueryTrace) -> None:
        """Write trace to disk as JSON."""
        timestamp = trace.timestamp.replace(":", "-").split(".")[0]
        filename = f"{timestamp}_{hash(trace.query) % 10000}.json"
        path = self.log_dir / filename
        
        with open(path, "w") as f:
            f.write(trace.to_json())
    
    def report(self) -> dict:
        """Generate a summary report of collected traces."""
        if not self.traces:
            return {"traces_count": 0, "message": "No traces collected"}
        
        latencies = [t.total_elapsed_ms for t in self.traces]
        signal_counts = {}
        for trace in self.traces:
            for sig in trace.signals:
                signal_counts[sig.signal_name] = signal_counts.get(sig.signal_name, 0) + 1
        
        return {
            "traces_count": len(self.traces),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "signal_usage": signal_counts,
            "log_dir": str(self.log_dir) if self.enabled else None,
        }
    
    def clear(self) -> None:
        """Clear all traces and log files."""
        if not self.enabled:
            return
        self.traces.clear()
        for f in self.log_dir.glob("*.json"):
            f.unlink()


class QueryTraceContext:
    """Context manager for tracing a single query."""
    
    def __init__(self, tracer: "DebugTracer | None", query: str = "", intent: str = ""):
        self.tracer = tracer
        self.query = query
        self.intent = intent
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.signals: list[SignalTrace] = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
