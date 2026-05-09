"""
Locus benchmarking suite.

Modules:
  synthetic  — generate synthetic corpora and ground-truth QA pairs
  ablation   — compare retrieval signal combinations (BM25-only → full 6-signal)
  latency    — measure index and query performance at scale
  baseline   — compare against rank_bm25 (optional dependency)
  report     — format results as tables and JSON
"""

from .synthetic import SyntheticCorpus, QAPair
from .ablation import AblationBench, AblationResult
from .latency import LatencyBench, LatencyResult
from .report import BenchReport

__all__ = [
    "SyntheticCorpus", "QAPair",
    "AblationBench", "AblationResult",
    "LatencyBench", "LatencyResult",
    "BenchReport",
]
