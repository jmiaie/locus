"""LocusEngine.stream_retrieve — incremental yield over the existing signals."""

from locus.core import LocusEngine
from locus.retrieval.bm25 import ScoredChunk
from locus.retrieval.streaming import StreamingConfig


def _engine(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    (tmp_path / "auth.md").write_text(
        "# Auth\n\nJWT authentication tokens protect the API.\n\nSee [[ops]]."
    )
    (tmp_path / "ops.md").write_text(
        "# Ops\n\nOperational runbook for the authentication service."
    )
    engine.index(tmp_path)
    return engine


def test_stream_retrieve_yields_unique_scored_chunks(tmp_path):
    engine = _engine(tmp_path)
    config = StreamingConfig(
        adaptive_dropout=False,
        total_budget_ms=60_000,
        signal_timeout_ms=60_000,
    )
    chunks = list(engine.stream_retrieve("authentication", limit=10, config=config))
    assert chunks, "expected at least one streamed chunk"
    assert all(isinstance(c, ScoredChunk) for c in chunks)
    assert len(chunks) <= 10
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # Dropout off and a high limit: the fused retrieve hit is one of the signal hits.
    fused = engine.retrieve("authentication", limit=5)
    assert fused
    streamed_ids = {c.chunk_id for c in chunks}
    assert fused[0].chunk_id in streamed_ids


def test_stream_retrieve_respects_limit(tmp_path):
    engine = _engine(tmp_path)
    config = StreamingConfig(adaptive_dropout=False, total_budget_ms=60_000)
    chunks = list(engine.stream_retrieve("authentication", limit=1, config=config))
    assert len(chunks) == 1


def test_stream_retrieve_dropout_skips_when_budget_already_spent(tmp_path):
    engine = _engine(tmp_path)
    config = StreamingConfig(
        adaptive_dropout=True,
        total_budget_ms=0.0,
        signal_timeout_ms=50.0,
    )
    chunks = list(engine.stream_retrieve("authentication", limit=5, config=config))
    assert chunks == []


def test_stream_retrieve_empty_corpus(tmp_path):
    engine = LocusEngine(store_path=tmp_path / ".locus")
    config = StreamingConfig(adaptive_dropout=False, total_budget_ms=60_000)
    assert list(engine.stream_retrieve("anything", config=config)) == []


def test_stream_retrieve_signals_callback(tmp_path):
    engine = _engine(tmp_path)
    seen: list[str] = []

    def on_signal(name, results, elapsed_ms):
        seen.append(name)
        assert isinstance(results, list)
        assert elapsed_ms >= 0

    config = StreamingConfig(
        adaptive_dropout=False,
        total_budget_ms=60_000,
        on_signal_complete=on_signal,
    )
    list(engine.stream_retrieve("authentication", limit=10, config=config))
    assert "bm25" in seen
    assert seen[0] == "bm25"
