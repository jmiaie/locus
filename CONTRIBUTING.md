# Contributing to Locus

Contributions are welcome — bug reports, feature requests, documentation improvements, and pull requests.

## Development Setup

```bash
git clone https://github.com/jmiaie/locus
cd locus
pip install -e ".[dev]"
pytest tests/ -v
```

All 167 tests should pass. Run the suite before submitting a PR.

## Project Structure

```
locus/
├── memory/         # Corpus, TemporalKG, Chunker, EntityResolver, Extractor
├── retrieval/      # BM25, KGRetriever, LinkWalker, Structural, Recency,
│                   # LinkPopularity, Fusion, Classifier
├── context/        # ContextBulletin, ContextBudget
├── mcp/            # MCP server, tools, prompts, HTTP transport
├── bridge/         # OMPA vault importer
├── core.py         # LocusEngine orchestrator
├── cluster.py      # LocusCluster
├── watcher.py      # LocusWatcher
├── eval.py         # LocusEval (Recall@K, MRR)
├── doctor.py       # LocusDoctor (health checks)
└── export.py       # KGExporter (GraphML, JSONL, DOT)
```

## Code Style

- **No comments by default.** Add one only when the *why* is non-obvious.
- **No docstrings for obvious methods.** Module-level docstrings are fine.
- **No abstractions beyond the task.** If it's not needed now, don't add it.
- **No backwards-compatibility hacks.** Change the code, not the interface.
- **Tests for every new feature.** Add to `tests/test_locus.py`.

## Adding a Retrieval Signal

1. Create `locus/retrieval/my_signal.py` with a class that has `search(...) -> list[ScoredChunk]`
2. Add it to `locus/retrieval/__init__.py`
3. Instantiate in `LocusEngine.__init__()` in `core.py`
4. Add to the `rrf_fuse(...)` call in `retrieve()` with a chosen weight constant
5. Export from `locus/__init__.py` if it's user-facing

## Adding an MCP Tool

1. Add the tool definition to `locus/mcp/tools.py` under `TOOLS`
2. Add a handler branch in `_call_tool()` in `locus/mcp/server.py`
3. Add a corresponding engine method in `locus/core.py`
4. Add tests

## Reporting Issues

Open an issue at [github.com/jmiaie/locus/issues](https://github.com/jmiaie/locus/issues) with:
- Python version
- Locus version (`locus status | python -c "import sys,json; d=json.load(sys.stdin); print(d['version'])"`)
- Minimal reproduction case
