"""LLM query-transformation plugin, tested with a mock OpenAI-compatible transport (no network)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from locus.core import LocusEngine
from locus.llm import (LLMBudgetExceeded, LLMError, LLMQueryTransformer, OpenAICompatibleClient,
                       _parse_list, llm_retrieve)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "examples" / "construction" / "docs"


class MockTransport:
    """Records requests; answers by system-prompt kind."""

    def __init__(self, replies=None, usage=(20, 10)):
        self.calls = []
        self.replies = replies or {}
        self.usage = usage

    def __call__(self, url, headers, body, timeout):
        req = json.loads(body)
        self.calls.append({"url": url, "headers": headers, "body": req})
        system = req["messages"][0]["content"]
        kind = "multi" if "JSON array" in system else "hyde" if "passage" in system else "rewrite"
        text = self.replies.get(kind, {"rewrite": "slab thickness vibration MRI",
                                       "multi": '["slab thickness change order", "MRI vibration criterion", "RFI-011"]',
                                       "hyde": "The Level 2 slab was thickened to 7.5 inches to meet the MRI vibration criterion."}[kind])
        return {"choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": self.usage[0], "completion_tokens": self.usage[1]}}


def client(transport, **kw):
    return OpenAICompatibleClient(base_url="https://llm.example/v1/", api_key="sk-test", model="m", transport=transport, **kw)


@pytest.fixture(scope="module")
def eng(tmp_path_factory):
    e = LocusEngine(store_path=tmp_path_factory.mktemp("llm") / ".locus")
    e.index(DOCS)
    return e


def test_request_shape_and_usage():
    t = MockTransport()
    c = client(t)
    out = LLMQueryTransformer(c).rewrite("why is the slab thicker?")
    assert out == "slab thickness vibration MRI"
    call = t.calls[0]
    assert call["url"] == "https://llm.example/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["body"]["model"] == "m" and call["body"]["temperature"] == 0.0
    assert c.usage == {"calls": 1, "prompt_tokens": 20, "completion_tokens": 10}
    assert "sk-test" not in repr(c)


def test_memoised_calls_are_not_rebilled():
    t = MockTransport()
    c = client(t)
    tr = LLMQueryTransformer(c)
    tr.hyde("q")
    tr.hyde("q")
    assert len(t.calls) == 1 and c.usage["calls"] == 1


def test_token_budget_guard():
    t = MockTransport(usage=(60, 50))
    c = client(t, max_total_tokens=100)
    tr = LLMQueryTransformer(c)
    tr.rewrite("a")
    with pytest.raises(LLMBudgetExceeded):
        tr.rewrite("b")
    assert len(t.calls) == 1


@pytest.mark.parametrize("text,want", [
    ('["a b", "c"]', ["a b", "c"]),
    ('Sure:\n```json\n["x", "y", "z", "w"]\n```', ["x", "y", "z"]),
    ("1. first query\n2) second\n- third", ["first query", "second", "third"]),
])
def test_parse_list(text, want):
    assert _parse_list(text, 3) == want


def test_queries_include_original_and_dedupe():
    tr = LLMQueryTransformer(client(MockTransport(replies={"rewrite": "Query: same"})))
    assert tr.queries("same", "rewrite") == ["same"]
    assert tr.queries("orig", "multi", n=3)[0] == "orig"
    assert tr.queries("orig", "hyde", include_original=False)[0].startswith("The Level 2 slab")
    with pytest.raises(ValueError):
        tr.queries("q", "telepathy")


def test_llm_retrieve_rrf_merge(eng):
    t = MockTransport()
    for strategy in ("rewrite", "multi", "hyde"):
        out = llm_retrieve(eng, "Why was the Level 2 slab thickened?", strategy=strategy, client=client(t), limit=5)
        assert out and len(out) <= 5
        assert len({c.chunk_id for c in out}) == len(out)
    # multi: the RRF merge contains hits for the generated queries
    out = llm_retrieve(eng, "Why was the Level 2 slab thickened?", strategy="multi", client=client(MockTransport()), limit=5)
    assert any("RFI-011" in c.doc_path or "CO-002" in c.doc_path for c in out)


def test_llm_retrieve_passes_filters(eng):
    out = llm_retrieve(eng, "slab vibration", strategy="multi", client=client(MockTransport()), limit=5,
                       filters={"type": "rfi"})
    assert out and all(c.doc_path.startswith("rfis/") for c in out)


def test_transport_errors_surface():
    def boom(*a):
        raise LLMError("HTTP 401")
    with pytest.raises(LLMError):
        LLMQueryTransformer(client(boom)).rewrite("x")
    with pytest.raises(LLMError, match="Unexpected"):
        LLMQueryTransformer(client(lambda *a: {"nope": 1})).rewrite("x")


def test_from_env(monkeypatch):
    for v in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "LOCUS_LLM_MODEL", "LOCUS_LLM_MAX_TOKENS_TOTAL"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(LLMError, match="No LLM credentials"):
        OpenAICompatibleClient.from_env()
    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    c = OpenAICompatibleClient.from_env()
    assert c.base_url == "http://localhost:8000/v1" and c.model == "gpt-4o-mini"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k2")
    monkeypatch.setenv("LOCUS_LLM_MAX_TOKENS_TOTAL", "5000")
    c = OpenAICompatibleClient.from_env()
    assert c.base_url == "https://openrouter.ai/api/v1" and c.api_key == "k2"
    assert c.model == "openai/gpt-4o-mini" and c.max_total_tokens == 5000


def test_cli_llm_without_credentials_fails_cleanly(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("OPENROUTER_API_KEY", "OPENAI_API_KEY")}
    env["PYTHONPATH"] = str(ROOT)
    r = subprocess.run([sys.executable, "-m", "locus", "--store", str(tmp_path / "s"), "retrieve", "x", "--llm", "multi"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "No LLM credentials" in r.stderr


def test_core_never_imports_network_modules():
    code = "import sys, locus, locus.core; print('urllib.request' in sys.modules, 'locus.llm' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT)}).stdout.split()
    assert out[1] == "False"
