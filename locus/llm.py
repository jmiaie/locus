"""
Optional LLM query transformations: rewrite, multi-query (RRF merge) and HyDE.

Provider-agnostic: any OpenAI-compatible ``/chat/completions`` endpoint
(OpenAI, OpenRouter, vLLM, llama.cpp server, Ollama's OpenAI API, LiteLLM...).
Standard library only (``urllib``); nothing here runs unless you call it, and
the rest of Locus never makes network calls.

Configuration (``OpenAICompatibleClient.from_env``):

* ``OPENROUTER_API_KEY`` -> base URL ``https://openrouter.ai/api/v1``
* else ``OPENAI_API_KEY`` with ``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``)
* ``LOCUS_LLM_MODEL`` picks the model (defaults: ``openai/gpt-4o-mini`` on
  OpenRouter, ``gpt-4o-mini`` elsewhere); ``LOCUS_LLM_MAX_TOKENS_TOTAL`` caps
  total tokens per client (prompt + completion) as a spend guard.

Usage::

    from locus.llm import OpenAICompatibleClient, llm_retrieve
    client = OpenAICompatibleClient.from_env()
    chunks = llm_retrieve(engine, "why is the L2 slab thicker?", strategy="multi", client=client)
    print(client.usage)          # {'calls': 1, 'prompt_tokens': ..., 'completion_tokens': ...}

Strategies (every one keeps the original query in the RRF merge unless
``include_original=False``, so a bad rewrite cannot remove good hits):

* ``rewrite`` — one rewritten keyword query.
* ``multi``   — *n* alternative phrasings, all retrieved and fused (RRF, k=60).
* ``hyde``    — a short hypothetical answer passage, retrieved as a query
  (Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels").

Calls use temperature 0 and are memoised per client, so repeating a query does
not re-bill; results can still vary across providers and model versions.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from .retrieval.fusion import rrf_fuse

if TYPE_CHECKING:
    from .core import LocusEngine
    from .retrieval.bm25 import ScoredChunk

logger = logging.getLogger(__name__)

__all__ = [
    "LLMError", "LLMBudgetExceeded", "OpenAICompatibleClient", "LLMQueryTransformer",
    "llm_retrieve", "STRATEGIES",
]

STRATEGIES = ("rewrite", "multi", "hyde")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

#: transport(url, headers, body_bytes, timeout) -> parsed JSON response dict
Transport = Callable[[str, dict, bytes, float], dict]


class LLMError(RuntimeError):
    """The endpoint failed or returned something unusable."""


class LLMBudgetExceeded(LLMError):
    """The client's token budget is used up; no further calls are made."""


def _urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> dict:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - URL is configured by the user
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"LLM endpoint returned HTTP {e.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise LLMError(f"LLM request failed: {e}") from None


@dataclass
class OpenAICompatibleClient:
    """Minimal chat-completions client with usage accounting and a token cap."""

    base_url: str
    api_key: str = field(repr=False)
    model: str
    timeout: float = 60.0
    max_total_tokens: Optional[int] = None
    transport: Optional[Transport] = field(default=None, repr=False)
    extra_headers: dict = field(default_factory=dict, repr=False)
    usage: dict = field(default_factory=lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._memo: dict[str, str] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, model: Optional[str] = None, **kwargs) -> "OpenAICompatibleClient":
        """Build a client from ``OPENROUTER_API_KEY`` or ``OPENAI_API_KEY``/``OPENAI_BASE_URL``."""
        cap = os.environ.get("LOCUS_LLM_MAX_TOKENS_TOTAL")
        if cap and "max_total_tokens" not in kwargs:
            kwargs["max_total_tokens"] = int(cap)
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            return cls(base_url=OPENROUTER_BASE_URL, api_key=key,
                       model=model or os.environ.get("LOCUS_LLM_MODEL") or "openai/gpt-4o-mini", **kwargs)
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return cls(base_url=os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL, api_key=key,
                       model=model or os.environ.get("LOCUS_LLM_MODEL") or "gpt-4o-mini", **kwargs)
        raise LLMError("No LLM credentials: set OPENROUTER_API_KEY, or OPENAI_API_KEY (+ OPENAI_BASE_URL)")

    @property
    def total_tokens(self) -> int:
        return self.usage["prompt_tokens"] + self.usage["completion_tokens"]

    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0) -> str:
        """Return the assistant text for *messages* (memoised; counts tokens)."""
        memo_key = json.dumps([self.model, messages, max_tokens, temperature], sort_keys=True)
        with self._lock:
            if memo_key in self._memo:
                return self._memo[memo_key]
            if self.max_total_tokens is not None and self.total_tokens >= self.max_total_tokens:
                raise LLMBudgetExceeded(
                    f"LLM token budget exhausted ({self.total_tokens} >= {self.max_total_tokens})")
        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}",
                   **self.extra_headers}
        data = (self.transport or _urllib_transport)(f"{self.base_url}/chat/completions", headers, body,
                                                    self.timeout)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected LLM response: {str(data)[:300]}") from None
        u = data.get("usage") or {}
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
            self._memo[memo_key] = text
        logger.debug("LLM call %d: %s tokens", self.usage["calls"], u)
        return text


_REWRITE = ("Rewrite the user's question as one concise keyword search query for a lexical (BM25) "
            "search engine over a document collection. Keep key entities, codes and numbers; add "
            "important synonyms. Reply with the query only.")
_MULTI = ("Write {n} different search queries that would each help find documents answering the "
          "user's question in a lexical (BM25) search engine. Vary vocabulary and use synonyms; keep "
          "key entities, codes and numbers. Reply with a JSON array of {n} strings only.")
_HYDE = ("Write a short passage (3-5 sentences) that would plausibly appear in a document answering "
         "the user's question. Use specific domain vocabulary. Reply with the passage only.")


def _clean(text: str) -> str:
    text = text.strip().strip("`").strip()
    text = re.sub(r"^(query|search query)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip()


def _parse_list(text: str, n: int) -> list[str]:
    t = text.strip()
    m = re.search(r"\[.*\]", t, flags=re.S)
    if m:
        try:
            items = json.loads(m.group(0))
            out = [_clean(str(x)) for x in items if str(x).strip()]
            if out:
                return out[:n]
        except ValueError:
            pass
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln) for ln in t.splitlines()]
    return [_clean(ln) for ln in lines if _clean(ln)][:n]


class LLMQueryTransformer:
    """Turns one query into rewritten / multiple / hypothetical-document queries."""

    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    def rewrite(self, query: str) -> str:
        out = _clean(self.client.chat([{"role": "system", "content": _REWRITE},
                                       {"role": "user", "content": query}], max_tokens=64))
        return out or query

    def multi_query(self, query: str, n: int = 3) -> list[str]:
        out = self.client.chat([{"role": "system", "content": _MULTI.format(n=n)},
                                {"role": "user", "content": query}], max_tokens=48 * n + 32)
        return _parse_list(out, n)

    def hyde(self, query: str) -> str:
        out = self.client.chat([{"role": "system", "content": _HYDE},
                                {"role": "user", "content": query}], max_tokens=200)
        return out.strip() or query

    def queries(self, query: str, strategy: str, n: int = 3, include_original: bool = True) -> list[str]:
        """The list of queries to retrieve for *strategy* (original first when included)."""
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        if strategy == "rewrite":
            extra = [self.rewrite(query)]
        elif strategy == "multi":
            extra = self.multi_query(query, n=n)
        else:
            extra = [self.hyde(query)]
        qs = ([query] if include_original else []) + extra
        return list(dict.fromkeys(q for q in qs if q.strip())) or [query]


def llm_retrieve(
    engine: "LocusEngine",
    query: str,
    strategy: str = "multi",
    client: Optional[OpenAICompatibleClient] = None,
    limit: int = 5,
    n: int = 3,
    include_original: bool = True,
    depth: Optional[int] = None,
    **retrieve_kwargs,
) -> list["ScoredChunk"]:
    """Retrieve with LLM-transformed queries and fuse the lists with RRF (k=60).

    Each query is retrieved with ``engine.retrieve(q, limit=depth, **retrieve_kwargs)``
    (default depth ``max(2*limit, 20)``); filters, mode etc. pass through.  With
    one query (``include_original=False`` and ``rewrite``/``hyde``) the list is
    returned as-is.
    """
    client = client or OpenAICompatibleClient.from_env()
    qs = LLMQueryTransformer(client).queries(query, strategy, n=n, include_original=include_original)
    depth = depth or max(2 * limit, 20)
    retrieve_kwargs.setdefault("use_cache", False)
    lists = [engine.retrieve(q, limit=depth, **retrieve_kwargs) for q in qs]
    if len(lists) == 1:
        return lists[0][:limit]
    return rrf_fuse(lists, limit=limit)
