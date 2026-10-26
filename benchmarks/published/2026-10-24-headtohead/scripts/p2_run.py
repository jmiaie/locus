"""Phase 2 head-to-head on a fixed LongMemEval-S subset. usage: p2_run.py SYSTEM [--workers N] [--limit N]
Each question = fresh memory store; ingest the haystack sessions (one call per session), query once,
map retrieved memories -> source session ids, R@5 = any answer session in the top-5 distinct sessions."""
import os, sys, json, time, re, tempfile, asyncio, argparse, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
GW = "http://127.0.0.1:8765"; MODEL = "openai/gpt-4o-mini"
ROOT = Path("/workspace/llmbench"); RES = ROOT / "results"; TMP = ROOT / "tmp"
ap = argparse.ArgumentParser(); ap.add_argument("system"); ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--limit", type=int); ap.add_argument("--subset", default="lme_subset30.json"); ap.add_argument("--max-usd", type=float, default=None); ap.add_argument("--order", default=None)
A = ap.parse_args()
SUB = json.load(open(ROOT / A.subset)); tag = Path(A.subset).stem.replace("lme_subset", "lme")
ROWS = {json.loads(l)["question_id"]: json.loads(l) for l in open("/workspace/stem/data/lme_s_470.jsonl")}
PROVIDER = os.environ.get("P2_PROVIDER", "openrouter")  # provider switch; default = original OpenRouter path, unchanged
if PROVIDER == "gemini":  # Gemini free tier via gateway_gemini.py (127.0.0.1:8766); MiniLM embeddings served by the same gateway
    GW = "http://127.0.0.1:8766"; MODEL = os.environ.get("P2_GEMINI_MODEL", "gemini-3.5-flash-lite")
OUT = RES / (f"{tag}__{A.system}.jsonl" if PROVIDER == "openrouter" else f"{tag}__{A.system}_{PROVIDER}.jsonl")

def dedup_sessions(sids, k=10):
    out = []
    for s in sids:
        if s and s not in out: out.append(s)
        if len(out) == k: break
    return out

def run_mem0(e, wd):
    from mem0 import Memory
    cfg = {"llm": {"provider": "openai", "config": {"model": MODEL, "openai_base_url": f"{GW}/or/mem0/v1", "api_key": "dummy", "temperature": 0}},
           "embedder": {"provider": "huggingface", "config": {"huggingface_base_url": f"{GW}/emb/v1", "model": "all-MiniLM-L6-v2", "embedding_dims": 384}},
           "vector_store": {"provider": "qdrant", "config": {"collection_name": "m", "path": str(Path(wd) / "q"), "embedding_model_dims": 384}},
           "history_db_path": str(Path(wd) / "h.db")}
    m = Memory.from_config(cfg); uid = "u"
    t0 = time.perf_counter(); errs = 0
    for sid, sess in zip(e["haystack_session_ids"], e["haystack_sessions"]):
        msgs = [{"role": t["role"], "content": t["content"]} for t in sess]
        try: m.add(msgs, user_id=uid, metadata={"sid": sid}, infer=True)
        except Exception as ex: errs += 1; print("mem0 add err", type(ex).__name__, str(ex)[:150], flush=True)
    ing = time.perf_counter() - t0
    t = time.perf_counter(); r = m.search(e["question"], top_k=100, filters={"user_id": uid}, threshold=0.0); qms = (time.perf_counter() - t) * 1000
    res = r["results"] if isinstance(r, dict) else r
    sids = [(x.get("metadata") or {}).get("sid") for x in res]
    try: n_mem = len((m.get_all(filters={"user_id": uid}, top_k=100000) or {}).get("results", []))
    except Exception: n_mem = None
    return dedup_sessions(sids), ing, qms, {"n_memories": n_mem, "add_errors": errs, "n_hits": len(res)}

def run_langmem(e, wd):
    from langgraph.store.memory import InMemoryStore
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langmem import create_memory_store_manager
    emb = OpenAIEmbeddings(model="all-MiniLM-L6-v2", base_url=f"{GW}/emb/v1", api_key="dummy", check_embedding_ctx_length=False)
    store = InMemoryStore(index={"dims": 384, "embed": emb})
    llm = ChatOpenAI(model=MODEL, base_url=f"{GW}/or/langmem/v1", api_key="dummy", temperature=0)
    mgr = create_memory_store_manager(llm, namespace=("memories", "u"), store=store)
    owner = {}; t0 = time.perf_counter(); errs = 0
    for sid, sess in zip(e["haystack_session_ids"], e["haystack_sessions"]):
        msgs = [{"role": t["role"], "content": t["content"]} for t in sess]
        before = {it.key for it in store.search(("memories", "u"), limit=100000)}
        try: mgr.invoke({"messages": msgs}, config={"recursion_limit": 100})
        except Exception as ex: errs += 1; print("langmem err", type(ex).__name__, str(ex)[:150], flush=True)
        for it in store.search(("memories", "u"), limit=100000):
            if it.key not in before: owner[it.key] = sid
    ing = time.perf_counter() - t0
    t = time.perf_counter(); hits = store.search(("memories", "u"), query=e["question"], limit=100); qms = (time.perf_counter() - t) * 1000
    return dedup_sessions([owner.get(h.key) for h in hits]), ing, qms, {"n_memories": len(owner), "add_errors": errs, "n_hits": len(hits)}

def run_graphiti(e, wd):
    return asyncio.run(_graphiti(e, wd))
async def _graphiti(e, wd):
    from datetime import datetime, timezone
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.nodes import EpisodeType
    route = "or" if PROVIDER == "openrouter" else "gm"
    lc = LLMConfig(api_key="dummy", model=MODEL, small_model=MODEL, base_url=f"{GW}/{route}/graphiti/v1", temperature=0)
    wd = str(ROOT / ("graphiti_db" if PROVIDER == "openrouter" else f"graphiti_db_{PROVIDER}") / e["question_id"]); Path(wd).mkdir(parents=True, exist_ok=True)
    llm_kw = {}
    if PROVIDER != "openrouter":  # gateway may hold a request for hours until the free daily quota resets: no client timeout/retries
        from openai import AsyncOpenAI
        llm_kw["client"] = AsyncOpenAI(api_key="dummy", base_url=lc.base_url, timeout=None, max_retries=0)
    drv = KuzuDriver(db=str(Path(wd) / "kuzu")); drv._database = "u"   # workaround: 0.31.2 KuzuDriver lacks _database
    from graphiti_core.graph_queries import get_fulltext_indices
    from graphiti_core.driver.driver import GraphProvider
    for q in ["INSTALL fts", "LOAD fts"] + get_fulltext_indices(GraphProvider.KUZU):  # workaround: FTS indexes never created
        await drv.client.execute(q)
    g = Graphiti(graph_driver=drv, llm_client=OpenAIGenericClient(config=lc, **llm_kw),
                 embedder=OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key="dummy", embedding_model="all-MiniLM-L6-v2", embedding_dim=384, base_url=f"{GW}/emb/v1")),
                 cross_encoder=OpenAIRerankerClient(config=lc))
    await g.build_indices_and_constraints()
    ep2sid = {}; t0 = time.perf_counter(); errs = 0
    for sid, date, sess in zip(e["haystack_session_ids"], e["haystack_dates"], e["haystack_sessions"]):
        body = "\n".join(f"{t['role']}: {t['content']}" for t in sess)
        ref = datetime.strptime(date.split(" (")[0] + " " + date.split(") ")[1], "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
        try:
            r = await g.add_episode(name=sid, episode_body=body, source_description="chat session", reference_time=ref, source=EpisodeType.message, group_id="u")
            ep2sid[r.episode.uuid] = sid
        except Exception as ex: errs += 1; print("graphiti err", type(ex).__name__, str(ex)[:200], flush=True)
    ing = time.perf_counter() - t0
    json.dump(ep2sid, open(Path(wd) / "ep2sid.json", "w"))
    t = time.perf_counter(); edges = await g.search(e["question"], group_ids=["u"], num_results=100); qms = (time.perf_counter() - t) * 1000
    sids = [ep2sid.get(u) for ed in edges for u in (ed.episodes or [])]
    # secondary: same hybrid (BM25 + cosine, RRF) edge search with the cosine floor removed (default 0.6 is tuned for OpenAI embeddings, not MiniLM)
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
    cfg = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True); cfg.limit = 100; cfg.edge_config.sim_min_score = 0.0
    t = time.perf_counter(); res = await g.search_(e["question"], config=cfg, group_ids=["u"]); rms = (time.perf_counter() - t) * 1000
    sids2 = [ep2sid.get(u) for ed in res.edges for u in (ed.episodes or [])]
    await g.close()
    return dedup_sessions(sids), ing, qms, {"n_edges_hit": len(edges), "add_errors": errs, "episodes": len(ep2sid),
                                            "relaxed_ranked": dedup_sessions(sids2), "relaxed_ms": rms, "relaxed_edges": len(res.edges)}


def _sess_docs(e):
    return [(sid, "\n".join(f"{t['role']}: {t['content']}" for t in sess)) for sid, sess in zip(e["haystack_session_ids"], e["haystack_sessions"])]

def run_locus(e, wd):
    sys.path.insert(0, "/workspace/scorecard-20261020/repos/locus")
    from locus.core import LocusEngine
    dd = Path(wd) / "docs"; dd.mkdir(); back = {}
    for sid, txt in _sess_docs(e):
        k = re.sub(r"[^\w-]", "_", sid); back[k] = sid; (dd / f"{k}.md").write_text(txt)
    t0 = time.perf_counter(); eng = LocusEngine(store_path=Path(wd) / ".locus"); eng.index(dd); ing = time.perf_counter() - t0
    t = time.perf_counter(); ch = eng.retrieve(e["question"], limit=100, use_cache=False, mode="bm25"); qms = (time.perf_counter() - t) * 1000
    return dedup_sessions([back.get(Path(c.doc_path).stem) for c in ch]), ing, qms, {}

_OMPA_MODEL = None
def run_ompa(e, wd):
    global _OMPA_MODEL
    sys.path.insert(0, "/workspace/scorecard-20261020/repos/ompa")
    from ompa import Ompa
    if _OMPA_MODEL is None:
        import torch; torch.set_num_threads(2)
        from sentence_transformers import SentenceTransformer
        _OMPA_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    t0 = time.perf_counter()
    ao = Ompa(vault_path=str(Path(wd) / "vault"), enable_semantic=True, embedding_backend=_OMPA_MODEL); back = {}
    for sid, txt in _sess_docs(e):
        k = re.sub(r"[^\w-]", "_", sid); back[k] = sid; ao.write(content=txt, file_path=f"work/docs/{k}.md", tags=["bench"])
    ao.rebuild_index(); ing = time.perf_counter() - t0
    t = time.perf_counter(); rs = ao.search(e["question"], limit=10); qms = (time.perf_counter() - t) * 1000
    return dedup_sessions([back.get(Path(r.path).stem) for r in rs]), ing, qms, {}

def run_letta(e, wd):
    """Letta 0.16.8 server: one agent per question; each session turn inserted as an archival passage tagged with its
    session id (Letta does not chunk archival inserts; MiniLM truncates long inputs, so turn-level passages).
    Retrieval = the agent's own archival_memory_search tool, invoked by the agent (LLM) when asked the question;
    we also record Letta's direct archival search API (no LLM) for comparison."""
    import httpx
    L = httpx.Client(base_url="http://127.0.0.1:8283/v1", timeout=600)
    emb = {"embedding_endpoint_type": "openai", "embedding_endpoint": f"{GW}/emb/v1", "embedding_model": "all-MiniLM-L6-v2", "embedding_dim": 384, "embedding_chunk_size": 300}
    llm = {"model": MODEL, "model_endpoint_type": "openai", "model_endpoint": f"{GW}/or/letta/v1", "context_window": 32000, "temperature": 0.0, "max_tokens": 1024}
    r = L.post("/agents/", json={"name": f"lme-{e['question_id']}-{int(time.time())}", "llm_config": llm, "embedding_config": emb,
                                  "memory_blocks": [{"label": "human", "value": "The user."}, {"label": "persona", "value": "You are a helpful assistant with long-term archival memory of past conversations with the user."}],
                                  "include_base_tools": True, "tools": ["archival_memory_search", "archival_memory_insert"]}); r.raise_for_status(); aid = r.json()["id"]
    t0 = time.perf_counter(); pid2sid = {}
    from datetime import datetime, timezone
    for sid, date, sess in zip(e["haystack_session_ids"], e["haystack_dates"], e["haystack_sessions"]):
        ts = datetime.strptime(date.split(" (")[0] + " " + date.split(") ")[1], "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc).isoformat()
        for t in sess:
            rr = L.post(f"/agents/{aid}/archival-memory", json={"text": f"{t['role']}: {t['content']}", "tags": [sid], "created_at": ts}); rr.raise_for_status()
            for p in rr.json(): pid2sid[p["id"]] = sid
    ing = time.perf_counter() - t0
    t = time.perf_counter()
    s = L.get(f"/agents/{aid}/archival-memory/search", params={"query": e["question"], "top_k": 100}); s.raise_for_status()
    direct_ms = (time.perf_counter() - t) * 1000
    direct = dedup_sessions([(x.get("tags") or [None])[0] for x in s.json()["results"]])
    t = time.perf_counter()
    m = L.post(f"/agents/{aid}/messages", json={"messages": [{"role": "user", "content": "Search your archival memory of our past conversations to answer: " + e["question"]}]})
    qms = (time.perf_counter() - t) * 1000; m.raise_for_status()
    sids, calls = [], 0
    for msg in m.json().get("messages", []):
        if msg.get("message_type") == "tool_return_message":
            ret = msg.get("tool_return") or ""
            import ast
            try: items = ast.literal_eval(ret)
            except Exception:
                try: items = json.loads(ret)
                except Exception: items = None
            if isinstance(items, dict): items = items.get("results") or items.get("message") or []
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict): sids.append((it.get("tags") or [None])[0])
        if msg.get("message_type") == "tool_call_message" and "archival" in json.dumps(msg.get("tool_call") or {}): calls += 1
    L.delete(f"/agents/{aid}")
    return dedup_sessions(sids), ing, qms, {"direct_ranked": direct, "direct_ms": direct_ms, "archival_search_calls": calls, "passages": len(pid2sid)}

def worker(qid):
    e = ROWS[qid]
    try:
        with tempfile.TemporaryDirectory(dir=TMP) as wd:
            ranked, ing, qms, extra = globals()["run_" + A.system](e, wd)
        return {"qid": qid, "qtype": e["question_type"], "ranked": ranked, "answers": e["answer_session_ids"], "ms": qms, "ingest_s": round(ing, 1), **extra}
    except Exception as ex:
        traceback.print_exc(); return {"qid": qid, "error": f"{type(ex).__name__}: {str(ex)[:300]}"}

if __name__ == "__main__":
    done = set()
    if OUT.exists(): done = {json.loads(l)["qid"] for l in open(OUT) if "error" not in json.loads(l)}
    todo = [q for q in SUB if q not in done]
    print(A.system, "todo", len(todo), flush=True)
    import sqlite3
    def spent():
        c = sqlite3.connect(str(ROOT / ("gateway.sqlite" if PROVIDER == "openrouter" else f"gateway_{PROVIDER}.sqlite"))); v = c.execute("select coalesce(sum(cost),0) from log where tag=?", (A.system,)).fetchone()[0]; c.close(); return v
    if A.order: todo = [q for q in json.load(open(ROOT / A.order)) if q in set(todo)]
    if A.limit is not None: todo = todo[: max(0, A.limit - len(done))]  # --limit = total target incl. done
    s0 = spent(); print(A.system, "spent so far", round(s0, 4), flush=True)
    with ProcessPoolExecutor(A.workers, max_tasks_per_child=1) as ex:
        for i in range(0, len(todo), A.workers):
            if PROVIDER != "openrouter" and (ROOT / "GEMINI_STOP").exists():
                print(A.system, "STOP: gemini spend guard file present", flush=True); break
            if A.max_usd is not None and spent() >= A.max_usd:
                print(A.system, "STOP: per-system budget reached", round(spent(), 4), flush=True); break
            for row in ex.map(worker, todo[i:i + A.workers]):
                row["cum_usd_system"] = round(spent(), 5)
                if A.system == "graphiti" and row.get("add_errors") and not os.environ.get("P2_KEEP_MODEL_ERRORS"):  # any failed ingest (e.g. gateway 402/stop) invalidates the question -> retry later
                    row = {"qid": row["qid"], "error": f"incomplete ingest: {row['add_errors']} add_episode errors", "cum_usd_system": row["cum_usd_system"]}
                if PROVIDER != "openrouter":
                    row["provider"] = PROVIDER; row["llm_model"] = MODEL
                    if (ROOT / "GEMINI_STOP").exists() and "error" not in row: row = {"qid": row["qid"], "error": "GEMINI_STOP during question (ingest incomplete)"}
                with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")
                print(A.system, row["qid"], row.get("error") or (row["ingest_s"], row["ranked"][:3], row["answers"]), "cum$", row["cum_usd_system"], flush=True)
