"""On-prem packaging: token handling, CLI env defaults, Dockerfile/compose policy, construction demo."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from locus.mcp.http_server import make_server, resolve_token

ROOT = Path(__file__).resolve().parent.parent


# --- token resolution --------------------------------------------------------

def test_resolve_token_from_file_strips_newline(tmp_path):
    f = tmp_path / "token"
    f.write_text("abc123\n")
    assert resolve_token(None, str(f)) == "abc123"


def test_resolve_token_plain_and_none():
    assert resolve_token("t0k", None) == "t0k"
    assert resolve_token(None, None) is None
    assert resolve_token("", None) is None


def test_resolve_token_rejects_both_and_empty(tmp_path):
    f = tmp_path / "token"
    f.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        resolve_token(None, str(f))
    with pytest.raises(ValueError, match="not both"):
        resolve_token("x", str(f))
    with pytest.raises(ValueError, match="Cannot read"):
        resolve_token(None, str(tmp_path / "missing"))


def test_token_file_satisfies_non_loopback_policy(tmp_path):
    from locus.core import LocusEngine
    f = tmp_path / "token"
    f.write_text("s3cret-from-file")
    eng = LocusEngine(store_path=tmp_path / ".locus")
    with pytest.raises(ValueError):
        make_server(eng, host="0.0.0.0", port=0, token=resolve_token(None, None))
    srv = make_server(eng, host="0.0.0.0", port=0, token=resolve_token(None, str(f)))
    srv.server_close()


def _cli(args, env_extra, tmp_path):
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCUS_")}
    env.update(env_extra)
    return subprocess.run([sys.executable, "-m", "locus", *args], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=60)


def test_cli_serve_env_non_loopback_without_token_refuses(tmp_path):
    r = _cli(["serve"], {"LOCUS_HTTP_HOST": "0.0.0.0", "LOCUS_HTTP_PORT": "0",
                         "LOCUS_STORE": str(tmp_path / ".locus")}, tmp_path)
    assert r.returncode == 2
    assert "Refusing to bind" in r.stderr


def test_cli_serve_bad_token_file_is_an_error(tmp_path):
    r = _cli(["serve", "--token-file", str(tmp_path / "nope")], {}, tmp_path)
    assert r.returncode == 2
    assert "Cannot read token file" in r.stderr


def test_cli_store_env(tmp_path):
    store = tmp_path / "custom-store"
    r = _cli(["status"], {"LOCUS_STORE": str(store)}, tmp_path)
    assert r.returncode == 0, r.stderr
    assert store.exists()


# --- container policy (static checks; the image itself is tested by
#     scripts/container-smoke-test.sh, which needs docker or podman) -----------

def test_dockerfile_policy():
    df = (ROOT / "Dockerfile").read_text()
    assert re.search(r"^ARG BASE=\S+@sha256:[0-9a-f]{64}$", df, re.M), "base image must be pinned by digest"
    assert re.search(r"^USER \$\{LOCUS_UID\}", df, re.M) and "ARG LOCUS_UID=10001" in df
    assert "HEALTHCHECK" in df
    assert "LOCUS_HTTP_HOST=127.0.0.1" in df
    assert "--no-index" in df
    assert "--token" not in df.split("ENTRYPOINT", 1)[1], "no token on the command line"


def test_compose_policy():
    dc = (ROOT / "deploy" / "docker-compose.yml").read_text()
    assert '"127.0.0.1:7391:7391"' in dc
    assert "read_only: true" in dc
    assert "LOCUS_HTTP_TOKEN_FILE: /run/secrets/locus_token" in dc
    assert "cap_drop: [ALL]" in dc
    assert "no-new-privileges:true" in dc


def test_dockerignore_only_ships_the_wheel():
    lines = [l for l in (ROOT / ".dockerignore").read_text().splitlines() if l and not l.startswith("#")]
    assert lines == ["*", "!dist/*.whl"]


# --- construction demo ---------------------------------------------------------

def test_construction_demo_eval():
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "construction" / "demo.py"), "--eval"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    s = json.loads(r.stdout)
    assert s["all"]["n"] == 21
    # Floors, not targets: the set is small and synthetic.
    assert s["all"]["R@5"] >= 0.9
    pit = s["point_in_time"]
    assert pit["n"] == 6 and pit["R@5"] == 1.0
    assert pit["R@1"] > pit["R@1_without_as_of_view"]


def test_construction_docs_are_labelled_synthetic():
    docs = sorted((ROOT / "examples" / "construction" / "docs").rglob("*.md"))
    assert len(docs) >= 20
    for d in docs:
        assert "SYNTHETIC DEMO DOCUMENT" in d.read_text(), d


def test_construction_eval_paths_exist():
    base = ROOT / "examples" / "construction"
    qs = json.loads((base / "eval_questions.json").read_text())["questions"]
    assert 15 <= len(qs) <= 25
    for q in qs:
        for p in q["expected_docs"]:
            assert (base / "docs" / p).is_file(), p
    for f in json.loads((base / "facts.json").read_text())["facts"]:
        assert (base / "docs" / f["source"]).is_file(), f["source"]
