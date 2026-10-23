"""Directory indexing does not follow symlinks by default."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from locus.core import LocusEngine
from locus.memory.corpus import iter_index_files
from locus.watcher import LocusWatcher

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not hasattr(os, "symlink") or sys.platform == "win32",
                                reason="symlinks unavailable")


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("# A\n\nalpha inside document")
    (root / "sub" / "b.md").write_text("# B\n\nbeta inside document")
    outside = tmp_path / "outside"
    (outside / "dir").mkdir(parents=True)
    (outside / "secret.md").write_text("# Secret\n\nzeppelin outside secret")
    (outside / "dir" / "c.md").write_text("# C\n\nzeppelin outside dir")
    (root / "link_out.md").symlink_to(outside / "secret.md")
    (root / "linkdir").symlink_to(outside / "dir", target_is_directory=True)
    (root / "link_in.md").symlink_to(root / "sub" / "b.md")
    return root


def names(files, root):
    return sorted(str(f.relative_to(root)) for f in files)


def test_default_skips_all_symlinks(tree, monkeypatch):
    monkeypatch.delenv("LOCUS_FOLLOW_SYMLINKS", raising=False)
    assert names(iter_index_files(tree), tree) == ["a.md", "sub/b.md"]


def test_inside_mode(tree):
    assert names(iter_index_files(tree, follow_symlinks="inside"), tree) == ["a.md", "link_in.md", "sub/b.md"]


def test_follow_all(tree):
    got = names(iter_index_files(tree, follow_symlinks=True), tree)
    assert "link_out.md" in got and "link_in.md" in got


def test_env_and_invalid(tree, monkeypatch):
    monkeypatch.setenv("LOCUS_FOLLOW_SYMLINKS", "inside")
    assert "link_in.md" in names(iter_index_files(tree), tree)
    with pytest.raises(ValueError):
        iter_index_files(tree, follow_symlinks="sometimes")


def test_engine_index_never_reads_outside(tree, tmp_path, monkeypatch):
    monkeypatch.delenv("LOCUS_FOLLOW_SYMLINKS", raising=False)
    eng = LocusEngine(store_path=tmp_path / ".locus")
    eng.index(tree)
    assert sorted(eng.corpus.list_docs()) == ["a.md", "sub/b.md"]
    assert eng.retrieve("zeppelin", use_cache=False) == []


def test_watcher_respects_policy(tree, tmp_path, monkeypatch):
    monkeypatch.delenv("LOCUS_FOLLOW_SYMLINKS", raising=False)
    eng = LocusEngine(store_path=tmp_path / ".w")
    w = LocusWatcher(eng, tree)
    w._cycle()
    assert sorted(eng.corpus.list_docs()) == ["a.md", "sub/b.md"]


def test_cli_follow_symlinks_flag(tree, tmp_path, monkeypatch):
    env = {k: v for k, v in os.environ.items() if k != "LOCUS_FOLLOW_SYMLINKS"}
    env["PYTHONPATH"] = str(ROOT)
    store = tmp_path / ".c"
    subprocess.run([sys.executable, "-m", "locus", "--store", str(store), "index", str(tree),
                    "--follow-symlinks", "inside"], check=True, capture_output=True, env=env)
    assert "link_in.md" in LocusEngine(store_path=store).corpus.list_docs()
    assert "link_out.md" not in LocusEngine(store_path=store).corpus.list_docs()
