"""Offline tests for the corpus fetcher: no network, no git."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fetch_corpus", HERE / "fetch_corpus.py")
fc = importlib.util.module_from_spec(_spec)
sys.modules["fetch_corpus"] = fc  # dataclasses look the module up here
_spec.loader.exec_module(fc)


def test_sources_are_pinned_and_licensed():
    sources = fc.load_sources()
    assert {s.name for s in sources} == {"traefik", "immich", "home-assistant"}
    for s in sources:
        # A branch name would make the corpus drift under the golden set.
        assert re.fullmatch(r"[0-9a-f]{40}", s.commit), f"{s.name}: commit is not a full SHA"
        assert s.license, f"{s.name}: license missing"
        assert s.include and s.extensions


def test_duplicate_source_names_are_rejected(tmp_path: Path):
    entry = (
        '[[source]]\nname = "a"\nrepo = "r"\ncommit = "c"\nlicense = "MIT"\n'
        'include = ["x"]\nextensions = [".md"]\n'
    )
    f = tmp_path / "sources.toml"
    f.write_text(entry * 2)
    with pytest.raises(ValueError, match="duplicate"):
        fc.load_sources(f)


def _source(include, extensions=(".md",)):
    return fc.Source("s", "repo", "0" * 40, "MIT", tuple(include), tuple(extensions))


def test_select_files_filters_by_path_and_extension(tmp_path: Path):
    (tmp_path / "docs" / "sub").mkdir(parents=True)
    (tmp_path / "docs" / "a.md").write_text("a")
    (tmp_path / "docs" / "sub" / "b.md").write_text("b")
    (tmp_path / "docs" / "image.png").write_text("x")
    (tmp_path / "single.md").write_text("c")
    (tmp_path / "outside.md").write_text("d")

    files = fc.select_files(tmp_path, _source(["docs", "single.md"]))
    rel = [f.relative_to(tmp_path).as_posix() for f in files]
    assert rel == ["docs/a.md", "docs/sub/b.md", "single.md"]


def test_select_files_fails_loudly_on_missing_path(tmp_path: Path):
    # A path that vanished upstream must stop the fetch, not shrink the corpus quietly.
    with pytest.raises(FileNotFoundError, match="not found"):
        fc.select_files(tmp_path, _source(["docs/gone"]))


def test_downloaded_corpus_is_never_committed():
    root = HERE.parent.parent
    probe = "data/corpus/traefik/x.md"
    result = subprocess.run(["git", "check-ignore", "-q", probe], cwd=root)
    assert result.returncode == 0, "data/corpus/ must stay gitignored"
