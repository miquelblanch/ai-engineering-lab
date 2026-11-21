"""Download the public P1 corpus at pinned commits.

Usage:
    uv run python projects/p1-rag-evals-lab/fetch_corpus.py [--dest data/corpus]

For each source in corpus/sources.toml it does a partial, sparse git fetch of
one commit (only the listed paths are downloaded), copies the documents into
<dest>/<source>/ keeping their relative paths, and writes <dest>/manifest.json
with a sha256 per file. Two people running this get byte-identical corpora,
and the manifest proves it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SOURCES_FILE = HERE / "corpus" / "sources.toml"
DEFAULT_DEST = REPO_ROOT / "data" / "corpus"


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    commit: str
    license: str
    include: tuple[str, ...]
    extensions: tuple[str, ...]


def load_sources(path: Path = SOURCES_FILE) -> list[Source]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    sources = [
        Source(
            name=s["name"],
            repo=s["repo"],
            commit=s["commit"],
            license=s["license"],
            include=tuple(s["include"]),
            extensions=tuple(s["extensions"]),
        )
        for s in data["source"]
    ]
    names = [s.name for s in sources]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate source names in {path}: {names}")
    return sources


def select_files(checkout: Path, source: Source) -> list[Path]:
    """Documents under the included paths, with an allowed extension, sorted."""
    found: set[Path] = set()
    for entry in source.include:
        target = checkout / entry
        if target.is_file():
            candidates = [target]
        elif target.is_dir():
            candidates = [p for p in target.rglob("*") if p.is_file()]
        else:
            raise FileNotFoundError(f"{source.name}: '{entry}' not found at {source.commit[:12]}")
        found.update(p for p in candidates if p.suffix in source.extensions)
    return sorted(found)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def checkout_source(source: Source, workdir: Path) -> Path:
    """Partial clone of one commit, with only the included paths materialised."""
    repo_dir = workdir / source.name
    repo_dir.mkdir()
    _git("init", "-q", cwd=repo_dir)
    _git("remote", "add", "origin", source.repo, cwd=repo_dir)
    _git("sparse-checkout", "set", "--no-cone", *source.include, cwd=repo_dir)
    _git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", source.commit, cwd=repo_dir)
    _git("checkout", "-q", "FETCH_HEAD", cwd=repo_dir)
    return repo_dir


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(dest: Path, sources: list[Source]) -> dict:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    manifest: dict = {"sources": []}

    with tempfile.TemporaryDirectory() as tmp:
        for source in sources:
            checkout = checkout_source(source, Path(tmp))
            files = select_files(checkout, source)
            entries = []
            for f in files:
                rel = f.relative_to(checkout)
                out = dest / source.name / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, out)
                entries.append({"path": f"{source.name}/{rel.as_posix()}", "sha256": sha256(out)})
            manifest["sources"].append(
                {
                    "name": source.name,
                    "repo": source.repo,
                    "commit": source.commit,
                    "license": source.license,
                    "documents": len(entries),
                    "files": entries,
                }
            )
            print(f"{source.name:16} {len(entries):5} documents  ({source.license})")

    total = sum(s["documents"] for s in manifest["sources"])
    manifest["total_documents"] = total
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{'total':16} {total:5} documents  ->  {dest}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    fetch(args.dest, load_sources())


if __name__ == "__main__":
    main()
