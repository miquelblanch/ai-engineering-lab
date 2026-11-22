"""Load and check the P1 golden set.

Usage:
    uv run python projects/p1-rag-evals-lab/golden_set.py [--corpus data/corpus] [--audit]

The golden set is a JSONL file, one question per line:

    id             "q001"... unique
    source         traefik | immich | home-assistant
    type           identifier: the question hinges on an exact token from the docs
                   (forwardAuth, immich_postgres, purge_keep_days)
                   semantic:   plain-language question, no doc-specific tokens
    question       what a user would type
    identifiers    the exact tokens an identifier question hinges on; [] for semantic
    relevant_docs  corpus paths that answer it; the first one is the primary
    evidence       verbatim quote from the primary doc that answers the question
    answer         short reference answer, used later for faithfulness

Without the corpus, only the structure is checked (that is what CI runs).
With the corpus, every path must be in the manifest and every evidence quote
must appear in its primary document, so a label can't silently point at the
wrong file. `--audit` lists, for identifier questions, other documents that
contain the same identifiers but are not labelled: candidates for a missing
label, to review by hand.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
GOLDEN_FILE = HERE / "golden" / "public.jsonl"
DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus"

SOURCES = {"traefik", "immich", "home-assistant"}
TYPES = {"identifier", "semantic"}
FIELDS = {"id", "source", "type", "question", "identifiers", "relevant_docs", "evidence", "answer"}

# Heuristic for tokens that only make sense to someone who has read the docs
# (snake_case, camelCase, ALL_CAPS, dotted options, YAML tags, dotfiles). Used
# to keep semantic questions free of them; identifier questions list theirs
# explicitly in `identifiers`, which is more precise than any regex.
IDENTIFIER = re.compile(
    r"![a-z_]+"  # !secret, !include_dir_named
    r"|\b[a-z]+_[a-z0-9_]+\b"  # immich_postgres, purge_keep_days
    r"|\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b"  # DB_URL, IMMICH_PORT
    r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b"  # forwardAuth, ipAllowList
    r"|\b[A-Za-z]+\.[A-Za-z.]+\b"  # api.insecure, homeassistant/status
    r"|\.[a-z]+\b"  # .immich, .xmp
)


@dataclass(frozen=True)
class Question:
    id: str
    source: str
    type: str
    question: str
    identifiers: tuple[str, ...]
    relevant_docs: tuple[str, ...]
    evidence: str
    answer: str

    @property
    def primary_doc(self) -> str:
        return self.relevant_docs[0]


def load_rows(rows: list[dict]) -> list[Question]:
    questions = []
    for n, row in enumerate(rows, 1):
        if set(row) != FIELDS:
            raise ValueError(f"row {n}: fields {sorted(row)} != {sorted(FIELDS)}")
        row = {
            **row,
            "identifiers": tuple(row["identifiers"]),
            "relevant_docs": tuple(row["relevant_docs"]),
        }
        questions.append(Question(**row))
    return questions


def load(path: Path = GOLDEN_FILE) -> list[Question]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return load_rows([json.loads(line) for line in lines if line.strip()])


def looks_like_identifiers(text: str) -> set[str]:
    return set(IDENTIFIER.findall(text))


def check_structure(questions: list[Question]) -> list[str]:
    """Problems that can be found without the corpus."""
    errors = []
    seen_ids: set[str] = set()
    seen_q: set[str] = set()
    for q in questions:
        where = f"{q.id}:"
        if not re.fullmatch(r"q\d{3}", q.id):
            errors.append(f"{where} id must look like q001")
        if q.id in seen_ids:
            errors.append(f"{where} duplicate id")
        if q.question.lower() in seen_q:
            errors.append(f"{where} duplicate question")
        seen_ids.add(q.id)
        seen_q.add(q.question.lower())
        if q.source not in SOURCES:
            errors.append(f"{where} unknown source {q.source!r}")
        if q.type not in TYPES:
            errors.append(f"{where} unknown type {q.type!r}")
        if not q.relevant_docs:
            errors.append(f"{where} no relevant_docs")
        if len(set(q.relevant_docs)) != len(q.relevant_docs):
            errors.append(f"{where} repeated doc in relevant_docs")
        for doc in q.relevant_docs:
            if doc.split("/", 1)[0] != q.source:
                errors.append(f"{where} {doc} is not under {q.source}/")
        if not q.evidence.strip() or not q.answer.strip():
            errors.append(f"{where} empty evidence or answer")
        # The identifier/semantic split only means something if the labels are honest.
        if q.type == "identifier":
            if not q.identifiers:
                errors.append(f"{where} identifier question with no identifiers listed")
            for ident in q.identifiers:
                if ident not in q.question:
                    errors.append(f"{where} identifier {ident!r} is not in the question")
        if q.type == "semantic":
            if q.identifiers:
                errors.append(f"{where} semantic question lists identifiers")
            found = looks_like_identifiers(q.question)
            if found:
                errors.append(f"{where} semantic question contains {sorted(found)}")
    return errors


def _normalise(text: str) -> str:
    return " ".join(text.split())


def check_against_corpus(questions: list[Question], corpus: Path) -> list[str]:
    """Every doc exists in the pinned corpus and every quote is in its primary doc."""
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    known = {f["path"] for s in manifest["sources"] for f in s["files"]}
    errors = []
    for q in questions:
        for doc in q.relevant_docs:
            if doc not in known:
                errors.append(f"{q.id}: {doc} is not in the corpus manifest")
        if q.primary_doc in known:
            text = _normalise((corpus / q.primary_doc).read_text(encoding="utf-8"))
            if _normalise(q.evidence) not in text:
                errors.append(f"{q.id}: evidence not found in {q.primary_doc}")
            for ident in q.identifiers:
                if ident not in text:
                    errors.append(f"{q.id}: identifier {ident!r} not in {q.primary_doc}")
    return errors


def audit(questions: list[Question], corpus: Path) -> list[str]:
    """Unlabelled docs that contain all of a question's identifiers. For human review."""
    docs = {
        p.relative_to(corpus).as_posix(): p.read_text(encoding="utf-8")
        for p in corpus.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    lines = []
    for q in questions:
        if q.type != "identifier":
            continue
        others = sorted(
            d
            for d, text in docs.items()
            if d not in q.relevant_docs and all(i in text for i in q.identifiers)
        )
        if others:
            shown = ", ".join(others[:4]) + (f" (+{len(others) - 4})" if len(others) > 4 else "")
            lines.append(f"{q.id} {list(q.identifiers)}: {shown}")
    return lines


def summary(questions: list[Question]) -> str:
    rows = []
    for source in sorted(SOURCES):
        cells = [
            f"{t} {sum(q.source == source and q.type == t for q in questions):3}"
            for t in sorted(TYPES)
        ]
        rows.append(f"{source:16} " + "  ".join(cells))
    rows.append(f"{'total':16} {len(questions)} questions")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden", type=Path, default=GOLDEN_FILE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--audit", action="store_true", help="list possible missing labels")
    args = parser.parse_args()

    questions = load(args.golden)
    errors = check_structure(questions)
    if (args.corpus / "manifest.json").is_file():
        errors += check_against_corpus(questions, args.corpus)
        if args.audit:
            print("\n".join(audit(questions, args.corpus)) or "audit: nothing to review")
    else:
        print(f"corpus not found at {args.corpus}: structure checks only")
    print(summary(questions))
    for e in errors:
        print("ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
