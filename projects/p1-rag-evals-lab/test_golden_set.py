"""Checks on the golden set that run in CI, without the corpus."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("golden_set", HERE / "golden_set.py")
gs = importlib.util.module_from_spec(_spec)
sys.modules["golden_set"] = gs  # dataclasses look the module up here
_spec.loader.exec_module(gs)

QUESTIONS = gs.load()


def test_golden_set_is_well_formed():
    assert gs.check_structure(QUESTIONS) == []


def test_size_and_balance():
    assert len(QUESTIONS) == 100
    by_source = Counter(q.source for q in QUESTIONS)
    by_type = Counter(q.type for q in QUESTIONS)
    # Neither a source nor a question type may dominate the averages.
    assert all(n >= 25 for n in by_source.values()), by_source
    assert all(n >= 40 for n in by_type.values()), by_type


def _row(**overrides):
    row = {
        "id": "q001",
        "source": "immich",
        "type": "identifier",
        "question": "What is immich_postgres?",
        "identifiers": ["immich_postgres"],
        "relevant_docs": ["immich/docs/docs/FAQ.mdx"],
        "evidence": "immich_postgres",
        "answer": "The database container.",
    }
    return {**row, **overrides}


def _errors(tmp_path: Path, *rows: dict) -> list[str]:
    f = tmp_path / "golden.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return gs.check_structure(gs.load(f))


def test_valid_row_passes(tmp_path: Path):
    assert _errors(tmp_path, _row()) == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source": "nextcloud"}, "unknown source"),
        ({"relevant_docs": ["traefik/docs/content/index.md"]}, "is not under immich/"),
        ({"identifiers": []}, "no identifiers listed"),
        ({"identifiers": ["DB_URL"]}, "is not in the question"),
        ({"type": "semantic", "identifiers": []}, "semantic question contains"),
        ({"evidence": " "}, "empty evidence"),
    ],
)
def test_bad_rows_are_caught(tmp_path: Path, overrides: dict, message: str):
    errors = _errors(tmp_path, _row(**overrides))
    assert any(message in e for e in errors), errors


def test_duplicates_are_caught(tmp_path: Path):
    errors = _errors(tmp_path, _row(), _row())
    assert any("duplicate id" in e for e in errors)
    assert any("duplicate question" in e for e in errors)


def test_unknown_fields_are_rejected(tmp_path: Path):
    f = tmp_path / "golden.jsonl"
    f.write_text(json.dumps(_row(extra=1)) + "\n")
    with pytest.raises(ValueError, match="fields"):
        gs.load(f)


def test_evidence_is_checked_against_the_corpus(tmp_path: Path):
    corpus = tmp_path / "corpus"
    doc = corpus / "immich" / "docs" / "docs" / "FAQ.mdx"
    doc.parent.mkdir(parents=True)
    doc.write_text("Run it in the\n  immich_postgres   container.")
    manifest = {"sources": [{"files": [{"path": "immich/docs/docs/FAQ.mdx"}]}]}
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    q = gs.load_rows([_row(evidence="in the immich_postgres container")])
    assert gs.check_against_corpus(q, corpus) == []  # whitespace is normalised
    q = gs.load_rows([_row(evidence="in the immich_server container")])
    assert gs.check_against_corpus(q, corpus) == [
        "q001: evidence not found in immich/docs/docs/FAQ.mdx"
    ]
