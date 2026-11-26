"""Tests for the LLM reranker and the H1-H4 checks.

No API key, no Ollama, no corpus: the Anthropic client is a fake that
returns a fixed ranking and counts its calls.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rerank as rr  # noqa: E402
import retrieval as rt  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_rerank", HERE / "eval_rerank.py")
er = importlib.util.module_from_spec(_spec)
sys.modules["eval_rerank"] = er
_spec.loader.exec_module(er)


class FakeClient:
    """Answers every call with `ranking`; records the requests."""

    def __init__(self, ranking, stop_reason="end_turn"):
        self.calls = []
        self.ranking, self.stop_reason = ranking, stop_reason
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        text = json.dumps({"ranking": self.ranking})
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
            stop_reason=self.stop_reason,
        )


CANDS = [rr.Candidate(d, f"text of {d}") for d in ["a.md", "b.md", "c.md"]]

# ---------------------------------------------------------------- pieces


def test_parse_ranking_repairs_the_models_answer():
    # repeat, out of range, wrong type and a missing passage
    assert rr.parse_ranking([3, 3, 9, "1", True, 1], 4) == [3, 1, 2, 4]
    assert rr.parse_ranking([], 3) == [1, 2, 3]


def test_representative_chunk_is_the_best_ranked_one_in_any_retriever():
    a0, a1, b0 = rt.Chunk("a.md", 0, "a0"), rt.Chunk("a.md", 1, "a1"), rt.Chunk("b.md", 0, "b0")
    bm25 = [(b0, 3.0), (a1, 2.0)]
    dense = [(a0, 0.9), (b0, 0.8)]
    got = rr.representative_chunks([bm25, dense], ["a.md", "b.md", "z.md"])
    assert [c.text for c in got] == ["a0", "b0", ""]


def test_prompt_numbers_passages_and_shows_paths():
    p = rr.build_prompt("why?", CANDS)
    assert "Question: why?" in p
    assert "[1] a.md\ntext of a.md" in p and "[3] c.md" in p
    assert "all 3 passages" in p


def test_cost_uses_input_and_output_prices():
    assert rr.cost_usd("claude-opus-5-5", 1_000_000, 100_000) == pytest.approx(4.0 + 2.0)


# ---------------------------------------------------------------- reranker


def test_rerank_applies_the_models_order_and_sends_no_temperature(tmp_path):
    client = FakeClient([3, 1, 2])
    res = rr.LLMReranker(client, cache=tmp_path / "c.jsonl").rerank("q", CANDS)
    assert res.order == ["c.md", "a.md", "b.md"]
    assert not res.cached and res.error is None
    call = client.calls[0]
    assert "temperature" not in call
    assert call["output_config"]["format"]["schema"] == rr.SCHEMA


def test_cache_makes_a_rerun_free_and_identical(tmp_path):
    cache = tmp_path / "c.jsonl"
    first = rr.LLMReranker(FakeClient([2, 3, 1]), cache=cache).rerank("q", CANDS)
    client = FakeClient([1, 2, 3])  # would answer differently if called
    again = rr.LLMReranker(client, cache=cache).rerank("q", CANDS)
    assert client.calls == []
    assert again.cached and again.order == first.order
    assert again.latency_s == first.latency_s  # original latency kept for H4


def test_reversed_order_maps_numbers_back_to_documents():
    # shown as [c, b, a]; the model puts [1] (= c.md) last
    res = rr.LLMReranker(FakeClient([2, 3, 1])).rerank("q", CANDS, order="reversed")
    assert res.order == ["b.md", "a.md", "c.md"]


def test_refusal_keeps_rrf_order():
    res = rr.LLMReranker(FakeClient([3, 2, 1], stop_reason="refusal")).rerank("q", CANDS)
    assert res.order == ["a.md", "b.md", "c.md"]
    assert res.error == "refusal"


# ---------------------------------------------------------------- hypotheses


def q(id, bm25, dense, rrf, rerank, cost=0.01, latency=1.0):
    mrr = {s: (1 / p if p else 0.0) for s, p in dict(rrf=rrf, rerank=rerank).items()}
    return {
        "id": id,
        "type": "semantic",
        "first_hit": {"bm25": bm25, "dense": dense, "rrf": rrf, "rerank": rerank},
        "scores": {s: {"MRR": mrr[s]} for s in mrr},
        "rerank": {"cost_eur": cost, "latency_s": latency},
    }


def test_weak_questions_follow_the_first_run_rule():
    rows = [q("w1", 1, 5, 2, 1), q("w2", None, 3, None, 4), q("ok", 2, 3, 1, 1)]
    assert [r["id"] for r in er.weak_questions(rows)] == ["w1", "w2"]


def test_better_counts_finding_a_missing_doc_as_improvement():
    assert er.better(4, None) and er.better(1, 2)
    assert not er.better(2, 2) and not er.better(None, 3)


def test_hypotheses_verdicts():
    rows = [
        q("w1", 1, 5, 2, 1),  # weak, improves
        q("t1", 1, 1, 1, 2, latency=4.0),  # was 1st, drops -> H3 fails
        q("t2", 1, 1, 1, 1),
    ]
    summary = {
        "rrf": {"all": {"MRR": 0.818}},
        "rerank": {"all": {"MRR": 0.9}},
    }
    h = {r["h"]: r for r in er.hypotheses(rows, summary)}
    assert h["H1"]["holds"]
    assert not h["H2"]["holds"] and "1 of 1" in h["H2"]["measured"]
    assert not h["H3"]["holds"] and "t1" in h["H3"]["measured"]
    assert not h["H4"]["holds"]  # p95 latency is 4 s


def test_p95():
    assert er.p95([float(i) for i in range(1, 101)]) == 95.0
    assert er.p95([2.0]) == 2.0
