"""Tests for the P1 retrieval harness: metrics, RRF, BM25, chunking, dense cache.

No Ollama and no corpus needed: the dense index gets a fake embedder.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import retrieval as rt  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_retrieval", HERE / "eval_retrieval.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["eval_retrieval"] = ev
_spec.loader.exec_module(ev)


# ---------------------------------------------------------------- metrics


def test_recall_counts_share_of_relevant_in_top_k():
    assert ev.recall_at(["a", "b", "c"], ["a", "z"], 5) == 0.5
    assert ev.recall_at(["x", "y", "a"], ["a"], 2) == 0.0


def test_mrr_uses_first_relevant_position():
    assert ev.reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5
    assert ev.reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_is_one_for_perfect_ranking_and_lower_when_pushed_down():
    assert ev.ndcg_at(["a", "b", "x"], ["a", "b"], 10) == pytest.approx(1.0)
    assert ev.ndcg_at(["x", "a"], ["a"], 10) == pytest.approx(1 / 1.584962500721156)


# ---------------------------------------------------------------- RRF


def test_rrf_rewards_agreement_between_rankings():
    assert rt.rrf([["a", "b"], ["b", "a"], ["b"]])[0] == "b"


def test_rrf_can_bury_a_bm25_top_hit_missing_from_dense():
    """Hypothesis 4: right doc 1st in BM25 but absent from dense; a wrong doc
    2nd and 3rd. 1/61 = 0.0164 < 1/62 + 1/63 = 0.0320, so the wrong doc wins."""
    bm25 = ["right", "wrong"]
    dense = ["d1", "d2", "wrong"]
    fused = rt.rrf([bm25, dense])
    assert fused[0] == "wrong"
    assert ev.reciprocal_rank(fused, ["right"]) < 1.0


# ---------------------------------------------------------------- BM25 and text


def test_tokenize_keeps_identifiers_whole():
    assert rt.tokenize("Set immich_postgres and forwardAuth!") == [
        "set",
        "immich_postgres",
        "and",
        "forwardauth",
    ]


def test_bm25_prefers_rare_exact_identifier():
    chunks = [
        rt.Chunk("a.md", 0, "the database container is configured here"),
        rt.Chunk("b.md", 0, "set immich_postgres password in the env file"),
        rt.Chunk("c.md", 0, "the database and the container and the env"),
    ]
    hits = rt.BM25(chunks).search("immich_postgres password")
    assert hits[0][0].doc == "b.md"
    assert all(score > 0 for _, score in hits)


def test_chunk_splits_at_headings_and_windows_long_sections():
    text = "# One\n" + " ".join(f"w{i}" for i in range(700)) + "\n## Two\nshort section"
    chunks = rt.chunk("d.md", text, words=300, overlap=50)
    assert chunks[-1].text.startswith("## Two")
    first_section = [c for c in chunks if not c.text.startswith("## Two")]
    assert len(first_section) == 3  # windows at 0, 250, 500
    # overlap: the last 50 words of a window start the next one
    assert first_section[0].text.split()[-50:] == first_section[1].text.split()[:50]


def test_to_doc_ranking_dedupes_and_cuts_at_depth():
    hits = [(rt.Chunk(d, 0, d), 1.0) for d in ["a", "a", "b", "c", "b", "d"]]
    assert rt.to_doc_ranking(hits, depth=3) == ["a", "b", "c"]


# ---------------------------------------------------------------- dense


def fake_embedder(calls):
    vocab = ["traefik", "immich", "assistant"]

    def embed(texts):
        calls.append(len(texts))
        return [[float(w in t.lower()) for w in vocab] for t in texts]

    return embed


def test_dense_ranks_by_cosine_and_caches_embeddings(tmp_path):
    chunks = [
        rt.Chunk("t.md", 0, "Traefik routers"),
        rt.Chunk("i.md", 0, "Immich photos"),
        rt.Chunk("h.md", 0, "Home Assistant automations"),
    ]
    cache = tmp_path / "emb.npz"
    calls: list[int] = []
    index = rt.DenseIndex(chunks, fake_embedder(calls), cache)
    assert index.search("where are immich photos")[0][0].doc == "i.md"
    assert calls == [3, 1]  # corpus once, then the query

    calls.clear()
    rt.DenseIndex(chunks, fake_embedder(calls), cache)
    assert calls == []  # second build reads everything from the cache
