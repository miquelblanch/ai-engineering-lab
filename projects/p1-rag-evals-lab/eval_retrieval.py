"""Compare BM25, dense and RRF on the P1 golden set.

Usage:
    uv run python projects/p1-rag-evals-lab/eval_retrieval.py [options]

    --systems bm25,dense,rrf   which retrievers to run (dense needs Ollama)
    --depth 20                 candidates each retriever hands to RRF
    --model bge-m3             Ollama embedding model
    --ollama-url URL           default http://localhost:11434

Metrics, all at document level and averaged over questions:

    recall@5   share of a question's relevant documents found in the top 5
    MRR        1 / position of the first relevant document (0 if not found)
    nDCG@10    like recall@10, but a relevant document counts more the higher
               it appears (gain 1 / log2(position + 1))

Each is reported for identifier questions, semantic questions and all of
them, because an average over all 100 hides exactly the trade-off P1 is
about. It also counts how often RRF ranks the first relevant document lower
than the best single retriever did (hypothesis 4).

The full per-question result goes to runs/ (gitignored); the summary table
is printed in Markdown, ready to paste into the learning log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import retrieval as rt  # noqa: E402

_spec = importlib.util.spec_from_file_location("golden_set", HERE / "golden_set.py")
gs = importlib.util.module_from_spec(_spec)
sys.modules["golden_set"] = gs
_spec.loader.exec_module(gs)

RUNS = rt.REPO_ROOT / "runs"

# ---------------------------------------------------------------- metrics


def recall_at(ranking: list[str], relevant: list[str], k: int) -> float:
    return len(set(ranking[:k]) & set(relevant)) / len(relevant)


def first_hit(ranking: list[str], relevant: list[str]) -> int | None:
    """1-based position of the first relevant document, None if absent."""
    for pos, doc in enumerate(ranking, start=1):
        if doc in relevant:
            return pos
    return None


def reciprocal_rank(ranking: list[str], relevant: list[str]) -> float:
    pos = first_hit(ranking, relevant)
    return 1 / pos if pos else 0.0


def ndcg_at(ranking: list[str], relevant: list[str], k: int) -> float:
    dcg = sum(1 / math.log2(pos + 1) for pos, d in enumerate(ranking[:k], start=1) if d in relevant)
    ideal = sum(1 / math.log2(pos + 1) for pos in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


METRICS = {
    "recall@5": lambda r, rel: recall_at(r, rel, 5),
    "MRR": reciprocal_rank,
    "nDCG@10": lambda r, rel: ndcg_at(r, rel, 10),
}

# ---------------------------------------------------------------- run


def run(
    systems: list[str], depth: int, embedder: rt.Embedder | None, corpus: Path, model: str
) -> dict:
    questions = gs.load()
    docs = rt.load_corpus(corpus)
    chunks = rt.chunk_corpus(docs)
    print(
        f"{len(docs)} documents, {len(chunks)} chunks, {len(questions)} questions", file=sys.stderr
    )

    base: dict[str, object] = {}
    if "bm25" in systems or "rrf" in systems:
        base["bm25"] = rt.BM25(chunks)
    if "dense" in systems or "rrf" in systems:
        if embedder is None:
            raise SystemExit("dense retrieval needs an embedder (is Ollama running?)")
        print("embedding the corpus (cached after the first run)...", file=sys.stderr)
        base["dense"] = rt.DenseIndex(
            chunks, embedder, rt.DEFAULT_CACHE / f"embeddings-{model}.npz"
        )

    results = []
    for q in questions:
        rankings = {
            name: rt.to_doc_ranking(idx.search(q.question), depth) for name, idx in base.items()
        }
        if "rrf" in systems:
            rankings["rrf"] = rt.rrf([rankings["bm25"], rankings["dense"]])
        results.append(
            {
                "id": q.id,
                "type": q.type,
                "relevant": q.relevant_docs,
                "first_hit": {s: first_hit(rankings[s], q.relevant_docs) for s in systems},
                "scores": {
                    s: {m: f(rankings[s], q.relevant_docs) for m, f in METRICS.items()}
                    for s in systems
                },
                "top10": {s: rankings[s][:10] for s in systems},
            }
        )
    return {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "systems": systems,
            "depth": depth,
            "model": model,
            "chunk_words": rt.CHUNK_WORDS,
            "chunk_overlap": rt.CHUNK_OVERLAP,
            "rrf_k": rt.RRF_K,
            "chunks": len(chunks),
        },
        "questions": results,
    }


def summarise(run_data: dict) -> dict:
    """Mean of each metric per system and question type."""
    out: dict = {}
    qs = run_data["questions"]
    for s in run_data["config"]["systems"]:
        out[s] = {}
        for group in ("identifier", "semantic", "all"):
            sel = [q for q in qs if group == "all" or q["type"] == group]
            out[s][group] = {m: sum(q["scores"][s][m] for q in sel) / len(sel) for m in METRICS}
            out[s][group]["n"] = len(sel)
    return out


def rrf_worse_than_best(run_data: dict) -> dict[str, int]:
    """Questions where RRF puts the first relevant doc lower than the best single retriever."""
    counts = {"identifier": 0, "semantic": 0}
    for q in run_data["questions"]:
        fh = q["first_hit"]
        singles = [p for p in (fh.get("bm25"), fh.get("dense")) if p]
        if singles and (fh["rrf"] is None or fh["rrf"] > min(singles)):
            counts[q["type"]] += 1
    return counts


def table(summary: dict) -> str:
    groups = ("identifier", "semantic", "all")
    lines = [
        "| System | " + " | ".join(f"{m} ({g})" for g in groups for m in METRICS) + " |",
        "|---|" + "---:|" * (len(groups) * len(METRICS)),
    ]
    for s, by_group in summary.items():
        cells = [f"{by_group[g][m]:.3f}" for g in groups for m in METRICS]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--systems", default="bm25,dense,rrf")
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--model", default="bge-m3")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--corpus", type=Path, default=rt.DEFAULT_CORPUS)
    args = p.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - {"bm25", "dense", "rrf"}
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)}")
    needs_dense = "dense" in systems or "rrf" in systems
    embedder = rt.ollama_embedder(args.model, args.ollama_url) if needs_dense else None

    data = run(systems, args.depth, embedder, args.corpus, args.model)
    summary = summarise(data)
    data["summary"] = summary

    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"p1-retrieval-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print(table(summary))
    if "rrf" in systems:
        worse = rrf_worse_than_best(data)
        print(
            f"\nRRF ranks the first relevant doc lower than the best single retriever in "
            f"{worse['identifier']} identifier and {worse['semantic']} semantic questions."
        )
    print(f"\nFull results: {out.relative_to(rt.REPO_ROOT)}")


if __name__ == "__main__":
    main()
