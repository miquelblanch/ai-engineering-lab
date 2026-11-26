"""Rerank the top 20 RRF documents with Claude and test reranker hypotheses H1-H4.

Usage:
    uv run python projects/p1-rag-evals-lab/eval_rerank.py [options]

    --estimate            count input tokens (free endpoint) and print the expected
                          cost of a full run; makes no paid call
    --limit N             only the first N questions (smoke test)
    --model ID            default claude-opus-5-5 (see rerank.PRICES_USD)
    --effort LEVEL        low | medium | high (default low)
    --order rrf|reversed  order the passages are shown in; "reversed" measures
                          position bias against an "rrf" run
    --eur-per-usd RATE    exchange rate used for the € figures (default 0.86)
    --no-cache            call the API even for prompts already answered

Needs Ollama with bge-m3 (for RRF, as in eval_retrieval.py) and an Anthropic
API key in ANTHROPIC_API_KEY.

Hypotheses (README, written 2026-11-26 before any run):

    H1  MRR (all questions) goes from 0.818 to >= 0.88
    H2  at least 7 of the weak questions (RRF below the best single retriever,
        13 in the first run) move their first relevant document up
    H3  no question that RRF had at position 1 moves down
    H4  cost < 0.50 EUR per question (mean) and latency < 3 s per question (p95)

Per-question results, with tokens, latency and cost of every call, go to
runs/ (gitignored); the summary prints in Markdown.
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

import rerank as rr  # noqa: E402
import retrieval as rt  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_retrieval", HERE / "eval_retrieval.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["eval_retrieval"] = ev
_spec.loader.exec_module(ev)
gs = ev.gs

RERANK_TOP = 20  # candidates the LLM reorders; the rest keep their RRF position
H1_TARGET = 0.88
H2_MIN_IMPROVED = 7
H4_MAX_EUR = 0.50
H4_MAX_LATENCY_S = 3.0
SYSTEMS = ["bm25", "dense", "rrf", "rerank"]

# ---------------------------------------------------------------- candidates


def candidates_per_question(questions, chunks, embedder, emb_model: str, depth: int) -> list[dict]:
    """BM25, dense and RRF exactly as eval_retrieval.py, plus the passages to rerank."""
    bm25 = rt.BM25(chunks)
    print("embedding the corpus (cached after the first run)...", file=sys.stderr)
    dense = rt.DenseIndex(chunks, embedder, rt.DEFAULT_CACHE / f"embeddings-{emb_model}.npz")
    out = []
    for q in questions:
        hits = {"bm25": bm25.search(q.question), "dense": dense.search(q.question)}
        rankings = {name: rt.to_doc_ranking(h, depth) for name, h in hits.items()}
        rankings["rrf"] = rt.rrf([rankings["bm25"], rankings["dense"]])
        top = rankings["rrf"][:RERANK_TOP]
        out.append(
            {
                "q": q,
                "rankings": rankings,
                "candidates": rr.representative_chunks(list(hits.values()), top),
            }
        )
    return out


# ---------------------------------------------------------------- hypotheses


def better(new: int | None, old: int | None) -> bool:
    """New first-hit position strictly better than old (None = not found)."""
    return new is not None and (old is None or new < old)


def weak_questions(results: list[dict]) -> list[dict]:
    """RRF ranks the first relevant doc below the best single retriever (same rule as H4 of P1)."""
    weak = []
    for q in results:
        fh = q["first_hit"]
        singles = [p for p in (fh["bm25"], fh["dense"]) if p]
        if singles and (fh["rrf"] is None or fh["rrf"] > min(singles)):
            weak.append(q)
    return weak


def p95(values: list[float]) -> float:
    s = sorted(values)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


def hypotheses(results: list[dict], summary: dict) -> list[dict]:
    rrf_mrr = summary["rrf"]["all"]["MRR"]
    new_mrr = summary["rerank"]["all"]["MRR"]

    weak = weak_questions(results)
    improved = [q["id"] for q in weak if better(q["first_hit"]["rerank"], q["first_hit"]["rrf"])]

    top1 = [q for q in results if q["first_hit"]["rrf"] == 1]
    dropped = [q["id"] for q in top1 if q["first_hit"]["rerank"] != 1]

    costs = [q["rerank"]["cost_eur"] for q in results]
    lat = [q["rerank"]["latency_s"] for q in results]
    mean_cost = sum(costs) / len(costs)
    lat_p95 = p95(lat)

    return [
        {
            "h": "H1",
            "claim": f"MRR >= {H1_TARGET}",
            "measured": f"MRR {rrf_mrr:.3f} -> {new_mrr:.3f}",
            "holds": new_mrr >= H1_TARGET,
        },
        {
            "h": "H2",
            "claim": f">= {H2_MIN_IMPROVED} weak questions improve",
            "measured": f"{len(improved)} of {len(weak)} improve: {', '.join(improved) or '-'}",
            "holds": len(improved) >= H2_MIN_IMPROVED,
        },
        {
            "h": "H3",
            "claim": "no RRF position-1 question drops",
            "measured": f"{len(dropped)} of {len(top1)} drop: {', '.join(dropped) or '-'}",
            "holds": not dropped,
        },
        {
            "h": "H4",
            "claim": f"< {H4_MAX_EUR:.2f} EUR (mean) and < {H4_MAX_LATENCY_S:.0f} s (p95)",
            "measured": f"{mean_cost:.4f} EUR mean, {lat_p95:.2f} s p95 "
            f"(mean {sum(lat) / len(lat):.2f} s, max {max(lat):.2f} s)",
            "holds": mean_cost < H4_MAX_EUR and lat_p95 < H4_MAX_LATENCY_S,
        },
    ]


def hypothesis_table(rows: list[dict]) -> str:
    lines = ["| | Hypothesis | Measured | Holds? |", "|---|---|---|---|"]
    for r in rows:
        mark = "yes" if r["holds"] else "**no**"
        lines.append(f"| {r['h']} | {r['claim']} | {r['measured']} | {mark} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- run


def score(rankings: dict[str, list[str]], relevant: list[str]) -> dict:
    return {
        "first_hit": {s: ev.first_hit(rankings[s], relevant) for s in SYSTEMS},
        "scores": {
            s: {m: f(rankings[s], relevant) for m, f in ev.METRICS.items()} for s in SYSTEMS
        },
        "top10": {s: rankings[s][:10] for s in SYSTEMS},
    }


def run(items: list[dict], reranker: rr.LLMReranker, order: str, eur_per_usd: float) -> list:
    results = []
    for n, item in enumerate(items, start=1):
        q, rankings = item["q"], dict(item["rankings"])
        res = reranker.rerank(q.question, item["candidates"], order=order)
        rankings["rerank"] = res.order + rankings["rrf"][RERANK_TOP:]
        cost = rr.cost_usd(reranker.model, res.input_tokens, res.output_tokens) * eur_per_usd
        row = {"id": q.id, "type": q.type, "relevant": q.relevant_docs}
        row.update(score(rankings, q.relevant_docs))
        row["rerank"] = {
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "latency_s": res.latency_s,
            "cost_eur": cost,
            "cached": res.cached,
            "error": res.error,
        }
        results.append(row)
        fh = row["first_hit"]
        print(
            f"{n:>3}/{len(items)} {q.id}  rrf {fh['rrf']} -> rerank {fh['rerank']}"
            f"  {res.latency_s:.1f}s  {cost:.4f} EUR{'  (cache)' if res.cached else ''}"
            f"{'  ERROR ' + res.error if res.error else ''}",
            file=sys.stderr,
        )
    return results


def estimate(items: list[dict], client, model: str, eur_per_usd: float, out_tokens: int) -> None:
    """Exact input tokens from the free count_tokens endpoint; output is a guess."""
    total_in = 0
    for item in items:
        prompt = rr.build_prompt(item["q"].question, item["candidates"])
        total_in += client.messages.count_tokens(
            model=model, system=rr.SYSTEM, messages=[{"role": "user", "content": prompt}]
        ).input_tokens
    n = len(items)
    usd = rr.cost_usd(model, total_in, out_tokens * n)
    print(f"{n} questions, {total_in:,} input tokens ({total_in / n:,.0f} per question)")
    print(f"assuming ~{out_tokens} output tokens per question (answer + thinking)")
    print(f"expected cost: {usd:.2f} USD = {usd * eur_per_usd:.2f} EUR ({model})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--est-output-tokens", type=int, default=500)
    p.add_argument("--limit", type=int)
    p.add_argument("--model", default=rr.DEFAULT_MODEL, choices=sorted(rr.PRICES_USD))
    p.add_argument("--effort", default=rr.DEFAULT_EFFORT, choices=["low", "medium", "high"])
    p.add_argument("--order", default="rrf", choices=["rrf", "reversed"])
    p.add_argument("--eur-per-usd", type=float, default=0.86)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--emb-model", default="bge-m3")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--corpus", type=Path, default=rt.DEFAULT_CORPUS)
    args = p.parse_args()

    import anthropic  # imported here so the tests don't need it

    client = anthropic.Anthropic()
    questions = gs.load()[: args.limit]
    chunks = rt.chunk_corpus(rt.load_corpus(args.corpus))
    embedder = rt.ollama_embedder(args.emb_model, args.ollama_url)
    items = candidates_per_question(questions, chunks, embedder, args.emb_model, args.depth)

    if args.estimate:
        estimate(items, client, args.model, args.eur_per_usd, args.est_output_tokens)
        return

    cache = None if args.no_cache else rt.DEFAULT_CACHE / "rerank.jsonl"
    reranker = rr.LLMReranker(client, args.model, args.effort, cache)
    results = run(items, reranker, args.order, args.eur_per_usd)

    data = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "systems": SYSTEMS,
            "model": args.model,
            "effort": args.effort,
            "order": args.order,
            "rerank_top": RERANK_TOP,
            "depth": args.depth,
            "eur_per_usd": args.eur_per_usd,
            "questions": len(results),
        },
        "questions": results,
    }
    data["summary"] = ev.summarise(data)
    data["hypotheses"] = hypotheses(results, data["summary"])
    total = sum(q["rerank"]["cost_eur"] for q in results)
    spent = sum(q["rerank"]["cost_eur"] for q in results if not q["rerank"]["cached"])
    errors = [q["id"] for q in results if q["rerank"]["error"]]

    ev.RUNS.mkdir(exist_ok=True)
    out = ev.RUNS / f"p1-rerank-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print(ev.table(data["summary"]))
    print()
    if args.limit:
        print(f"Partial run ({len(results)} questions): the verdicts below are not the test.\n")
    print(hypothesis_table(data["hypotheses"]))
    print(
        f"\nSpent in this run: {spent:.2f} EUR (cached answers are free). "
        f"Cost of the whole run without cache: {total:.2f} EUR ({args.model}, effort {args.effort})"
    )
    if errors:
        print(f"Calls that kept the RRF order (refusal, truncation or bad JSON): {errors}")
    print(f"\nFull results: {out.relative_to(rt.REPO_ROOT)}")


if __name__ == "__main__":
    main()
