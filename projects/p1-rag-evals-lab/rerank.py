"""LLM reranker: reorders the top 20 RRF documents with one LLM call per question.

Two providers, same prompt and schema: Claude (Messages API) and OpenAI
(Responses API), chosen by model name. Written by hand, without a framework,
so every step can be read:

    representative_chunks()  one passage per candidate document (its best chunk)
    build_prompt()           question + 20 numbered passages
    parse_ranking()          turns the model's answer into a full, valid order
    LLMReranker              one API call per question, cached on disk, with
                             tokens, latency and cost recorded for each call

Listwise, not pointwise: the model sees all 20 passages at once and returns
their order. One call per question keeps cost and latency low and fits any
API rate limit; the price is position bias (models favour passages near the
top of the list), which `order="reversed"` lets you measure.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from retrieval import Chunk

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "low"

# USD per million tokens (input, output), Claude API list prices, 2026-11.
# Output includes thinking tokens, which are billed as output.
# Sources: Claude API pricing; OpenAI model page for gpt-6-luna (both read 2026-11-26).
PRICES_USD = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "gpt-6-luna": (0.10, 0.50),
}

# Effort levels each provider accepts (OpenAI also has "none": no reasoning).
EFFORTS = {
    "anthropic": ("low", "medium", "high"),
    "openai": ("none", "low", "medium", "high"),
}


def provider_of(model: str) -> str:
    return "openai" if model.startswith("gpt-") else "anthropic"


SYSTEM = (
    "You rank documentation passages by how well they answer a user's question "
    "about self-hosted software (Traefik, Immich, Home Assistant).\n"
    "A passage is relevant when it contains the information needed to answer the "
    "question, not when it merely shares words with it. Judge every passage on its "
    "content alone: its number and its place in the list say nothing about its relevance."
)

SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ranking"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Candidate:
    doc: str  # corpus path, as in the golden set
    text: str  # the passage the model reads: the document's best chunk


@dataclass(frozen=True)
class RerankResult:
    order: list[str]  # documents, most relevant first
    input_tokens: int
    output_tokens: int
    latency_s: float  # time of the original API call, also when read from cache
    cached: bool
    error: str | None = None  # refusal / truncation / unparseable answer: RRF order kept


# ---------------------------------------------------------------- passages


def representative_chunks(
    hit_lists: Sequence[Sequence[tuple[Chunk, float]]], docs: Sequence[str]
) -> list[Candidate]:
    """For each document, the chunk that ranked highest in any retriever.

    The golden set labels documents, but a document can be thousands of words.
    The model reads one chunk per document: the one BM25 or dense search found
    closest to the question, which is also what a RAG system would pass on.
    """
    best: dict[str, tuple[int, Chunk]] = {}
    wanted = set(docs)
    for hits in hit_lists:
        for pos, (c, _) in enumerate(hits):
            if c.doc in wanted and (c.doc not in best or pos < best[c.doc][0]):
                best[c.doc] = (pos, c)
    return [Candidate(d, best[d][1].text if d in best else "") for d in docs]


def build_prompt(question: str, candidates: Sequence[Candidate]) -> str:
    passages = "\n\n".join(f"[{i}] {c.doc}\n{c.text}" for i, c in enumerate(candidates, start=1))
    return (
        f"Question: {question}\n\n"
        f"Passages:\n\n{passages}\n\n"
        f"Return the numbers of all {len(candidates)} passages in `ranking`, "
        "most relevant first."
    )


def parse_ranking(ids: Sequence[object], n: int) -> list[int]:
    """A full order of 1..n from whatever the model returned.

    Keeps valid numbers in the model's order, drops repeats and out-of-range
    values, and appends any passage the model forgot in its original position.
    A forgotten passage is not lost; it just gets no promotion.
    """
    seen: list[int] = []
    for i in ids:
        if isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= n and i not in seen:
            seen.append(i)
    return seen + [i for i in range(1, n + 1) if i not in seen]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_USD[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


# ---------------------------------------------------------------- reranker


class LLMReranker:
    """One API call per question, answers cached by prompt hash.

    `client` is an `anthropic.Anthropic()` for Claude models or an
    `openai.OpenAI()` for gpt-* models (or a fake with the same method in
    tests). The cache is a JSONL file: one line per paid
    call, appended immediately, so a crash halfway through a run doesn't make
    you pay for the same answers twice, and a rerun reproduces the first one.
    """

    def __init__(
        self,
        client,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        cache: Path | None = None,
        max_tokens: int = 4096,
    ):
        if model not in PRICES_USD:
            raise ValueError(f"no price for {model}: add it to PRICES_USD")
        if effort not in EFFORTS[provider_of(model)]:
            raise ValueError(f"{model} does not take effort {effort!r}")
        self.client = client
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.cache = cache
        self._stored: dict[str, dict] = {}
        if cache and cache.is_file():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._stored[entry["key"]] = entry

    def key(self, prompt: str) -> str:
        raw = json.dumps([self.model, self.effort, SYSTEM, prompt])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def rerank(
        self, question: str, candidates: Sequence[Candidate], order: str = "rrf"
    ) -> RerankResult:
        """`order` is how passages are shown: "rrf" (as fused) or "reversed"."""
        shown = list(candidates) if order == "rrf" else list(reversed(candidates))
        prompt = build_prompt(question, shown)
        k = self.key(prompt)
        entry = self._stored.get(k)
        cached = entry is not None
        if entry is None:
            entry = self._call(prompt)
            entry["key"] = k
            self._stored[k] = entry
            if self.cache:
                self.cache.parent.mkdir(parents=True, exist_ok=True)
                with self.cache.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")

        error = entry.get("error")
        if error:
            ranked = list(candidates)  # keep the RRF order
        else:
            ranked = [shown[i - 1] for i in parse_ranking(entry["ranking"], len(shown))]
        return RerankResult(
            order=[c.doc for c in ranked],
            input_tokens=entry["input_tokens"],
            output_tokens=entry["output_tokens"],
            latency_s=entry["latency_s"],
            cached=cached,
            error=error,
        )

    def _call(self, prompt: str) -> dict:
        start = time.perf_counter()
        if provider_of(self.model) == "openai":
            entry = self._call_openai(prompt)
        else:
            entry = self._call_anthropic(prompt)
        entry["latency_s"] = round(time.perf_counter() - start, 3)
        entry["model"] = self.model
        return entry

    def _call_anthropic(self, prompt: str) -> dict:
        # No temperature: current Claude models reject sampling parameters
        # (400 on API version 2026-10-21). Determinism comes from the cache.
        r = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
        )
        entry = _entry(r.usage.input_tokens, r.usage.output_tokens, r.stop_reason)
        if r.stop_reason in ("refusal", "max_tokens"):
            entry["error"] = r.stop_reason
            return entry
        return _parse_into(entry, next((b.text for b in r.content if b.type == "text"), ""))

    def _call_openai(self, prompt: str) -> dict:
        # Responses API. Same system prompt, same JSON schema (strict), no
        # temperature either, so both providers get exactly the same request.
        # output_tokens already includes the reasoning tokens.
        r = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM,
            input=prompt,
            max_output_tokens=self.max_tokens,
            reasoning={"effort": self.effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ranking",
                    "schema": SCHEMA,
                    "strict": True,
                }
            },
        )
        entry = _entry(r.usage.input_tokens, r.usage.output_tokens, r.status)
        if r.status == "incomplete":
            reason = getattr(r.incomplete_details, "reason", None) or "incomplete"
            entry["error"] = "max_tokens" if reason == "max_output_tokens" else reason
            return entry
        refused = any(
            getattr(c, "type", None) == "refusal"
            for item in r.output
            for c in (getattr(item, "content", None) or [])
        )
        if refused:
            entry["error"] = "refusal"
            return entry
        return _parse_into(entry, r.output_text)


def _entry(input_tokens: int, output_tokens: int, stop_reason: str | None) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason,
        "ranking": [],
        "error": None,
    }


def _parse_into(entry: dict, text: str) -> dict:
    try:
        entry["ranking"] = json.loads(text)["ranking"]
    except (json.JSONDecodeError, KeyError, TypeError):
        entry["error"] = "unparseable"
    return entry
