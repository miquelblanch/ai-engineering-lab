"""Exercise 1 of the DeepEval course: one metric, three answers.

Question q001 from the golden set, three hand-written answers, one metric
(answer relevancy, threshold 0.7) and gpt-6-luna as the judge. Each run
appends its scores to runs/ex01_answer_relevancy.jsonl so three runs can be
compared. Skipped when there is no OPENAI_API_KEY (as in CI): it calls a paid API.

Run from the repo root:
    uv run pytest projects/p1-rag-evals-lab/exercises/test_ex01_answer_relevancy.py -s
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import AnswerRelevancyMetric  # noqa: E402  (loads .env on import)
from deepeval.models import OpenAIModel  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

JUDGE = "gpt-6-luna"
THRESHOLD = 0.7
RUNS = Path(__file__).resolve().parents[1] / "runs" / "ex01_answer_relevancy.jsonl"

QUESTION = "What does the trustForwardHeader option do in the forwardAuth middleware?"

# My prediction, written before the first run (2026-11-27).
ANSWERS = {
    "A": {
        "text": "It makes forwardAuth trust all X-Forwarded-* headers. It is deprecated "
        "since v3.6.14; use forwardedHeaders.trustedIPs on the entryPoint instead.",
        "predicted": 0.95,
    },
    "B": {
        "text": "It makes forwardAuth trust all X-Forwarded-* headers.",
        "predicted": 0.6,
    },
    "C": {
        "text": "forwardAuth delegates authentication to an external service before "
        "forwarding the request.",
        "predicted": 0.3,
    },
}

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY (paid judge calls)"
)


@pytest.mark.parametrize("label", list(ANSWERS))
def test_answer_relevancy(label: str) -> None:
    answer = ANSWERS[label]
    metric = AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=OpenAIModel(model=JUDGE),
        include_reason=True,
        async_mode=False,
    )
    metric.measure(LLMTestCase(input=QUESTION, actual_output=answer["text"]))

    print(f"\n[{label}] score {metric.score:.2f} (predicted {answer['predicted']:.2f})")
    print(f"[{label}] reason: {metric.reason}")

    RUNS.parent.mkdir(exist_ok=True)
    with RUNS.open("a") as f:
        row = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "answer": label,
            "score": metric.score,
            "predicted": answer["predicted"],
            "reason": metric.reason,
            "cost_usd": metric.evaluation_cost,
        }
        f.write(json.dumps(row) + "\n")

    assert metric.score >= THRESHOLD, f"{label} scored {metric.score:.2f} < {THRESHOLD}"
