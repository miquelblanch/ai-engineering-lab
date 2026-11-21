# <Title as a question or a claim>

> One sentence: what was tested and what came out of it.

**Project:** Pn · **Dates:** <start> – <end> · **Hours:** <n> · **Code:** `projects/pn-.../`

---

## The problem

What wasn't known before starting, and why it matters. A concrete case, not a generality.

## Hypothesis

What I expected to find, written **before** measuring. If the result contradicts it, it stays here as it was.

## What was measured, and how

- **Data:** corpus, size, where it comes from, how it was sanitised.
- **Eval set:** how many cases, who labelled them, how they were chosen.
- **Metrics:** which ones, and why these and not others.
- **Variants compared:** baseline and alternatives.
- **Reproduce:** the exact command.

```bash
uv run python -m ...
```

## Results

| Variant | Metric 1 | Metric 2 | Cost / latency |
| --- | --- | --- | --- |
| Baseline | | | |
| Alternative | | | |

What the table says, in two or three sentences. Including what it doesn't say.

## What surprised me

The finding I didn't see coming, with the evidence that shows it.

## Decisions taken from this

What changes in a real system because of this result, and what doesn't.

## What is still unknown

Limits of the measurement: sample size, judge bias, corpus representativeness. What would have to be measured next to be sure.
