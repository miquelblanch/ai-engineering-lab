# P3 · Evals in CI

**Question:** can LLM quality block a merge the same way a failing test does?

**Status:** planned.

**Plan:**

- Turn the P1 and P2 evals into a reusable library.
- Run them in GitHub Actions and fail the build when a metric drops below its threshold.
- Cost and degradation alerts in Langfuse, with three months of observability data.

**Case study:** *Evals as CI quality gates for a personal AI stack.*
