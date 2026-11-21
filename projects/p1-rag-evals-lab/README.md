# P1 · RAG Evals Lab

**Question:** does hybrid retrieval (dense + BM25, fused with reciprocal rank fusion) actually beat dense search alone on a real operational corpus, and by how much?

**Status:** starts December 2026.

**Plan:**

- A hand-labelled golden set of about 100 questions, including identifier-heavy ones (container names, error codes) where dense search is expected to fail.
- An eval harness with pytest + DeepEval: recall@5, MRR, nDCG@10 and faithfulness.
- Two LLM judges (a local model and Claude) calibrated against 30 of my own labels.
- Comparisons: dense vs BM25 vs RRF vs reranker, chunk sizes, two embedding models.
- Traces in Langfuse.

**Case study:** *Does hybrid retrieval actually help?*
