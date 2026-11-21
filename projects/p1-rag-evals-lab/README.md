# P1 · RAG Evals Lab

**Question:** does hybrid retrieval (dense + BM25, fused with reciprocal rank fusion) actually beat dense search alone on a real operational corpus, and by how much?

**Status:** corpus ready; eval harness starts December 2026.

## Two corpora, two roles

| | Public corpus | Private corpus |
| --- | --- | --- |
| What | Traefik, Immich and Home Assistant documentation | My own notes vault |
| Where | Downloaded by `fetch_corpus.py`, never committed | Stays on my machine |
| Golden set | Committed here, ~100 questions | Kept local, ~30 questions |
| What gets published | Everything: data source, questions, per-question results | Aggregate metrics only |

The public corpus makes every number reproducible. The private one checks that the conclusion holds on the system I actually run. If hybrid wins on both, the result is solid; if it wins on only one, that is a finding in itself.

Both are operational documentation full of exact identifiers (`forwardAuth`, `immich_postgres`, `binary_sensor`), which is where dense retrieval is expected to struggle.

## Public corpus

```bash
uv run python projects/p1-rag-evals-lab/fetch_corpus.py
```

| Source | Documents | License |
| --- | ---: | --- |
| Traefik (`docs/content`) | 174 | MIT |
| Immich (`docs/docs`) | 87 | AGPL-3.0 |
| Home Assistant (`source/_docs` + 15 integrations) | 103 | CC BY-NC-SA 4.0 |
| **Total** | **364** | |

Each source is pinned to a commit in [`corpus/sources.toml`](corpus/sources.toml). The script writes `data/corpus/manifest.json` with a sha256 per file, so two runs, or two people, can confirm they have the same corpus. The documents keep their original licenses and are not redistributed by this repo.

## Private corpus

The harness will read it from a local path given by the `PRIVATE_CORPUS_DIR` environment variable. Nothing from it (documents, questions or per-question results) is committed. `data/private/` is gitignored as a second safety net.

## Plan

- A hand-labelled golden set, including identifier-heavy questions where dense search is expected to fail.
- An eval harness with pytest + DeepEval: recall@5, MRR, nDCG@10 and faithfulness.
- Two LLM judges (a local model and Claude) calibrated against 30 of my own labels.
- Comparisons: dense vs BM25 vs RRF vs reranker, chunk sizes, two embedding models.
- Traces in Langfuse.

**Case study:** *Does hybrid retrieval actually help?*
