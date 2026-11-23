# P1 · RAG Evals Lab

**Question:** does hybrid retrieval (dense + BM25, fused with reciprocal rank fusion) actually beat dense search alone on a real operational corpus, and by how much?

**Status:** first retrieval run done (2026-11-23); reranker next.

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

## Hypothesis (written 2026-11-23, before any run)

Written before running anything, so the results can prove me wrong.

1. **Identifier questions (48):** BM25 > RRF > dense. Dense will miss more than expected. *I think: true.*
2. **Semantic questions (52):** dense > RRF > BM25, with a smaller gap than in (1). *I think: true.*
3. **All questions:** RRF wins or ties with the best, because it is never the worst in any group. Its value is less about winning than about not losing. *I think: probably true.*
4. **Risk:** when one retriever is much worse than the other in a group, RRF can end up *below* the best one. Fusing with a bad retriever also subtracts. *I think: false.*

## Retrieval harness

```bash
ollama pull bge-m3                       # embedding model, ~1.2 GB
uv sync --group p1                       # numpy
uv run python projects/p1-rag-evals-lab/eval_retrieval.py
```

| File | What it does |
| --- | --- |
| [`retrieval.py`](retrieval.py) | Chunking (headings, 300-word windows, 50 overlap), BM25 written out, dense cosine over bge-m3 via Ollama, RRF (k=60) |
| [`eval_retrieval.py`](eval_retrieval.py) | recall@5, MRR and nDCG@10 per system, split by identifier / semantic; counts questions where RRF puts the first relevant document lower than the best single retriever (hypothesis 4) |

Every metric is at document level: a document scores as its best chunk, and each retriever hands its top 20 documents to RRF. The first run embeds ~4,900 chunks once and caches them in `data/cache/` (gitignored). The summary prints as a Markdown table; per-question results go to `runs/`.

## Results: first run (2026-11-23)

Public corpus (364 documents, 4,933 chunks), 100 questions, bge-m3 on a Mac Mini M4 Pro. Mean over questions; 1.0 is perfect.

| System | recall@5 (id) | MRR (id) | nDCG@10 (id) | recall@5 (sem) | MRR (sem) | nDCG@10 (sem) | recall@5 (all) | MRR (all) | nDCG@10 (all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.889 | 0.804 | 0.825 | 0.731 | 0.562 | 0.627 | 0.807 | 0.679 | 0.722 |
| Dense | 0.955 | 0.781 | 0.817 | 0.885 | 0.711 | 0.778 | 0.918 | 0.745 | 0.797 |
| **RRF** | **0.955** | **0.895** | **0.912** | **0.923** | **0.748** | **0.793** | **0.938** | **0.818** | **0.850** |

RRF ranks the first relevant document lower than the best single retriever for that question in 13 of 100 questions (5 identifier, 8 semantic).

**Against the hypothesis:**

1. Identifier: partly wrong. BM25 beats dense on MRR only narrowly (0.804 vs 0.781), and dense finds the document in the top 5 more often.
2. Semantic: partly wrong. Dense beats BM25, but the gap is *larger* than on identifier questions, not smaller.
3. RRF wins or ties: right, and it wins every group on every metric except a tie on identifier recall@5.
4. RRF below the best: false on average, true for 13 individual questions.

**My conclusion:** sometimes keyword search is clearly right, sometimes semantic search is, and they fail on different questions. Hybrid search wins for these three corpora (Traefik, Immich, Home Assistant).

**Caveat:** Claude wrote the golden set from the documents themselves, so absolute scores are probably inflated. The comparison between systems is fair: all three took the same test.

## Plan

- ~~A hand-labelled golden set~~ Done: [100 questions](golden/README.md), 48 built around exact identifiers and 52 in plain language, each with its source document and a verbatim quote.
- An eval harness with pytest + DeepEval: recall@5, MRR, nDCG@10 and faithfulness.
- Two LLM judges (a local model and Claude) calibrated against 30 of my own labels.
- Comparisons: dense vs BM25 vs RRF vs reranker, chunk sizes, two embedding models.
- Traces in Langfuse.

**Case study:** *Does hybrid retrieval actually help?*
