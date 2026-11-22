# Golden set (public corpus)

`public.jsonl` holds 100 hand-labelled questions over the pinned public corpus. Each line is one question:

```json
{"id": "q036", "source": "immich", "type": "identifier",
 "question": "How do I dump the database from the immich_postgres container for a manual backup?",
 "identifiers": ["immich_postgres"],
 "relevant_docs": ["immich/docs/docs/administration/backup-and-restore.md",
                   "immich/docs/docs/guides/template-backup-script.md"],
 "evidence": "docker exec -t immich_postgres pg_dump --clean --if-exists",
 "answer": "docker exec -t immich_postgres pg_dump ..., piped to gzip."}
```

| | Traefik | Immich | Home Assistant | Total |
| --- | ---: | ---: | ---: | ---: |
| identifier | 17 | 16 | 15 | 48 |
| semantic | 18 | 17 | 17 | 52 |

## Two kinds of question

- **identifier**: the question hinges on an exact token from the docs (`forwardAuth`, `immich_postgres`, `purge_keep_days`). These are the questions where dense retrieval is expected to struggle and BM25 to shine. The tokens are listed in `identifiers`.
- **semantic**: the question is phrased the way someone who hasn't read the docs would ask it, with no doc-specific tokens and often without the product name ("the photo server", "the proxy"). Here dense retrieval should do well and BM25 worse.

The split lets every metric be reported per type. If hybrid retrieval helps, it should lift identifier questions without hurting semantic ones; the average over all 100 would hide that.

## Labelling rules

1. **A document is relevant if it answers the question**, not if it merely mentions the term. `immich_postgres` appears in six documents; only the two that show the backup command are relevant to the backup question.
2. **The first document is the primary one** and must contain `evidence` verbatim. `golden_set.py` checks this against the corpus, so a label can't point at the wrong file.
3. **Other documents that also answer are listed after it.** Missing one would count a correct retrieval as a miss and make every system look worse than it is. `golden_set.py --audit` lists unlabelled documents that contain the same identifiers; each was reviewed by hand.
4. **Semantic questions must not contain identifier-like tokens** (snake_case, camelCase, ALL_CAPS, dotted options). The checker enforces it, so the two groups stay honest.
5. **`answer` is a short reference answer**, written from the evidence. Retrieval metrics ignore it; it is there for the faithfulness evals later.

## Checking it

```bash
uv run python projects/p1-rag-evals-lab/golden_set.py           # structure; plus corpus checks if data/corpus exists
uv run python projects/p1-rag-evals-lab/golden_set.py --audit   # possible missing labels
```

CI runs the structural checks only, because the corpus is not committed.

## Known limits

- One labeller (me), no second opinion yet. The 30 labels for judge calibration will be a partial cross-check.
- Questions were written while reading the docs, so some semantic questions may still share wording with their answer. Paraphrasing harder would make dense retrieval look better; this is a deliberate middle ground.
- The labels are tied to the commits in `corpus/sources.toml`. Moving a commit means re-running the checker and fixing what breaks.
