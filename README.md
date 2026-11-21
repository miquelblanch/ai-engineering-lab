# ai-engineering-lab

Code for my AI Engineering learning projects, one folder per project.

The write-ups (case studies, learning log, progress) live on my site: **[miquel.cleverhans.es](https://miquel.cleverhans.es/en/)**. This repo holds what you need to check them: the code, the eval sets and the commands to reproduce every number a case study claims.

## Projects

| # | Project | Question it answers | Status |
| --- | --- | --- | --- |
| P1 | [RAG Evals Lab](projects/p1-rag-evals-lab/) | Does hybrid retrieval actually help on a real operational corpus? | Starts Dec 2026 |
| P2 | [homelab-mcp + agent evals](projects/p2-homelab-mcp/) | How well does an ops agent do on its own incident history? | Planned |
| P3 | [Evals in CI](projects/p3-evals-ci/) | Can LLM quality gate a merge the way tests do? | Planned |
| P4 | [Local model benchmark](projects/p4-local-models/) | What can a Mac Mini M4 Pro really serve, on real tasks? | Planned |
| P5 | [Transformer + LoRA](projects/p5-transformer-lora/) | From attention to a fine-tuned triage model on a Mac Mini | Planned |

The plan behind them, and the skills each one is meant to build, is on the [progress page](https://miquel.cleverhans.es/en/progreso/).

## How the work is written up

Each project ends with a case study and is built in pieces, each with a learning log entry. Both follow the templates in [`templates/`](templates/):

- [`case-study.md`](templates/case-study.md): problem, hypothesis, what was measured, result, and what is still unknown.
- [`learning-log.md`](templates/learning-log.md): eight questions answered in my own words after each piece. If I can't explain it, it isn't done.

## Running it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                 # dev tools: pytest, ruff
uv sync --group p1      # plus one project's dependencies, once it exists
uv run pytest
uv run ruff check .
```

## Ground rules

- **No private data.** Evals run on a sanitised or public corpus. Nothing from my home network, my notes or my employer is committed; CI runs [gitleaks](https://github.com/gitleaks/gitleaks) with extra rules for private IPs and local paths.
- **Every number is reproducible.** A result that appears in a case study has a command here that produces it.
- **Negative results are published too.** If hybrid retrieval doesn't help, that is the case study.

## License

[MIT](LICENSE)
