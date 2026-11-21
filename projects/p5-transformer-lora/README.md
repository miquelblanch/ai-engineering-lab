# P5 · Transformer from scratch + LoRA

**Question:** can a small, locally fine-tuned model triage incidents as well as large hosted ones?

**Status:** planned.

**Plan:**

- A small transformer in PyTorch, trained on a public corpus, with loss and perplexity reported.
- A LoRA fine-tune with MLX for incident classification, compared against the base model and hosted models using the P2 evals.
- Published on Hugging Face with a model card.

**Case study:** *From attention to a fine-tuned triage model on a Mac Mini.*
