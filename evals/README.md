# Eval harness (WIN 1)

Measures retrieval and generation quality so every later change to the pipeline is
**provable**, not asserted. This is the baseline scoreboard the README table is built from.

## What it measures

| Metric | Type | How |
|---|---|---|
| hit-rate@k, MRR, nDCG@k | retrieval | pure math vs. ground-truth chunk hashes — no LLM, un-gameable |
| faithfulness | generation | hand-rolled LLM judge: are the answer's claims grounded in the context? |
| answer relevance | generation | judge: does the answer address the question? |
| context relevance | generation | judge: is the retrieved context on-topic? |
| abstention accuracy | behaviour | did it correctly say "I don't know" on unanswerable questions? |

Judge model ≠ generator model (generator = `gpt-4o-mini`, judge defaults to `gpt-4o`) so a
model never grades its own output.

## Prerequisites

`.env` must contain: `PINECONE_API_KEY`, `COHERE_API_KEY`, and `OPENAI_API_KEY` and/or
`GEMINI_API_KEY`. A **populated** Pinecone index (documents already ingested).

## Run it

```bash
# 1. Generate a golden set from your own corpus (writes data/golden.jsonl)
python -m evals.generate_golden --index <your-index-name>

# 2. (recommended) hand-curate ~50 items: fix leaky questions, rewrite a few in
#    real-user phrasing, and set "needs_review": false on the ones you trust.

# 3. Baseline scoreboard
python -m evals.evaluate --index <your-index-name>

# fast iterations:
python -m evals.evaluate --index <name> --no-judge      # retrieval only, no LLM cost
python -m evals.evaluate --index <name> --limit 20      # smoke run
```

Reports are written to `evals/reports/<pipeline>-<timestamp>.json`. Compare runs to see a
win's before/after. `evals/reports/` is disposable; commit `data/golden.jsonl` (it's the
test set).

## BEIR (standardized retrieval numbers — optional, step 2)

```bash
pip install -r evals/requirements-eval.txt
python -m evals.load_beir --dataset scifact --index beir-scifact
python -m evals.evaluate --index beir-scifact --golden data/beir_scifact.jsonl --no-judge
```

Heavier (embeds a whole benchmark corpus into a throwaway index), but gives an nDCG@10 you
can cite against published BEIR results.

## Files

| File | Role |
|---|---|
| `schema.py` | `GoldenItem`, `RunResult`, content-hash ground-truth key, jsonl IO |
| `metrics.py` | retrieval metrics (pure functions) |
| `judges.py` | hand-rolled LLM-as-judge |
| `pipeline_adapter.py` | wraps a pipeline as `run(query) -> RunResult`; add a subclass per win |
| `generate_golden.py` | own-corpus golden set generator |
| `load_beir.py` | BEIR subset loader + ingester |
| `evaluate.py` | runs a pipeline over the golden set, prints + saves the scoreboard |

## Adding a new pipeline version (later wins)

Subclass `BaselinePipeline` (or implement `run(query) -> RunResult`), register it in
`PIPELINES`, then `python -m evals.evaluate --pipeline <name> --index <name>`. Same golden
set, same metrics → directly comparable numbers.
