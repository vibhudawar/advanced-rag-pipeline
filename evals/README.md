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

## Results — retrieval (BEIR scifact, 2000-doc subset, 300 labelled queries)

| Pipeline | hit_rate@10 | MRR | nDCG@10 |
|---|---|---|---|
| WIN 1 — vector-only (`text-embedding-3-small`) | 0.9033 | 0.7588 | 0.7912 |
| **WIN 2 — hybrid (BM25 + vector, RRF k=60)** | **0.9200** | **0.7700** | **0.8005** |
| WIN 3 — NLU query rewriting + hybrid | 0.9167 | 0.7276 | 0.7649 |

WIN 2 improves every retrieval metric with no regression — the expected result on a
lexically-precise corpus, where BM25 recovers exact-term matches (entities, rare terms)
that dense vectors smooth over.

**WIN 3 is a documented negative result.** Adding an NLU query-rewriting step *regressed*
retrieval on this benchmark (MRR/nDCG down), across two designs (replace-query and
additive). Cause: BEIR queries are already well-formed, so LLM rewriting/expansion dilutes
precision — extra keyword queries spread RRF mass onto marginally-relevant docs. So the
retrieval default stays **hybrid**; NLU is kept non-default (`--pipeline nlu_hybrid`) for
reproducibility, and the CI gate correctly rejects it. NLU's value is on a different axis —
see routing below.

## Results — NLU intent routing (`python -m evals.routing`)

| Metric | Score |
|---|---|
| intent routing accuracy (greeting / off_topic / rag_query, 23 labelled) | **100%** (up from 82.6% after defining the intents in the prompt) |

This is the axis BEIR cannot test: NLU correctly routes greetings and support/transactional
requests *away* from document search (avoiding wasted, mis-cited retrieval) and sends real
questions to RAG. That — plus conversational restatement and metadata filtering — is where
NLU earns its place, not raw retrieval on clean queries.

## Committed test sets (both public, non-PII)

| File | Purpose | Has reference answers? | Corpus |
|---|---|---|---|
| `data/beir_scifact.jsonl` | **retrieval** benchmark (hit@k/MRR/nDCG) | no (qrels only) | BEIR scifact, 2000-doc subset |
| `data/gen_golden.jsonl` | **generation** eval (faithfulness / answer & context relevance / abstention) | yes + 15 unanswerables | same public scifact corpus |

`gen_golden.jsonl` exists because BEIR has no reference answers, so it can't score
generation. It was produced by `generate_golden.py` over the public `beir-scifact` index
(reused — no new ingestion, no PII) and shuffled (seed 42). Items are synthetic and marked
`needs_review: true` — curate a subset before treating the numbers as authoritative.

## Results — generation (gen_golden.jsonl, indicative 20-item sanity run, hybrid + judge)

| faithfulness | answer relevance | context relevance | abstention accuracy |
|---|---|---|---|
| 1.00 | 0.99 | 0.91 | **0.80** |

Abstention is the headroom: on unanswerable questions the current pipeline often answers
anyway instead of declining — the weakness WIN 4 (snippet-relevance gate + grounded
citations + disciplined "I don't know") targets. Retrieval on this set is ~1.0 (questions
were generated from their source chunk), which is why retrieval quality is measured on BEIR,
not here. Full generation baseline is run at the start of WIN 4.

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
win's before/after. `evals/reports/` is disposable.

> **PII note:** `data/golden.jsonl` is generated from *your* documents, which may be
> personal — it is **gitignored, not committed**. The committed test sets below are built
> from public corpora only.

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
