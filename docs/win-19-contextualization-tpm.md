# Win 19 — Contextualization at scale (TPM-safe)

## Why
Win 18 found that **contextualization is the dominant faithfulness lever** (0.69 → 0.83–0.86),
but it silently failed on the biggest documents: a re-ingest left only ~63% of chunks with a
situating header, and the two largest reports (122- and 82-chunk) got **none**.

## Root cause
The contextualizer resends the whole document as a per-chunk prefix, so input tokens ≈
`doc_tokens × num_chunks`. The old summarization trigger was **char-based only** (summarize if
the doc > 40k chars), which ignores chunk count. A 35k-char report split into 53 chunks resends
that text 53 times ≈ **470k input tokens in one burst** — 2.4× OpenAI's 200k tokens/min limit —
so the whole document 429'd and fell back to header-less chunks. (Prompt caching lowers *cost*
but cached tokens still count toward the *rate* limit.)

| Doc | chars | chunks | old decision | est. tokens |
|---|---|---|---|---|
| Argus | 34.8k | 53 | full doc | **470k** ❌ |
| JPMorgan | 23.7k | 38 | full doc | 235k ❌ |
| China Renaissance | 22.0k | 31 | full doc | 178k ❌ |
| Arete | 14.2k | 25 | full doc | 89k ✓ |

## Fix (`src/ingestion/contextualizer.py`)
1. **Token-budget-aware summarization** — summarize when `est_doc_tokens × num_chunks >
   CONTEXT_TOKEN_BUDGET` (default 120k), not just when the doc is long. The four blow-up docs
   above now situate against a bounded one-shot summary; small/few-chunk docs (Arete) keep the
   richer full-doc context.
2. **Retry with backoff** — `ChatOpenAI(max_retries=CONTEXT_MAX_RETRIES=6)` rides out transient
   429s from cross-document bursts instead of failing.
3. **Per-chunk isolation** — `llm.batch(..., return_exceptions=True)`; one failed request costs
   one header, not the whole document's.

## Result
Re-ingesting the 6-report corpus: **351/351 chunks (100%) now carry a context header**,
including the previously-failing Morningstar (122/122) and expert call (82/82). No 429 failures.
This also unblocks scale — per-document contextualization cost is now bounded regardless of
chunk count.

## Config
- `CONTEXT_TOKEN_BUDGET` (default `120000`) — summarize above this estimated blow-up.
- `CONTEXT_MAX_RETRIES` (default `6`) — 429 backoff retries.
