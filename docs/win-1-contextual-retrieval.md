# Win 1 — Contextual Retrieval

## Why
Retrieval quality is the ceiling on answer quality, and a chunk retrieved in isolation loses
its "where am I from / what is this about" context. Contextual Retrieval prepends a short,
LLM-generated situating header to each chunk **before embedding and BM25-indexing**, so both
retrieval halves match on the chunk *plus* its context. Published lift: ~35% fewer retrieval
failures (contextual embeddings), ~49% adding contextual BM25, ~67% stacked with reranking
(which we already have).

## How it works
At ingest (`api/main.py::_ingest_bytes`), after chunking:
1. `src/ingestion/contextualizer.py::contextualize_chunks(chunks, doc_text)` sends the whole
   document as a stable prefix (OpenAI prompt-caching makes the repeat cheap) and asks a cheap
   model (`CONTEXT_MODEL`, default `gpt-4o-mini`) for a 1-2 sentence header per chunk. Chunks
   are contextualised concurrently via LangChain `.batch`.
2. Long docs (> `CONTEXT_MAX_DOC_CHARS`, default 40k) are summarised once and situated against
   the summary.
3. Each chunk becomes: `text = "<header>\n\n<chunk>"` (embedded + BM25-indexed, since the BM25
   corpus is built from the stored `text`), with `metadata.raw_text` = the original chunk and
   `metadata.context` = the header.
4. Citations show `metadata.raw_text` (clean), not the header-prefixed text
   (`src/rag_pipeline.py::_citation`).

Failure is graceful: any error returns the chunks unchanged (no header), so ingestion never
fails on the context step.

## Config
- `CONTEXTUAL_RETRIEVAL` (default `true`) — master toggle.
- `CONTEXT_MODEL` (default `gpt-4o-mini`) — header-generation model.
- `CONTEXT_MAX_DOC_CHARS` (default `40000`) — summarise docs larger than this.

## Cost / latency
- Query time: **unchanged** (all work is at ingest).
- Ingest: +1 cheap LLM call per chunk (one-time), bounded by `MAX_INGEST_CHUNKS` and mitigated
  by prompt-caching + concurrency.

## Migration
Existing (non-contextual) chunks stay valid; only new ingests get headers. Re-ingest key docs
to benefit. Mixed indexes are fine.

## Proving the lift (Phase C — pending)
A/B on a small representative golden set: ingest the same docs into a **baseline** index (no
context) vs a **contextual** index, then run the eval harness `--retrieval-only`
(hit-rate@k / MRR / nDCG) plus a judged run (faithfulness). Ship only if contextual wins on
retrieval without hurting faithfulness. Needs representative docs + questions.

## Not in scope (deferred)
- Structure-aware chunking with heading/page extraction for PDFs (needs a layout parser; the
  context header captures most of the benefit for now).
- Span-level snippet extraction (tighter citations) — a future snippet-eval upgrade.
