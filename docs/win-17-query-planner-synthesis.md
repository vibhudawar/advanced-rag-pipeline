# Win 17 (Phase 2) — Query planner + diversity retrieval for multi-document answers

## Why
The bar for this project is **questions that need intelligence across several documents**
("how does Company A's AI strategy compare to Company B's?"). The pre-Phase-2 pipeline was
structurally biased against that: one semantic query → fuse top-10 → rerank to the **5 best
chunks**, which almost always collapse onto the single most-relevant document. Both sides of a
comparison rarely reached the generator together, so it could not synthesize across them.

## How it works
A lightweight **query planner** runs before retrieval, then retrieval fans out and diversifies.

### 1. Planner (`src/retrieval/nlu.py::plan_query`)
One structured LLM call (on the cheap generation model) that replaces the old plain
`condense_query` rewrite and returns a `QueryPlan`:
- `standalone_query` — the follow-up resolved against history (references, "instead", "more").
- `query_type` — `simple` | `comparison` | `aggregation`.
- `sub_queries` — for comparison/aggregation only: one focused retrieval query per
  entity/facet (deduped, capped at `MAX_SUB_QUERIES = 4`).

`simple` questions get an empty `sub_queries` list and take the **exact pre-Phase-2 path** —
this is what protects the retrieval benchmark. On any planner error the pipeline degrades to a
`simple` plan on the original query.

### 2. Diversity retrieval (`src/rag_pipeline.py::_retrieve_plan`)
- **simple** (and any doc-scoped request): unchanged single-query gated path
  (`_retrieve_gated`).
- **comparison / aggregation**: run one hybrid retrieval pass per sub-query (+ the standalone),
  dedup the union by content hash, rerank the union against the overall question with a wider
  depth (`RAG_UNION_RERANK_K`), then **diversify** (`_diversify`): round-robin across source
  documents (`RAG_PER_DOC_CAP` chunks max per doc) up to `RAG_MULTI_FINAL_K`, so several
  documents are guaranteed into the context. The existing grounded-generation synthesis rule
  (Win 15) then reasons across them.

  The LLM snippet gate is **deliberately skipped** on this path (as it is for doc-scoped
  requests). The gate judges each chunk against the umbrella question, and a single chunk about
  one entity rarely looks like it "answers the comparison" on its own — so gating collapses the
  diversified union and forces a false abstention (observed: a real "Argus vs JPMorgan" question
  abstained with one surviving chunk). Relevance is already handled by per-sub-query retrieval +
  rerank; the generator still abstains via its own grounded-prompt rule if the assembled context
  genuinely doesn't support an answer.

Round-robin diversity (best chunk of each doc first, then second-best, …) is a cheaper,
deterministic stand-in for embedding-space MMR — it directly targets the failure mode ("top-k
all from one doc") without a second embedding pass.

## Config (`config.py`)
- `RAG_MULTI_QUERY` (default `true`) — master toggle; off = single-query path everywhere.
- `RAG_UNION_RERANK_K` (default `12`) — rerank depth over the fan-out union.
- `RAG_PER_DOC_CAP` (default `3`) — max chunks from any one document in the final context.
- `RAG_MULTI_FINAL_K` (default `8`) — context size for multi-doc answers.

## Cost
The planner adds **one cheap `gpt-5.4-nano` call per query** (~$0.00007) and ~1s latency.
Comparison/aggregation questions additionally run one hybrid + rerank pass per sub-query
(bounded by `MAX_SUB_QUERIES`). Simple questions pay only the single planner call — no extra
retrieval.

## Measured — no regression on the public benchmark
`gpt-5.4-nano` generator, judge `gpt-4o`, gen_golden (scifact, n=30). The planner routes every
single-hop scifact query to `simple`, so the numbers must hold:

| Metric | Baseline (pre-Phase-2) | Phase 2 (planner on) |
|---|---|---|
| hit@10 / MRR / nDCG@10 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| faithfulness | 1.000 | 1.000 |
| answer relevance | 0.9833 | 0.9833 |
| context relevance | 0.9875 | 0.9917 |
| abstention accuracy | 0.90 | 0.90 |
| latency (mean) | 4.04s | 5.03s |

## Measured — multi-doc benefit on the demo corpus
The public benchmark has no comparison/aggregation questions, so the *benefit* was validated by
ingesting 4 Meta analyst reports from 4 different publishers (Argus, China Renaissance,
JPMorgan, Morningstar) into a throwaway index and inspecting how many distinct source documents
reach the generator:

| Question | Route | Sub-queries | Distinct docs in context | Outcome |
|---|---|---|---|---|
| "What does the Morningstar report say about Meta's AI spending?" | simple | 0 | 2 | grounded + cited |
| "How do Argus and JPMorgan differ in their view on Meta?" | comparison | 2 | 3 | Argus BUY $800 vs JPMorgan Neutral, cited |
| "How do the analysts differ in their outlook on Meta this quarter?" | aggregation | 4 | 4 (all publishers) | per-analyst synthesis, cited |

Pre-fix, the comparison question abstained (the gate collapsed the union to one chunk) — the
gate-skip above is what fixed it. The aggregation question originally misrouted to `simple`
until the planner's aggregation definition was broadened to cover "how do the analysts differ /
is there consensus" phrasing.

## Rejected — reranker-score gate on the simple path (negative result)
The plan included replacing the LLM snippet gate (one `gpt-4o-mini` call/query) with a cheap
reranker-score threshold: keep chunks with `rerank_score >= t`, abstain if none survive. It was
calibrated on gen_golden (75 items, 15 unanswerable) by recording each query's top rerank score
and splitting by answerable vs unanswerable:

| Group | median top rerank score |
|---|---|
| answerable (n=60) | 0.938 (but ~18% score ~0 — retrieval genuinely failed) |
| unanswerable (n=15) | 0.647 (max 0.775) |

The distributions **overlap badly**: unanswerable questions retrieve *topically similar* chunks
that the reranker scores high (median 0.647), because the reranker measures similarity, not
answerability. Sweeping every threshold t∈[0,1], the **best achievable abstention accuracy is
0.813** (at t=0.83, chosen in hindsight — optimistic), versus **0.900 for the LLM gate**. The
gate's value is exactly that it reasons about whether a chunk *answers* the question, which the
score cannot. So the gate stays; the score threshold is **not shipped**. The reranker keeps
`min_score=0.0` (Win 12) and the LLM gate does the filtering on the simple path.
