# Advanced RAG Pipeline

An agentic, multi-document Retrieval-Augmented Generation system that answers questions
requiring synthesis **across many documents**, with inline citations and disciplined
abstention. Built and tuned around a measured evaluation harness — every retrieval and
generation change is proven against metrics, not asserted.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat&logo=nextdotjs&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=flat&logo=pinecone&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3FCF8E?style=flat&logo=supabase&logoColor=white)

**Live demo:** https://advanced-rag-pipeline.vercel.app

[<img src="https://img.youtube.com/vi/O0mlcQ9cy9o/maxresdefault.jpg" alt="Watch the walkthrough video" width="640">](https://youtu.be/O0mlcQ9cy9o)

▶ **[Watch the 5-minute walkthrough](https://youtu.be/O0mlcQ9cy9o)**

---

## Table of contents

- [What it does](#what-it-does)
- [Key capabilities](#key-capabilities)
- [Architecture](#architecture)
- [How it works](#how-it-works)
- [Evaluation & metrics](#evaluation--metrics)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Configuration](#configuration)

---

## What it does

Upload a set of documents, then ask questions that no single document answers on its own —
comparisons ("how does A's position differ from B's?"), aggregations ("what is the consensus
across these reports?"), or simple lookups. The system routes each question, retrieves the
right evidence from across the corpus, synthesizes an answer, and **cites the exact sources**
for every claim. When the documents don't support an answer, it says so rather than
fabricating one.

The pipeline is deliberately reliability-first:

- **Grounded or nothing** — answers are generated only from retrieved context, with inline
  `[n]` citations; unsupported questions get an explicit "I don't have enough information".
- **Multi-document by design** — retrieval fans out and diversifies across source documents so
  both sides of a comparison actually reach the generator.
- **Measured, not guessed** — a committed evaluation harness scores retrieval and generation
  quality so every change is a provable win (or a documented negative result).

## Key capabilities

| Capability | What it means |
|---|---|
| **Hybrid retrieval** | Dense vectors (semantic) fused with BM25 (exact-term) via Reciprocal Rank Fusion — covers both "means the same thing" and "contains this exact token". |
| **Contextual retrieval** | Each chunk gets an LLM-generated header situating it in its document before embedding + indexing, so a chunk retrieved in isolation still carries its context. |
| **Query planning / routing** | One structured call classifies each question as `simple` / `comparison` / `aggregation` and decomposes multi-part questions into focused sub-queries. |
| **Multi-document synthesis** | Comparison/aggregation questions retrieve per sub-query, then diversify across source documents (per-document cap) so the answer spans multiple sources. |
| **Reranking + relevance gate** | Cohere reranks candidates; an LLM snippet gate drops irrelevant context so the model abstains cleanly on unanswerable questions. |
| **Grounded, cited generation** | Answers cite the snippets they use and synthesize across documents; abstains when context is insufficient. |
| **Financial-document ingestion** | Multi-column PDF parsing, metadata extraction, legal/disclosure boilerplate stripping, structure-aware chunking, and content-hash deduplication. |
| **Conversational** | Follow-ups are rewritten into standalone queries using chat history; per-user conversations persist. |
| **Observability** | Optional LangSmith tracing plus per-answer latency, token, and cost metrics streamed to the UI. |

## Architecture

![System architecture](assets/mermaid%20graphs/new-detailed-flow-graph.png)

## How it works

The system has two paths: an **ingestion** path that prepares documents for retrieval, and a
**query** path that answers questions over them.

### Ingestion

![Ingestion pipeline flow](assets/mermaid%20graphs/Ingestion-pipeline-flow.png)

1. **Parse** — PDFs are converted to markdown with `pymupdf4llm`, which reads multi-column
   layouts in the correct order (analyst reports, papers).
2. **Strip boilerplate** — legal disclosures, licensing watermarks, and jurisdiction/other-entity
   rating tables are removed so they don't crowd out or get mis-cited as real content.
3. **Extract metadata** — company, ticker, document type, date, etc. (all optional) ride on
   every chunk for filtering.
4. **Chunk** — split on markdown headers so each chunk keeps its section context.
5. **Contextualize** — a cheap model writes a 1–2 sentence header situating each chunk in its
   document; this is prepended before embedding and lexical indexing (the single biggest
   retrieval-quality lever). Summarization keeps this within provider rate limits at scale.
6. **Embed & upsert** to Pinecone; duplicate uploads are skipped via a content hash.

### Query

![Query flow](assets/mermaid%20graphs/query-flow.png)

1. **Plan** — resolve the follow-up against history and classify the question. `simple`
   questions take the single-query path; `comparison`/`aggregation` emit focused sub-queries.
2. **Retrieve** — hybrid search (dense + BM25) fused with RRF, filtered per user/document.
3. **Rerank** — Cohere `rerank-v3.5` orders the candidates.
4. **Assemble context** — multi-document questions diversify across source documents (a
   per-document cap ensures several sources reach the generator); simple questions pass through
   an LLM relevance gate that enables clean abstention.
5. **Generate** — the model answers only from the assembled context, cites the snippets it
   uses, synthesizes across documents, and abstains when support is missing. Tokens stream to
   the UI, followed by citations and per-answer metrics (latency, tokens, cost).

## Evaluation & metrics

Quality is measured with a committed harness (`evals/`) that scores retrieval with pure,
un-gameable metrics (hit-rate, MRR, nDCG against ground-truth chunk hashes) and generation with
an LLM-as-judge (a different model from the generator). Reports are saved so every change has a
before/after.

**Retrieval — BEIR SciFact** (public, 300 labelled queries)

| hit-rate@10 | MRR | nDCG@10 |
|---|---|---|
| 0.92 | 0.77 | 0.80 |

**Generation — public golden set** (SciFact-derived, LLM-judged)

| faithfulness | answer relevance | context relevance | abstention accuracy |
|---|---|---|---|
| ~1.00 | 0.98 | 0.99 | 0.90 |

**Domain evaluation — financial analyst reports** (16 curated multi-document questions:
comparisons, aggregations, cross-entity, and unanswerable)

| citation validity | source coverage | abstention accuracy | faithfulness | answer relevance |
|---|---|---|---|---|
| 1.00 | 0.85 | 0.94 | ~0.80 | 0.95 |

*Citation validity* = every inline citation resolves to a real source and abstentions carry
none; *source coverage* = fraction of the documents a question should draw on that were actually
cited. The domain set is small and intentionally hard (cross-document synthesis on real
reports); it exists to catch domain failures that a synthetic benchmark hides.

## Tech stack

- **Retrieval & orchestration** — LangChain, Pinecone (vector), in-memory BM25 (`rank_bm25`),
  Cohere reranker
- **Models** — OpenAI (embeddings + generation), optional Gemini
- **Backend** — FastAPI + SSE streaming, Python 3.10+
- **Frontend** — Next.js, Base UI, Tailwind
- **Data** — Supabase (auth + Postgres for conversations/documents)
- **Parsing** — `pymupdf4llm` (multi-column PDF → markdown)
- **Observability** — LangSmith (optional), per-answer usage/cost metrics

## Project structure

```
api/                    FastAPI app — auth, SSE /stream, /ingest, conversations, documents
src/
  ingestion/            parsing, boilerplate stripping, metadata, chunking, contextualizer, embeddings, Pinecone
  retrieval/            hybrid (BM25 + RRF), query planner (nlu), snippet gate, hashing
  reranking/            Cohere reranker
  generation/           grounded generation, citations, abstention
  rag_pipeline.py       the canonical production pipeline (what ships + what evals run)
  observability.py      token/cost accounting
  storage/              Supabase persistence
evals/                  metrics, LLM judges, runner, golden sets — the scoreboard
frontend/               Next.js app (chat, ingest, conversation history)
supabase/migrations/    SQL schema (conversations, messages, documents) + RLS
scripts/                one-off ops (apply_migration)
docs/                   design notes and measured results per change
config.py               central configuration
```

## Getting started

### Prerequisites
- Python 3.10+, Node 18+ (with `pnpm`)
- API keys: **OpenAI**, **Cohere**, **Pinecone**; a **Supabase** project

### Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the repo root:

```bash
OPENAI_API_KEY=...
COHERE_API_KEY=...
PINECONE_API_KEY=...
NEXT_PUBLIC_SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SECRET_KEY=...              # service role — server-side only
DATABASE_POOLED_URL=...              # for migrations only
# optional: GEMINI_API_KEY, LANGCHAIN_TRACING_V2=true, LANGSMITH_API_KEY
```

Apply the database schema, then run the API:

```bash
python -m scripts.apply_migration
uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
pnpm install
```

Create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=https://<project>.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=...
```

```bash
pnpm dev   # http://localhost:3000
```

Sign in, upload documents in the **Ingest** tab, then ask questions in **Chat**.

### Evaluation

```bash
# retrieval only (fast, no LLM cost)
python -m evals.evaluate --index <index> --golden data/beir_scifact.jsonl --no-judge
# full scoreboard (retrieval + generation, LLM-judged)
python -m evals.evaluate --index <index> --golden data/gen_golden.jsonl --pipeline production
```

## Configuration

Key toggles (all in `config.py`, overridable via environment):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_GENERATION_MODEL` | `gpt-5.4-nano` | Generation model (cost-efficient; retrieval does the heavy lifting) |
| `CONTEXTUAL_RETRIEVAL` | `true` | Prepend a situating header to each chunk at ingest |
| `STRIP_BOILERPLATE` | `true` | Remove legal/disclosure noise from documents before chunking |
| `RAG_MULTI_QUERY` | `true` | Enable query planning + multi-document diversity retrieval |
| `RAG_PER_DOC_CAP` | `3` | Max chunks from any one document in a multi-document answer |
| `VECTOR_TOP_K` / `FINAL_TOP_K` | `10` / `5` | Retrieval and post-rerank context sizes |

---

The previous version of this project (Streamlit prototype) is preserved as
[`README-old.md`](README-old.md).
