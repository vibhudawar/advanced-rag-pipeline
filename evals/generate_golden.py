"""Generate a golden set from your own Pinecone corpus.

    python -m evals.generate_golden --index <name> [--n-answerable 100] [--n-unanswerable 20]

Each answerable item is grounded in one sampled chunk, whose content hash becomes the
retrieval ground truth. Unanswerable items are questions in the corpus's domain that the
sampled context does NOT answer — they test the "I don't know" behaviour.

IMPORTANT: synthetic items are marked needs_review=True. They phrase questions the way
the source text is written, which makes retrieval look artificially easy. Curate ~50 of
them by hand (fix leaks, rewrite in real-user phrasing) before trusting the numbers — see
plan-v2.md WIN 1. This script bootstraps volume; it does not replace human curation.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import cast

from pydantic import BaseModel, Field

from config import EMBEDDING_PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY
from src.ingestion.DBIngestion import get_vector_store

from .schema import GoldenItem, QType, content_hash, write_jsonl

TEXT_KEY = "text"  # LangChain Pinecone stores chunk content under metadata["text"]


class _GenQA(BaseModel):
    query: str = Field(description="A realistic, standalone user question this passage fully answers. No 'this document'/'the passage'.")
    reference_answer: str = Field(description="The correct, concise answer, grounded only in the passage.")
    q_type: str = Field(description="One of: factoid, multi_hop, aggregation.")


class _GenUnanswerable(BaseModel):
    query: str = Field(description="A realistic question in the same domain that CANNOT be answered from the given text.")


def _make_gen_llm():
    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.4, openai_api_key=OPENAI_API_KEY)
    if GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, google_api_key=GEMINI_API_KEY)
    raise RuntimeError("No generator key found (set OPENAI_API_KEY or GEMINI_API_KEY).")


def _sample_chunks(index_name: str, pool: int) -> list[str]:
    """Return up to `pool` chunk texts sampled from the index."""
    from src.ingestion.EmbeddingCreator import get_embedder
    vs = get_vector_store()
    index = vs.pc.Index(index_name)
    dim = get_embedder(provider=EMBEDDING_PROVIDER).get_embedding_dimension()
    texts: list[str] = []

    # Preferred: list ids then fetch (serverless supports index.list).
    try:
        ids: list[str] = []
        for page in index.list(limit=min(pool, 100)):
            ids.extend(page if isinstance(page, list) else [page])
            if len(ids) >= pool:
                break
        ids = ids[:pool]
        for i in range(0, len(ids), 100):
            resp = index.fetch(ids=ids[i:i + 100])
            vectors = getattr(resp, "vectors", None) or resp.get("vectors", {})
            for v in vectors.values():
                md = getattr(v, "metadata", None) or v.get("metadata", {})
                if md and md.get(TEXT_KEY):
                    texts.append(md[TEXT_KEY])
    except Exception as e:  # noqa: BLE001 - fall back to random-vector query
        print(f"[sample] index.list unavailable ({e}); falling back to random-vector query")
        seen = set()
        for _ in range(max(1, pool // 50)):
            vec = [random.uniform(-1, 1) for _ in range(dim)]
            res = index.query(vector=vec, top_k=min(100, pool), include_metadata=True)
            for m in res.matches:
                md = m.metadata or {}
                if md.get(TEXT_KEY) and md[TEXT_KEY] not in seen:
                    seen.add(md[TEXT_KEY])
                    texts.append(md[TEXT_KEY])

    random.shuffle(texts)
    return texts


def generate(index_name: str, n_answerable: int, n_unanswerable: int, out: str) -> None:
    pool = max(n_answerable * 2, n_answerable + 50)
    chunks = _sample_chunks(index_name, pool)
    if not chunks:
        raise SystemExit(
            f"No chunks with a '{TEXT_KEY}' field found in index '{index_name}'. "
            "Is the index populated (ingest documents first)?"
        )
    print(f"[sample] pulled {len(chunks)} chunks from '{index_name}'")

    llm = _make_gen_llm()
    qa_chain = llm.with_structured_output(_GenQA)
    un_chain = llm.with_structured_output(_GenUnanswerable)

    items: list[GoldenItem] = []

    for i, text in enumerate(chunks[:n_answerable]):
        try:
            qa = qa_chain.invoke(
                "Write one realistic user question that the passage below FULLY answers, plus the "
                "correct concise answer. The question must stand alone (do not say 'this passage').\n\n"
                f"PASSAGE:\n{text}"
            )
            raw_type = qa.q_type if qa.q_type in {"factoid", "multi_hop", "aggregation"} else "factoid"
            items.append(GoldenItem(
                id=f"syn-{i:04d}", query=qa.query, reference_answer=qa.reference_answer,
                relevant_chunk_hashes=[content_hash(text)], q_type=cast(QType, raw_type), source="synthetic",
            ))
        except Exception as e:  # noqa: BLE001
            print(f"[gen] skipped chunk {i}: {e}")

    for j, text in enumerate(chunks[n_answerable:n_answerable + n_unanswerable]):
        try:
            un = un_chain.invoke(
                "Based on the domain of the text below, write one realistic question that a user "
                "might ask but that CANNOT be answered from this text (it asks about something the "
                "text does not cover).\n\n"
                f"TEXT:\n{text}"
            )
            items.append(GoldenItem(
                id=f"un-{j:04d}", query=un.query, reference_answer="",
                relevant_chunk_hashes=[], q_type="unanswerable", source="synthetic",
            ))
        except Exception as e:  # noqa: BLE001
            print(f"[gen] skipped unanswerable {j}: {e}")

    n = write_jsonl(items, out)
    answerable = sum(1 for it in items if it.q_type != "unanswerable")
    print(f"[done] wrote {n} items to {out} ({answerable} answerable, {n - answerable} unanswerable)")
    print("[next] curate ~50 by hand (set needs_review=false) before trusting scores.")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a golden set from your Pinecone corpus.")
    p.add_argument("--index", default=os.getenv("EVAL_INDEX"), help="Pinecone index name (or set EVAL_INDEX)")
    p.add_argument("--n-answerable", type=int, default=100)
    p.add_argument("--n-unanswerable", type=int, default=20)
    p.add_argument("--out", default="data/golden.jsonl")
    args = p.parse_args()
    if not args.index:
        raise SystemExit("Provide --index <name> (or set EVAL_INDEX in your environment).")
    generate(args.index, args.n_answerable, args.n_unanswerable, args.out)


if __name__ == "__main__":
    main()
