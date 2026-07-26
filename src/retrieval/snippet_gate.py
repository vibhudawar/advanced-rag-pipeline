"""LLM snippet-relevance gate (WIN 4).

After retrieval + reranking, the top chunks are *ranked* but not *filtered* — a reranker
still returns its best-k even when none actually answer the question. This gate asks an LLM,
in one batched call, which of the candidate snippets genuinely contain information relevant
to the question, and drops the rest.

Why it matters: if nothing survives the gate, the pipeline has no grounded context and
should abstain ("I don't know") rather than confabulate from marginally-related text. That
is the mechanism that turns the retrieval-always-returns-something behaviour into disciplined
non-answers on unanswerable questions.
"""

from __future__ import annotations

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, OPENAI_API_KEY


class _Relevance(BaseModel):
    relevant_indices: list[int] = Field(
        description="Indices of snippets that contain information helping answer the question. Empty if none do."
    )


def _make_llm(model: str | None = None):
    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
    if GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash", temperature=0.0, google_api_key=GEMINI_API_KEY)
    raise RuntimeError("No LLM key for the snippet gate (set OPENAI_API_KEY or GEMINI_API_KEY).")


class SnippetGate:
    """Filters reranked snippets down to those actually relevant to the question."""

    def __init__(self, model: str | None = None, max_chars: int = 800):
        self._chain = _make_llm(model).with_structured_output(_Relevance)
        self.max_chars = max_chars

    def filter(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        listing = "\n\n".join(f"[{i}] {d.page_content[:self.max_chars]}" for i, d in enumerate(docs))
        result = self._chain.invoke(
            "You filter retrieved snippets for a RAG system. Return the indices of the snippets "
            "that contain information genuinely useful for answering the QUESTION. Be strict: if a "
            "snippet is only loosely related and does not help answer it, exclude it. If none help, "
            f"return an empty list.\n\nQUESTION: {query}\n\nSNIPPETS:\n{listing}"
        )
        keep = {i for i in result.relevant_indices if 0 <= i < len(docs)}
        return [d for i, d in enumerate(docs) if i in keep]
