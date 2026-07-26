"""Grounded generation with inline citations and disciplined abstention (WIN 4).

Two rules the baseline RAG prompt didn't enforce:
  1. Cite the snippet number(s) used, inline, so every claim is traceable.
  2. If the (gated) context doesn't support an answer, say so explicitly instead of
     answering from parametric knowledge.

When the snippet gate returns nothing, we don't even call the LLM — we return the abstention
message directly. That guarantees correct "I don't know" behaviour on unanswerable questions
and saves a call.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from langchain_core.documents import Document

ABSTENTION = "I don't have enough information in the provided documents to answer that."

GROUNDED_SYSTEM_PROMPT = (
    "Answer the question using ONLY the numbered context snippets below.\n"
    "- Cite the snippet number(s) you use inline, e.g. [1] or [2].\n"
    "- Do NOT use outside knowledge.\n"
    f'- If the context does not contain enough information, reply exactly: "{ABSTENTION}"\n'
)

# recent conversation turns to include for follow-up context
_HISTORY_TURNS = 6


def _build_message(query: str, docs: list[Document],
                   history: Sequence[tuple[str, str]] | None = None) -> str:
    context = "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))
    hist = ""
    if history:
        recent = "\n".join(f"{role}: {content}" for role, content in history[-_HISTORY_TURNS:])
        hist = f"Conversation so far:\n{recent}\n\n"
    return f"{GROUNDED_SYSTEM_PROMPT}\n\n{hist}Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def generate_grounded(llm, query: str, docs: list[Document],
                      history: Sequence[tuple[str, str]] | None = None) -> str:
    """Generate a cited, grounded answer — or abstain if there is no supporting context."""
    if not docs:
        return ABSTENTION
    response = llm.invoke(_build_message(query, docs, history))
    return response.content if hasattr(response, "content") else str(response)


def stream_grounded(llm, query: str, docs: list[Document],
                    history: Sequence[tuple[str, str]] | None = None) -> Iterator[str]:
    """Stream a cited, grounded answer token-by-token — or yield the abstention once."""
    if not docs:
        yield ABSTENTION
        return
    for chunk in llm.stream(_build_message(query, docs, history)):
        text = chunk.content if hasattr(chunk, "content") else str(chunk)
        if text:
            yield text
