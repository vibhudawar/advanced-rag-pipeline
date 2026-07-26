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

from langchain_core.documents import Document

ABSTENTION = "I don't have enough information in the provided documents to answer that."

GROUNDED_SYSTEM_PROMPT = (
    "Answer the question using ONLY the numbered context snippets below.\n"
    "- Cite the snippet number(s) you use inline, e.g. [1] or [2].\n"
    "- Do NOT use outside knowledge.\n"
    f'- If the context does not contain enough information, reply exactly: "{ABSTENTION}"\n'
)


def generate_grounded(llm, query: str, docs: list[Document]) -> str:
    """Generate a cited, grounded answer — or abstain if there is no supporting context."""
    if not docs:
        return ABSTENTION
    context = "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))
    message = f"{GROUNDED_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm.invoke(message)
    return response.content if hasattr(response, "content") else str(response)
