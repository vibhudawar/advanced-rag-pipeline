"""Contextual Retrieval (Win 1).

For each chunk, generate a 1-2 sentence header that situates it within its source document,
then prepend it to the chunk *before embedding and BM25-indexing*. A chunk retrieved in
isolation then still carries "where am I from / what is this about", which is the single
biggest retrieval-quality lever (Anthropic's Contextual Retrieval: ~35% fewer retrieval
failures with contextual embeddings, ~49% adding contextual BM25 — and we index the header in
both halves).

Design:
- The whole document is sent as a stable *prefix* per chunk prompt, so OpenAI prompt-caching
  makes the repeated doc cheap; header generation uses a cheap model (config.CONTEXT_MODEL).
- Very long docs are summarised once (CONTEXT_MAX_DOC_CHARS) and the summary is used as context.
- Chunks are contextualised concurrently via LangChain `.batch`.
- Graceful: on any error we return the chunks unchanged (no header) so ingestion never fails.

Output: the same chunk dicts, with `text` = "<header>\n\n<chunk>" (embedded + BM25-indexed),
`metadata.raw_text` = the original chunk (for clean citations), `metadata.context` = header.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("rag.ingest.context")

_CONTEXT_PROMPT = """<document>
{doc}
</document>

Here is a chunk taken from that document:
<chunk>
{chunk}
</chunk>

Give a short, 1-2 sentence context that situates this chunk within the document (which
section/topic it belongs to and what it covers) to improve search retrieval of the chunk.
Answer with ONLY the context, no preamble."""

_SUMMARY_PROMPT = (
    "Summarise the following document in ~200 words, preserving section names, key entities, "
    "dates, and figures so it can situate excerpts:\n\n{doc}"
)


def _text(resp) -> str:
    return (resp.content if hasattr(resp, "content") else str(resp)).strip()


def contextualize_chunks(chunks: list[dict], doc_text: str, model: str | None = None,
                         max_doc_chars: int | None = None) -> list[dict]:
    """Prepend a situating header to each chunk. Returns chunks unchanged on any failure."""
    if not chunks:
        return chunks

    from config import CONTEXT_MAX_DOC_CHARS, CONTEXT_MODEL, OPENAI_API_KEY
    if not OPENAI_API_KEY:
        return chunks

    model = model or CONTEXT_MODEL
    cap = max_doc_chars or CONTEXT_MAX_DOC_CHARS

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=0.0, openai_api_key=OPENAI_API_KEY)

        # For long docs, situate against a one-shot summary instead of the full text.
        doc = doc_text
        if len(doc) > cap:
            doc = _text(llm.invoke(_SUMMARY_PROMPT.format(doc=doc_text[:cap * 2])))

        prompts = [_CONTEXT_PROMPT.format(doc=doc, chunk=c["text"]) for c in chunks]
        responses = llm.batch(prompts, config={"max_concurrency": 8})
    except Exception:
        logger.exception("contextualization failed; ingesting chunks without context headers")
        return chunks

    out: list[dict] = []
    for chunk, resp in zip(chunks, responses):
        header = ""
        try:
            header = _text(resp)
        except Exception:
            header = ""
        raw = chunk["text"]
        meta = {**chunk.get("metadata", {}), "raw_text": raw}
        if header:
            meta["context"] = header
        out.append({"text": f"{header}\n\n{raw}" if header else raw, "metadata": meta})
    return out
