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
- Because that prefix is resent per chunk, input tokens ≈ doc_tokens × num_chunks. When that
  product would exceed CONTEXT_TOKEN_BUDGET (or the doc is very long), we situate chunks against
  a one-shot ~200-word summary instead — bounded regardless of chunk count. Small/few-chunk docs
  keep the richer full-doc context. This is what keeps ingestion under OpenAI's 200k TPM limit.
- Requests retry with backoff on 429s (CONTEXT_MAX_RETRIES), and the batch returns exceptions
  per-chunk, so a single failed request costs one header — not the whole document's.
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
    """Prepend a situating header to each chunk. Returns chunks unchanged on total failure;
    individual chunk failures just yield that chunk without a header (per-chunk graceful)."""
    if not chunks:
        return chunks

    from config import (
        CONTEXT_MAX_DOC_CHARS,
        CONTEXT_MAX_RETRIES,
        CONTEXT_MODEL,
        CONTEXT_TOKEN_BUDGET,
        OPENAI_API_KEY,
    )
    if not OPENAI_API_KEY:
        return chunks

    model = model or CONTEXT_MODEL
    cap = max_doc_chars or CONTEXT_MAX_DOC_CHARS

    try:
        from langchain_openai import ChatOpenAI
        # max_retries lets the SDK back off and retry on 429s instead of failing the doc.
        llm = ChatOpenAI(model=model, temperature=0.0, openai_api_key=OPENAI_API_KEY,
                         max_retries=CONTEXT_MAX_RETRIES)

        # The doc prefix is resent per chunk, so cost ≈ (doc chars/4) × num_chunks tokens.
        # Summarise once when that would blow the budget (or the doc is very long); otherwise
        # keep the full doc for richer context. `~4 chars/token` is a rough but safe estimate.
        est_prefix_tokens = len(doc_text) / 4
        blowup = est_prefix_tokens * len(chunks)
        doc = doc_text
        if len(doc_text) > cap or blowup > CONTEXT_TOKEN_BUDGET:
            doc = _text(llm.invoke(_SUMMARY_PROMPT.format(doc=doc_text[:cap * 2])))

        prompts = [_CONTEXT_PROMPT.format(doc=doc, chunk=c["text"]) for c in chunks]
        # return_exceptions: one failed request costs one header, not the whole document.
        responses = llm.batch(prompts, config={"max_concurrency": 8}, return_exceptions=True)
    except Exception:
        logger.exception("contextualization failed; ingesting chunks without context headers")
        return chunks

    out: list[dict] = []
    n_failed = 0
    for chunk, resp in zip(chunks, responses):
        header = ""
        if not isinstance(resp, BaseException):
            try:
                header = _text(resp)
            except Exception:  # noqa: BLE001 - a malformed response just means no header
                header = ""
        if not header:
            n_failed += 1
        raw = chunk["text"]
        meta = {**chunk.get("metadata", {}), "raw_text": raw}
        if header:
            meta["context"] = header
        out.append({"text": f"{header}\n\n{raw}" if header else raw, "metadata": meta})
    if n_failed:
        logger.info("contextualization: %d/%d chunks without a header", n_failed, len(chunks))
    return out
