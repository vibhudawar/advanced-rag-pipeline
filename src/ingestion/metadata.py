"""Document-level metadata extraction for financial docs (Phase 1).

Best-effort and fully optional: extraction NEVER fails ingestion. It combines two cheap
sources — a deterministic filename parse (your convention encodes date/source/ticker) and one
LLM call to fill the rest (company, fiscal period, rating, topics) — then drops any null/empty
field so Pinecone (which rejects null metadata values) stays happy. Query-time filtering is
soft, so a document missing a field is never wrongly excluded.

Filename conventions handled:
  "YYYYMMDD - <Source> - <TICKER> - <Title> - N pages.pdf"
  "<TICKER> - ExpertCall_document_dated_DD-MM-YYYY.pdf"
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger("rag.ingest.metadata")


class DocMetadata(BaseModel):
    company: str | None = Field(None, description="Primary company discussed (full name).")
    ticker: str | None = Field(None, description="Stock ticker in uppercase, e.g. META.")
    doc_type: str | None = Field(
        None, description="One of: research_report, earnings_call, expert_call, article, other."
    )
    publisher: str | None = Field(None, description="Publisher / analyst firm, e.g. JPMorgan.")
    published_date: str | None = Field(None, description="Publication date as ISO YYYY-MM-DD.")
    fiscal_period: str | None = Field(None, description="Fiscal period if stated, e.g. 2Q26, FY25.")
    rating: str | None = Field(None, description="Analyst rating if present: BUY, HOLD, or SELL.")
    topics: list[str] = Field(default_factory=list, description="3-6 key themes, e.g. AI, capex.")


_FN_FULL = re.compile(
    r"^(\d{8})\s*-\s*(.+?)\s*-\s*([A-Z]{1,6})\s*-\s*(.+?)\s*-\s*\d+\s*pages", re.IGNORECASE
)
_FN_EXPERT = re.compile(r"^([A-Z]{1,6})\s*-\s*ExpertCall.*?(\d{2}-\d{2}-\d{4})", re.IGNORECASE)


def _doc_type_from_name(name: str) -> str | None:
    low = name.lower()
    if "expertcall" in low or "expert call" in low:
        return "expert_call"
    if "earnings" in low:
        return "earnings_call"
    if "research" in low or "report" in low:
        return "research_report"
    return None


def _parse_filename(filename: str) -> dict:
    """Deterministic, reliable fields from the filename. Missing keys are simply omitted."""
    m = _FN_FULL.match(filename)
    if m:
        d, publisher, ticker, _title = m.groups()
        out = {"published_date": f"{d[:4]}-{d[4:6]}-{d[6:8]}",
               "publisher": publisher.strip(), "ticker": ticker.upper()}
        # A firm-published note in this pattern is an analyst research report unless the name
        # says otherwise (expert call / earnings).
        out["doc_type"] = _doc_type_from_name(filename) or "research_report"
        return out
    m = _FN_EXPERT.match(filename)
    if m:
        ticker, ddmmyyyy = m.groups()
        day, mon, yr = ddmmyyyy.split("-")
        return {"ticker": ticker.upper(), "doc_type": "expert_call",
                "published_date": f"{yr}-{mon}-{day}"}
    return {}


_PROMPT = (
    "Extract metadata from this financial document. Use ONLY what the text states; leave any "
    "field null if it is not present — do not guess.\n\n"
    "Filename: {filename}\n\nDocument (beginning):\n{head}"
)


def extract_metadata(filename: str, text: str) -> dict:
    """Return a flat, Pinecone-safe metadata dict (only non-null keys). Best-effort."""
    from_name = _parse_filename(filename)

    from_llm: dict = {}
    try:
        from config import CONTEXT_MODEL, OPENAI_API_KEY
        if OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            chain = ChatOpenAI(model=CONTEXT_MODEL, temperature=0.0,
                               openai_api_key=OPENAI_API_KEY).with_structured_output(DocMetadata)
            result = chain.invoke(_PROMPT.format(filename=filename, head=text[:3000]))
            from_llm = result.model_dump()
    except Exception:
        logger.exception("metadata LLM extraction failed; falling back to filename only")

    # Filename fields are reliable → they win over the LLM for the keys they cover.
    merged = {**from_llm, **from_name}
    # Drop null/empty so Pinecone accepts the metadata (it rejects null values).
    return {k: v for k, v in merged.items() if v not in (None, "", [])}
