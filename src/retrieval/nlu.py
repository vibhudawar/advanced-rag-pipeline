"""Query understanding (NLU): restate -> classify -> generate structured search params.

Turns a raw user question into:
  - a self-contained **restatement** (pronouns/references resolved from conversation history),
  - an **intent** (greeting / off_topic / rag_query) for routing,
  - an **is_complex** flag (feeds adaptive depth later),
  - TWO query styles: short **keyword_queries** for the BM25/lexical retriever and one
    natural-language **semantic_query** for the dense retriever,
  - optional **metadata filters** (entities / doc_types / date range).

Why this replaces naive multi-paraphrase expansion: paraphrases only add recall on the
vector side. They don't help lexical search or enable filtering. The keyword/semantic split
is what makes hybrid retrieval (WIN 2) actually pay off, and the structured output kills the
fragile newline+regex parsing the old MQE prompt relied on.

Output is validated by Pydantic via `with_structured_output`, so the model must return the
right shape or LangChain retries — no manual parsing.
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, OPENAI_API_KEY


class QueryUnderstanding(BaseModel):
    intent: str = Field(description="One of: greeting, off_topic, rag_query.")
    restated_query: str = Field(description="Self-contained rewrite; resolve references from history, preserve all user constraints verbatim.")
    is_complex: bool = Field(default=False, description="True only if the query needs multi-step/multi-source research.")
    keyword_queries: list[str] = Field(default_factory=list, description="2-4 short keyword queries (3-10 words, include named entities) for lexical/BM25 search.")
    semantic_query: str = Field(description="One natural-language query capturing full intent, for dense/vector search.")
    entities: list[str] = Field(default_factory=list, description="Named entities mentioned (people, orgs, products).")
    doc_types: list[str] = Field(default_factory=list, description="Document types the user restricted to, if any (e.g. pdf, 10-K).")
    date_start: str | None = Field(default=None, description="ISO start date if a time range is implied, else null.")
    date_end: str | None = Field(default=None, description="ISO end date if a time range is implied, else null.")


_PROMPT = """You convert a user question into structured search parameters for a hybrid
(keyword + semantic) RAG system. Today's date is {today}.

Conversation history (may be empty):
{history}

User question:
{query}

Do the following:
1. RESTATE: rewrite the question to be fully self-contained. Resolve pronouns/references
   using the history. Preserve every user constraint (source, time, scope) verbatim. If it is
   already self-contained, keep it unchanged.
2. CLASSIFY intent:
   - greeting: pleasantries, thanks, or chit-chat with no information request.
   - off_topic: not an information-seeking question about the document knowledge base — e.g.
     account/billing/subscription/password/support or other transactional requests, or actions
     the system cannot perform. These should be routed to support, not to document search.
   - rag_query: a genuine question to be answered from the documents.
   Also set is_complex only for genuinely multi-step / multi-source questions.
3. GENERATE (scoped strictly to the restated query — do NOT broaden):
   - keyword_queries: 2-4 short queries (3-10 words), key terms only, each including any named
     entity; independent of each other.
   - semantic_query: one natural-language query capturing the full intent.
4. EXTRACT filters only if explicitly present: entities, doc_types, and a date range. Resolve
   relative dates (e.g. "last quarter") against today's date to ISO; add no dates otherwise."""


def _make_llm(model: str | None = None):
    """Prefer OpenAI (the working provider); fall back to Gemini."""
    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
    if GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash", temperature=0.0, google_api_key=GEMINI_API_KEY)
    raise RuntimeError("No LLM key for NLU (set OPENAI_API_KEY or GEMINI_API_KEY).")


class QueryUnderstander:
    """Stateless understander: one structured LLM call per query."""

    def __init__(self, model: str | None = None, today: str | None = None):
        self._chain = _make_llm(model).with_structured_output(QueryUnderstanding)
        self.today = today or datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat()

    def understand(self, query: str, history: str = "") -> QueryUnderstanding:
        return self._chain.invoke(
            _PROMPT.format(today=self.today, history=history or "(none)", query=query)
        )
