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
import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, OPENAI_API_KEY

logger = logging.getLogger("rag.retrieval.nlu")

# Cap the fan-out. Each sub-query is a separate retrieval pass (vector + BM25 + rerank), so
# this bounds cost for comparison/aggregation questions. 4 covers "compare A vs B vs C" and
# most multi-entity asks; beyond that, extra sub-queries add cost without much recall.
MAX_SUB_QUERIES = 4

_CONDENSE_PROMPT = """Given a conversation and a follow-up message, rewrite the follow-up into a \
standalone search query for retrieving from the user's documents.
- Resolve references ("it", "that", "instead", "more") using the conversation.
- Keep named entities and any constraints.
- Do NOT broaden or add topics; capture only what the user is asking for.
- If the follow-up is already standalone, return it unchanged.
Return ONLY the rewritten query, no preamble.

Conversation:
{history}

Follow-up: {query}

Standalone search query:"""


def condense_query(llm, query: str, history: Sequence[tuple[str, str]] | None) -> str:
    """Rewrite a conversational follow-up into a standalone retrieval query using history.

    Returns `query` unchanged when there's no history (first turn / evals) — so standalone
    queries are never broadened (which regressed the benchmark). One cheap LLM call otherwise;
    falls back to the original query on any error.
    """
    if not history:
        return query
    hist = "\n".join(f"{role}: {content}" for role, content in list(history)[-6:])
    try:
        resp = llm.invoke(_CONDENSE_PROMPT.format(history=hist, query=query))
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip().strip('"')
        return text or query
    except Exception:  # noqa: BLE001 - best-effort rewrite; fall back to the original query
        return query


class QueryPlan(BaseModel):
    """Routing + decomposition decision for a single user turn (Phase 2).

    One structured LLM call replaces the plain `condense_query` rewrite: it resolves the
    follow-up into a standalone query AND decides how to retrieve for it.
      - `simple`      : one document / one fact -> retrieve on `standalone_query` alone
                        (`sub_queries` empty). Identical to the pre-Phase-2 path.
      - `comparison`  : contrast entities/documents ("A vs B in AI") -> one focused
                        sub-query per side, retrieved separately so BOTH sides reach the
                        generator instead of the reranker collapsing onto the stronger match.
      - `aggregation` : span many documents ("which companies mention X") -> a sub-query per
                        facet/entity so coverage is broad, not top-k-of-one.
    """

    standalone_query: str = Field(
        description="The user's question rewritten to be fully self-contained (references resolved from history). Unchanged if already standalone."
    )
    query_type: str = Field(
        default="simple",
        description="One of: simple, comparison, aggregation. Use simple unless the question genuinely contrasts or spans multiple entities/documents.",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="For comparison/aggregation ONLY: 2-4 focused retrieval queries, one per entity/facet, each naming its entity. Empty for simple questions.",
    )


_PLAN_PROMPT = """You plan retrieval for a RAG system over a document corpus. Given the \
conversation and the user's latest message, produce a retrieval plan.

1. STANDALONE: rewrite the latest message into a self-contained query. Resolve references
   ("it", "that", "instead") using the conversation. Keep named entities and constraints. If
   it is already standalone, return it unchanged. Do NOT broaden or add topics.
2. CLASSIFY query_type:
   - simple: a single fact or a question about one entity/document.
   - comparison: explicitly contrasts two or more entities/documents (e.g. "how does A's
     strategy compare to B's", "A vs B").
   - aggregation: asks across many documents (e.g. "which companies mention X", "summarize
     what all the reports say about Y").
3. DECOMPOSE (comparison/aggregation only): produce 2-{max_sub} focused sub-queries, one per
   entity/facet, each naming its entity explicitly. For simple, return an empty list.

Conversation:
{history}

Latest message: {query}"""


def plan_query(llm, query: str, history: Sequence[tuple[str, str]] | None) -> QueryPlan:
    """Route + (optionally) decompose a user turn in one structured LLM call.

    Falls back to a `simple` plan on the original query if the call fails or returns junk —
    the pipeline then behaves exactly as it did pre-Phase-2. Sub-queries are de-duplicated and
    capped at MAX_SUB_QUERIES to bound retrieval cost.
    """
    hist = "\n".join(f"{role}: {content}" for role, content in list(history or [])[-6:]) or "(none)"
    try:
        chain = llm.with_structured_output(QueryPlan)
        plan: QueryPlan = chain.invoke(
            _PLAN_PROMPT.format(history=hist, query=query, max_sub=MAX_SUB_QUERIES)
        )
    except Exception:  # noqa: BLE001 - best-effort planning; degrade to a simple single-query plan
        logger.info("query planning failed; falling back to simple plan")
        return QueryPlan(standalone_query=query, query_type="simple", sub_queries=[])

    standalone = (plan.standalone_query or query).strip() or query
    qtype = plan.query_type if plan.query_type in {"simple", "comparison", "aggregation"} else "simple"
    if qtype == "simple":
        return QueryPlan(standalone_query=standalone, query_type="simple", sub_queries=[])

    # Dedup sub-queries (case-insensitive), drop empties, cap the fan-out. If decomposition
    # produced nothing usable, treat it as simple rather than fanning out on the whole query.
    seen: set[str] = set()
    subs: list[str] = []
    for s in plan.sub_queries:
        s = (s or "").strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            subs.append(s)
        if len(subs) >= MAX_SUB_QUERIES:
            break
    if not subs:
        return QueryPlan(standalone_query=standalone, query_type="simple", sub_queries=[])
    return QueryPlan(standalone_query=standalone, query_type=qtype, sub_queries=subs)


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
