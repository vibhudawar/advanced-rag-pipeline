"""Canonical production RAG pipeline — the single implementation that ships.

Consolidates the measured wins into one place:
  hybrid retrieval (vector + BM25, RRF)  [WIN 2]
    -> Cohere rerank
    -> LLM snippet-relevance gate          [WIN 4]
    -> grounded, cited generation + abstention  [WIN 4]

The FastAPI service (api/main.py) serves this, and the eval harness wraps it as the
`production` pipeline — so evals measure exactly what ships (no parallel implementation to
drift). Conversation history is accepted and threaded into generation; durable memory
(LangGraph checkpointer + Supabase) is wired in WIN 7b.

NOTE: BM25 is built in-memory from the whole index at construction. Fine for the current
corpora; for large corpora move the lexical half into the search engine (see hybrid.py).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.documents import Document
from langsmith import traceable

from config import (
    EMBEDDING_PROVIDER,
    LLM_PROVIDER,
    RAG_MULTI_FINAL_K,
    RAG_MULTI_QUERY,
    RAG_PER_DOC_CAP,
    RAG_UNION_RERANK_K,
)
from src.generation.grounded import generate_grounded, is_abstention, stream_grounded
from src.generation.llm_generator import get_llm_generator
from src.ingestion.DBIngestion import get_vector_store
from src.ingestion.EmbeddingCreator import get_embedder
from src.observability import summarize_usage
from src.reranking.reranker import get_reranker
from src.retrieval.hashing import content_hash
from src.retrieval.hybrid import BM25Index, reciprocal_rank_fusion
from src.retrieval.nlu import QueryPlan, plan_query
from src.retrieval.snippet_gate import SnippetGate

History = Sequence[tuple[str, str]]  # [(role, content), ...]


@dataclass
class AnswerResult:
    answer: str
    citations: list[dict]
    contexts: list[str]
    candidate_hashes: list[str]  # fused, pre-rerank — for retrieval eval
    latency_s: float = 0.0
    metadata: dict = field(default_factory=dict)


_CORPUS_CACHE_DIR = Path("data/bm25_cache")


def corpus_cache_path(index_name: str) -> Path:
    """Disk cache for the BM25 corpus (v2 carries metadata for filtering)."""
    return _CORPUS_CACHE_DIR / f"{index_name}.v2.json"


def _load_corpus(vector_store, index_name: str, embedder, text_key: str = "text",
                 max_docs: int = 10000) -> list[dict]:
    """Return the BM25 corpus as [{text, user_id, filename, source}], cached to disk after the
    first build. Metadata rides along so the lexical (BM25) half can be filtered per user/
    document, matching what the vector half filters server-side in Pinecone.

    Uses ONE large `query` (top_k=max_docs) instead of paginated list()+fetch() (~11x faster).
    For corpora larger than max_docs this returns only the top_k nearest to a constant probe
    vector — at that scale move lexical search into the engine (see src/retrieval/hybrid.py).
    """
    cache = corpus_cache_path(index_name)
    if cache.exists():
        return json.loads(cache.read_text())
    dim = embedder.get_embedding_dimension()
    index = vector_store.pc.Index(index_name)
    res = index.query(vector=[0.1] * dim, top_k=max_docs, include_metadata=True)
    rows: list[dict] = []
    for m in res.matches:
        md = m.metadata or {}
        text = md.get(text_key)
        if not text:
            continue
        rows.append({"text": text, "user_id": md.get("user_id"),
                     "filename": md.get("filename"), "source": md.get("source")})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def _meta_matches(meta: dict, filter_dict: dict) -> bool:
    """True if `meta` equals every key/value in `filter_dict` (AND semantics)."""
    return all(meta.get(k) == v for k, v in filter_dict.items())


def _doc_source(doc: Document) -> str:
    md = doc.metadata or {}
    return md.get("filename") or md.get("source") or ""


def _diversify(docs: list[Document], per_doc_cap: int, final_k: int) -> list[Document]:
    """Select up to `final_k` docs, round-robin across source documents, so a multi-doc answer
    sees several sources instead of the top-k all coming from one.

    `docs` must already be ranked best-first (post-rerank). Round 0 takes each source's best
    chunk, round 1 its second-best, etc., up to `per_doc_cap` per source. Rank order within a
    source is preserved. Chunks with no identifiable source share the "" bucket.
    """
    from collections import defaultdict

    buckets: dict[str, list[Document]] = defaultdict(list)
    order: list[str] = []
    for d in docs:
        src = _doc_source(d)
        if src not in buckets:
            order.append(src)
        buckets[src].append(d)

    selected: list[Document] = []
    for round_i in range(per_doc_cap):
        for src in order:
            if round_i < len(buckets[src]):
                selected.append(buckets[src][round_i])
                if len(selected) >= final_k:
                    return selected
    return selected


def _citation(doc: Document, n: int) -> dict:
    md = doc.metadata or {}
    # Show the raw chunk (Contextual Retrieval stores it in metadata) rather than the
    # header-prefixed text that's embedded/indexed, so citations stay clean.
    snippet = md.get("raw_text") or doc.page_content
    return {
        "n": n,
        "source": md.get("filename") or md.get("source") or md.get("beir_id") or f"chunk-{n}",
        "snippet": snippet[:300],
        "score": md.get("rerank_score") or md.get("score"),
    }


_CITE_MARKERS = re.compile(r"\[(\d+)\]")


def _build_citations(answer: str, gated: list[Document]) -> list[dict]:
    """Sources for an answer = only the snippets the answer actually cites inline (by ``[n]``).

    The generator sees the whole retrieved context numbered ``[1..N]`` but usually grounds its
    answer in a subset; listing the unused chunks is misleading (a listed 'source' may even
    contradict the answer). We keep each cited snippet's original number so the inline markers
    still resolve. Falls back to the full set only when the answer carries no valid marker, so a
    genuine answer never renders an empty Sources list.
    """
    all_c = [_citation(d, i + 1) for i, d in enumerate(gated)]
    referenced = {int(n) for n in _CITE_MARKERS.findall(answer or "")}
    cited = [c for c in all_c if c["n"] in referenced]
    return cited or all_c


class RagPipeline:
    def __init__(self, index_name: str, top_k: int = 10, rerank_top_k: int = 5,
                 rrf_k: int = 60, use_gate: bool = True, multi_query: bool | None = None,
                 union_rerank_k: int | None = None, per_doc_cap: int | None = None,
                 multi_final_k: int | None = None):
        self.index_name = index_name
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k
        # Phase 2 multi-query knobs (config-driven defaults; overridable for evals).
        self.multi_query = RAG_MULTI_QUERY if multi_query is None else multi_query
        self.union_rerank_k = union_rerank_k or RAG_UNION_RERANK_K
        self.per_doc_cap = per_doc_cap or RAG_PER_DOC_CAP
        self.multi_final_k = multi_final_k or RAG_MULTI_FINAL_K
        self.embedder = get_embedder(provider=EMBEDDING_PROVIDER)
        self.vector_store = get_vector_store()
        self.reranker = get_reranker()
        self.generator = get_llm_generator(provider=LLM_PROVIDER)
        self.gate = SnippetGate() if use_gate else None
        if index_name not in self.vector_store.list_indexes():
            raise ValueError(f"Index '{index_name}' not found.")
        rows = _load_corpus(self.vector_store, index_name, self.embedder)
        texts = [r["text"] for r in rows]
        self.bm25 = BM25Index([content_hash(t) for t in texts], texts)
        # content_hash -> {user_id, filename, source, text} so the lexical half can be filtered.
        self.chunk_meta = {content_hash(r["text"]): r for r in rows}

    def _fused(self, query: str, filter_dict: dict | None = None) -> tuple[list[str], dict[str, Document]]:
        # Vector half: Pinecone filters server-side. Lexical half: filter fused BM25 hits by the
        # metadata we cached, so both halves respect the same scope (user / document).
        vec_docs = self.vector_store.similarity_search(
            index_name=self.index_name, query=query, embedder=self.embedder, top_k=self.top_k,
            filter_dict=filter_dict,
        )
        vec_hashes = [content_hash(d.page_content) for d in vec_docs]
        bm25_hashes = self.bm25.search(query, self.top_k)
        if filter_dict:
            bm25_hashes = [h for h in bm25_hashes
                           if _meta_matches(self.chunk_meta.get(h, {}), filter_dict)]
        fused = reciprocal_rank_fusion([vec_hashes, bm25_hashes], k=self.rrf_k)
        return fused, {content_hash(d.page_content): d for d in vec_docs}

    def candidates(self, query: str) -> list[str]:
        """Fused hashes, pre-rerank — used by the retrieval-only eval."""
        return self._fused(query)[0]

    def _fused_docs(self, query: str, filter_dict: dict | None,
                    limit: int) -> tuple[list[str], list[Document]]:
        """Run hybrid retrieval and materialize the top `limit` fused hits as Documents
        (vector hits carry full metadata; BM25-only hits are rebuilt from the cached corpus)."""
        fused, by_hash = self._fused(query, filter_dict)
        docs: list[Document] = []
        for h in fused[:limit]:
            if h in by_hash:
                docs.append(by_hash[h])
            elif h in self.bm25.texts_by_key:
                meta = self.chunk_meta.get(h, {})
                docs.append(Document(
                    page_content=self.bm25.texts_by_key[h],
                    metadata={k: meta[k] for k in ("filename", "source", "user_id")
                              if meta.get(k)},
                ))
        return fused, docs

    def _retrieve_gated(self, query: str,
                        filter_dict: dict | None = None) -> tuple[list[str], list[Document]]:
        fused, docs = self._fused_docs(query, filter_dict, self.top_k)
        # When the user has scoped to a specific document, trust it: feed more of the doc and
        # skip the relevance gate (the gate is for separating relevant from irrelevant across a
        # mixed corpus — pointless, and harmful for vague asks, once we're inside one chosen doc).
        doc_scoped = bool(filter_dict and filter_dict.get("filename"))
        rerank_k = self.top_k if doc_scoped else self.rerank_top_k
        reranked = self.reranker.rerank(query=query, documents=docs,
                                        top_k=rerank_k) if docs else []
        gated = reranked if (doc_scoped or not self.gate) else self.gate.filter(query, reranked)
        return fused, gated

    def _retrieve_plan(self, plan: QueryPlan,
                       filter_dict: dict | None = None) -> tuple[list[str], list[Document]]:
        """Route retrieval by the query plan. Simple queries (and any doc-scoped request) take
        the single-query gated path unchanged. Comparison/aggregation fan out one retrieval pass
        per sub-query, rerank the deduped union against the overall question, gate it, then
        diversify across source documents so several docs reach the generator for synthesis."""
        doc_scoped = bool(filter_dict and filter_dict.get("filename"))
        if not self.multi_query or plan.query_type == "simple" or not plan.sub_queries or doc_scoped:
            return self._retrieve_gated(plan.standalone_query, filter_dict)

        queries = [plan.standalone_query, *plan.sub_queries]
        fused_all: list[str] = []
        union: dict[str, Document] = {}
        for q in queries:
            fused, docs = self._fused_docs(q, filter_dict, self.top_k)
            fused_all.extend(fused)
            for d in docs:
                union.setdefault(content_hash(d.page_content), d)
        if not union:
            return fused_all, []

        # Rerank the union against the overall question (a chunk about either side is relevant to
        # a comparison), then spread the survivors across source documents. We deliberately SKIP
        # the LLM snippet gate here: it judges each chunk against the umbrella question, and a
        # single chunk about one entity rarely looks like it "answers the comparison" on its own,
        # so the gate collapses the diversified union and forces a false abstention. Relevance is
        # already handled by per-sub-query retrieval + rerank; the generator still abstains via
        # its own grounded-prompt rule if the assembled context genuinely doesn't support an answer.
        reranked = self.reranker.rerank(query=plan.standalone_query,
                                        documents=list(union.values()), top_k=self.union_rerank_k)
        diversified = _diversify(reranked, self.per_doc_cap, self.multi_final_k)
        return fused_all, diversified

    @traceable(name="rag_answer", run_type="chain")
    def answer(self, query: str, history: History | None = None,
               filter_dict: dict | None = None) -> AnswerResult:
        t0 = time.time()
        # get_usage_metadata_callback aggregates token usage across every LLM call in the run
        # (snippet gate + generation) via a contextvar — no need to thread callbacks through.
        with get_usage_metadata_callback() as cb:
            # Plan retrieval (resolve follow-ups + route simple/comparison/aggregation), then
            # retrieve accordingly; generate with the original question + history so the answer
            # still reads naturally.
            plan = plan_query(self.generator.llm, query, history)
            fused, gated = self._retrieve_plan(plan, filter_dict)
            ans = generate_grounded(self.generator.llm, query, gated, history)
            usage = summarize_usage(cb.usage_metadata)
        abstained = is_abstention(ans)
        # No answer → no citations. A source under "I don't know" is contradictory (the gate can
        # pass a topically-adjacent chunk the generator then judges insufficient). Otherwise list
        # only the snippets the answer actually cited, not everything retrieval fed the model.
        citations = [] if abstained else _build_citations(ans, gated)
        return AnswerResult(
            answer=ans,
            citations=citations,
            contexts=[d.page_content for d in gated],
            candidate_hashes=fused,
            latency_s=time.time() - t0,
            metadata={"model": getattr(self.generator, "model_name", None),
                      "n_context": len(gated), "abstained": abstained,
                      "query_type": plan.query_type, "n_subqueries": len(plan.sub_queries),
                      **usage},
        )

    @traceable(name="rag_answer_stream", run_type="chain")
    def stream(self, query: str, history: History | None = None,
               filter_dict: dict | None = None) -> Iterator[dict]:
        """Yield events: {'type': 'token'|'citations'|'meta'|'done', 'data': ...}."""
        t0 = time.time()
        parts: list[str] = []
        with get_usage_metadata_callback() as cb:
            plan = plan_query(self.generator.llm, query, history)
            _, gated = self._retrieve_plan(plan, filter_dict)
            for token in stream_grounded(self.generator.llm, query, gated, history):
                parts.append(token)
                yield {"type": "token", "data": token}
            usage = summarize_usage(cb.usage_metadata)
        # Suppress citations when the model abstained — otherwise a source shows under "I don't
        # have enough information", which is misleading.
        answer = "".join(parts)
        abstained = is_abstention(answer)
        citations = [] if abstained else _build_citations(answer, gated)
        yield {"type": "citations", "data": citations}
        yield {"type": "meta", "data": {
            "latency_ms": round((time.time() - t0) * 1000),
            "num_sources": len(citations),
            "abstained": abstained,
            **usage,
        }}
        yield {"type": "done", "data": None}
