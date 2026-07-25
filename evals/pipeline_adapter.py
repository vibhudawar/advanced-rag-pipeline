"""Adapters that expose a RAG pipeline as `run(query) -> RunResult` for evaluation.

`BaselinePipeline` replicates the current production path (expand -> retrieve -> dedupe
-> rerank -> generate) from src/main_app.py, but statelessly (no conversation memory,
one fresh run per query) and with the intermediate chunk sets surfaced so we can score
retrieval separately from generation.

Later wins (hybrid+RRF, NLU, snippet gate, ...) each add a new subclass here. Because
they share the RunResult contract, evals/evaluate.py can score any of them unchanged —
that is what makes the before/after scoreboard possible.
"""

from __future__ import annotations

import time

from config import EMBEDDING_PROVIDER, LLM_PROVIDER
from src.generation.llm_generator import expand_query, get_llm_generator
from src.ingestion.DBIngestion import get_vector_store
from src.ingestion.EmbeddingCreator import get_embedder
from src.reranking.reranker import get_reranker

from .schema import RunResult, content_hash


class BaselinePipeline:
    label = "baseline"

    def __init__(self, index_name: str, top_k: int = 10, rerank_top_k: int = 5,
                 use_expansion: bool = True, retrieval_only: bool = False):
        self.index_name = index_name
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.use_expansion = use_expansion
        self.retrieval_only = retrieval_only  # skip rerank + generation (pure retrieval benchmark)
        self.embedder = get_embedder(provider=EMBEDDING_PROVIDER)
        self.vector_store = get_vector_store()
        self.reranker = get_reranker()
        self.generator = get_llm_generator(provider=LLM_PROVIDER)
        if index_name not in self.vector_store.list_indexes():
            raise ValueError(
                f"Index '{index_name}' not found. Available: {self.vector_store.list_indexes()}"
            )

    def run(self, query: str) -> RunResult:
        t0 = time.time()
        try:
            queries = expand_query(query) if self.use_expansion else [query]

            # Retrieve for each expanded query, dedupe by content (baseline behaviour).
            candidates = []
            seen = set()
            for q in queries:
                for doc in self.vector_store.similarity_search(
                    index_name=self.index_name, query=q,
                    embedder=self.embedder, top_k=self.top_k,
                ):
                    key = content_hash(doc.page_content)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(doc)

            candidate_hashes = [content_hash(d.page_content) for d in candidates]

            # Pure retrieval benchmark: skip rerank + generation (metrics use candidates).
            if self.retrieval_only:
                return RunResult(
                    query=query, answer="", candidate_hashes=candidate_hashes,
                    retrieved_hashes=[], contexts=[], latency_s=time.time() - t0,
                )

            # Rerank down to the final set.
            reranked = self.reranker.rerank(
                query=query, documents=candidates, top_k=self.rerank_top_k,
            ) if candidates else []

            contexts = [d.page_content for d in reranked]
            retrieved_hashes = [content_hash(c) for c in contexts]

            # Generate (collect the stream into a full answer).
            answer = "".join(self.generator.generate_stream(query, reranked)) if reranked else ""

            return RunResult(
                query=query, answer=answer,
                candidate_hashes=candidate_hashes,
                retrieved_hashes=retrieved_hashes,
                contexts=contexts,
                latency_s=time.time() - t0,
            )
        except Exception as e:  # noqa: BLE001 - eval harness: capture, don't crash the run
            return RunResult(query=query, answer="", latency_s=time.time() - t0, error=str(e))


# Registry so evaluate.py --pipeline <name> can pick one.
PIPELINES = {"baseline": BaselinePipeline}
