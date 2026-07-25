"""Hybrid retrieval: BM25 (lexical) + dense vectors, fused with Reciprocal Rank Fusion.

Why hybrid: pure vector search misses exact-token matches — names, tickers, acronyms,
error codes, rare scientific terms — because those get smoothed into the embedding.
BM25 nails them. Fusing the two covers both "means the same thing" and "contains this
exact token".

Why RRF (and not score-weighting): BM25 scores are unbounded (~0-30) and cosine
similarity is 0-1 — incomparable scales. Normalizing them into a weighted sum is fragile
and needs per-corpus tuning. RRF ignores the scores and fuses on *rank* alone, with a
single constant k (conventionally 60). It's a strong, tuning-free baseline.

This module is retrieval-mechanism only (no embeddings/DB deps): callers pass in the two
ranked lists (or use BM25Index for the lexical half). Document identity is whatever key
the caller chooses — the eval layer uses a content hash so results line up with ground truth.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization — matches what BM25Okapi is fed at index time."""
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Fuse ranked lists of keys into one. score(key) = Σ 1/(k + rank_in_list).

    Uses ranks only, so the input lists' scores need not be comparable. Keys absent from a
    list simply contribute nothing for it. Returns keys ordered by fused score, best first.
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, key in enumerate(lst, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda key: scores[key], reverse=True)


class BM25Index:
    """In-memory BM25 index over a fixed corpus, keyed by caller-supplied ids.

    Fine for corpora up to ~10^5 chunks (scoring is vectorized). For larger corpora move
    the lexical half into the search engine (e.g. Vespa/OpenSearch BM25).
    """

    def __init__(self, keys: Sequence[str], texts: Sequence[str]):
        if len(keys) != len(texts):
            raise ValueError("keys and texts must be the same length")
        self.keys = list(keys)
        self.texts_by_key = dict(zip(keys, texts))
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, top_k: int) -> list[str]:
        """Return up to top_k keys ranked by BM25 score (descending), score > 0 only."""
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.keys[i] for i in ranked[:top_k] if scores[i] > 0]
