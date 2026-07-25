"""Retrieval metrics — pure functions, no LLM, no dependencies.

All functions take:
  retrieved: ordered list of chunk hashes returned by the pipeline (best-first)
  relevant:  set of ground-truth relevant chunk hashes
and are safe to call with empty inputs.

These are deliberately un-gameable: they compare ids, not model opinions. That is why
they are the primary signal for retrieval regressions in CI.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """1.0 if any relevant chunk appears in the top-k, else 0.0."""
    if not relevant:
        return 0.0
    return 1.0 if any(h in relevant for h in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1/rank of the first relevant chunk (rank starts at 1); 0.0 if none found."""
    if not relevant:
        return 0.0
    for i, h in enumerate(retrieved, start=1):
        if h in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Binary-relevance nDCG@k. Ideal DCG assumes all relevant docs ranked first."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, h in enumerate(retrieved[:k], start=1):
        if h in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean(values: list[float | None]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0
