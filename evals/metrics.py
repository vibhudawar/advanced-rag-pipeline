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
import re
from collections.abc import Sequence

_CITE_MARKER = re.compile(r"\[(\d+)\]")


def _norm_source(s: str | None) -> str:
    """Normalize a source/filename for set comparison (basename-ish, case-insensitive)."""
    return (s or "").strip().lower()


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


def citation_validity(answer: str, citations: Sequence[dict], abstained: bool) -> bool:
    """True if the answer's inline citations are well-formed:
      - abstention: must carry NO citation markers (a source under "I don't know" is wrong);
      - otherwise: every inline [n] marker must resolve to a provided citation (1..len), and
        at least one citation must be present (grounded answers must cite).
    Pure/un-gameable: checks marker↔citation consistency, not model opinion.
    """
    markers = [int(m) for m in _CITE_MARKER.findall(answer or "")]
    if abstained:
        return not markers and not citations
    if not citations:
        return False
    valid_range = set(range(1, len(citations) + 1))
    return bool(markers) and all(m in valid_range for m in markers)


def source_coverage(citations: Sequence[dict], expected_sources: Sequence[str]) -> float | None:
    """Fraction of `expected_sources` (by filename) that appear among the cited sources.
    Returns None when nothing is expected (metric not applicable to this item).
    Document-level stand-in for multi-doc recall: did the answer draw on the docs it should?
    """
    if not expected_sources:
        return None
    cited = {_norm_source(c.get("source")) for c in citations}
    want = {_norm_source(s) for s in expected_sources}
    hit = sum(1 for s in want if s in cited)
    return hit / len(want)
