"""Unit tests for the deterministic parts of the eval harness.

These run on EVERY push (they need no API keys, network, or heavy ML deps) and guard the
metric math + ground-truth hashing that the CI regression gate depends on.
"""

import math

import pytest

from evals.metrics import hit_rate_at_k, mean, ndcg_at_k, reciprocal_rank
from evals.schema import GoldenItem, content_hash


def test_hit_rate():
    assert hit_rate_at_k(["x", "a", "y", "b"], {"a", "b"}, k=10) == 1.0
    assert hit_rate_at_k(["x", "a"], {"a"}, k=1) == 0.0          # 'a' is rank 2, outside k=1
    assert hit_rate_at_k(["a"], {"a"}, k=1) == 1.0
    assert hit_rate_at_k([], {"a"}, k=10) == 0.0
    assert hit_rate_at_k(["a"], set(), k=10) == 0.0              # no ground truth


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "y", "b"], {"a", "b"}) == 0.5   # first hit at rank 2
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0
    assert reciprocal_rank([], set()) == 0.0


def test_ndcg_matches_manual_computation():
    # 'a' at rank 2, 'b' at rank 4; ideal = both at ranks 1,2
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["x", "a", "y", "b"], {"a", "b"}, k=10) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=10) == 1.0
    assert ndcg_at_k([], {"a"}, k=10) == 0.0
    assert ndcg_at_k(["a"], set(), k=10) == 0.0


def test_mean_ignores_none():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([]) == 0.0
    assert mean([None]) == 0.0


def test_content_hash_stable_and_trimmed():
    assert content_hash("  hello  ") == content_hash("hello")     # whitespace-insensitive
    assert len(content_hash("x")) == 16
    assert content_hash("a") != content_hash("b")


def test_golden_item_roundtrip_and_answerability():
    gi = GoldenItem(id="t", query="q", reference_answer="a",
                    relevant_chunk_hashes=[content_hash("c")], q_type="factoid")
    assert GoldenItem.from_json(gi.to_json()).query == "q"
    assert gi.is_answerable is True
    un = GoldenItem(id="u", query="q2", reference_answer="",
                    relevant_chunk_hashes=[], q_type="unanswerable")
    assert un.is_answerable is False
