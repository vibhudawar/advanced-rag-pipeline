"""Shared data types for the eval harness.

Ground truth for retrieval is keyed on a **content hash of the chunk text**, not the
Pinecone vector id. LangChain's Pinecone integration auto-generates uuids on every
ingest, so ids are not stable across re-ingestion — the chunk text is. Both the golden
generator and the evaluator hash `page_content` the same way, so they always line up.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

QType = Literal["factoid", "multi_hop", "aggregation", "unanswerable"]
Source = Literal["synthetic", "curated", "beir"]


def content_hash(text: str) -> str:
    """Stable 16-hex-char fingerprint of a chunk's text. Used as the retrieval id."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


@dataclass
class GoldenItem:
    id: str
    query: str
    reference_answer: str            # "" for unanswerable items
    relevant_chunk_hashes: list[str]  # [] for unanswerable items
    q_type: QType
    source: Source = "synthetic"
    needs_review: bool = True         # synthetic items start unverified

    @property
    def is_answerable(self) -> bool:
        return self.q_type != "unanswerable" and bool(self.relevant_chunk_hashes)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> GoldenItem:
        return cls(**json.loads(line))


def write_jsonl(items: Iterable[GoldenItem], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[GoldenItem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it first: python -m evals.generate_golden --index <name>"
        )
    with path.open(encoding="utf-8") as f:
        return [GoldenItem.from_json(line) for line in f if line.strip()]


@dataclass
class RunResult:
    """One pipeline run over a single query."""
    query: str
    answer: str
    candidate_hashes: list[str] = field(default_factory=list)  # retrieved, pre-rerank (retrieval quality)
    retrieved_hashes: list[str] = field(default_factory=list)  # final set handed to the generator (post-rerank)
    contexts: list[str] = field(default_factory=list)          # the chunk texts the generator saw
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class ItemScore:
    id: str
    q_type: str
    # retrieval (None for unanswerable)
    hit_at_k: float | None = None
    mrr: float | None = None
    ndcg_at_k: float | None = None
    # generation (None if it errored)
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_relevance: float | None = None
    # behaviour
    abstained: bool | None = None
    abstention_correct: bool | None = None
    notes: dict = field(default_factory=dict)
