"""Stable content hash used as a chunk's identity across the codebase.

Both the production pipeline (candidate identity, BM25 keys) and the eval harness
(ground-truth `relevant_chunk_hashes`) must hash chunk text the SAME way, or retrieval
metrics won't line up. Keeping the single definition here — imported by both — guarantees
that. LangChain's Pinecone integration regenerates vector ids on every ingest, so the chunk
*text* is the only stable identity.
"""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """16-hex-char fingerprint of a chunk's (stripped) text."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
