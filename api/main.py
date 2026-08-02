"""FastAPI service exposing the RAG pipeline over Server-Sent Events (WIN 7a).

    POST /stream            -> SSE: `token` events, then a `citations` event, then `done`
    GET  /api/healthCheck   -> liveness
    GET  /api/readinessCheck-> readiness (builds the pipeline once)

Security (OX python-server-hardening): all secrets come from env via `config`; exception
details are logged server-side and NEVER returned to the client; CSP/HSTS + nosniff headers
are set on every response. Run: `uvicorn api.main:app`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from api.auth import AuthUser, get_current_user
from src.rag_pipeline import RagPipeline, corpus_cache_path
from src.storage.supabase_store import get_store

logger = logging.getLogger("rag.api")

# The authenticated caller, resolved from the validated Supabase bearer token.
CurrentUser = Annotated[AuthUser, Depends(get_current_user)]

RAG_INDEX = os.getenv("RAG_INDEX", "beir-scifact")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
# Scope retrieval to the requesting user's own documents (chunks carry user_id). Turn OFF only
# when serving a shared corpus without per-user tagging, e.g. the beir-scifact benchmark.
SCOPE_BY_USER = os.getenv("RAG_SCOPE_BY_USER", "true").lower() == "true"

# Ingestion guardrails (env-overridable). Bound file size and chunk count so a single upload
# can't run up unbounded embedding cost or memory.
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
MAX_INGEST_CHUNKS = int(os.getenv("MAX_INGEST_CHUNKS", "300"))

app = FastAPI(title="RAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)

_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    """Lazily build the pipeline once (builds the BM25 index from the corpus)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline(RAG_INDEX)
    return _pipeline


def reset_pipeline() -> None:
    """Drop the cached pipeline + stale BM25 corpus cache after an ingest, so the next chat
    rebuilds the lexical index over the newly upserted chunks. (Vector search already sees new
    chunks immediately since it queries Pinecone live; this brings the BM25 half up to date.)"""
    global _pipeline
    _pipeline = None
    try:
        corpus_cache_path(RAG_INDEX).unlink(missing_ok=True)
    except OSError:
        logger.exception("failed clearing corpus cache")


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


class RenameRequest(BaseModel):
    title: str


class StreamRequest(BaseModel):
    question: str
    conversation_id: str | None = None       # continue an existing thread
    document: str | None = None              # scope retrieval to one document (filename)
    history: list[dict] | None = None         # fallback if persistence is off


@app.get("/api/healthCheck")
def health_check():
    return {"status": "ok"}


@app.get("/api/readinessCheck")
def readiness_check():
    try:
        get_pipeline()
        return {"status": "ready"}
    except Exception:
        logger.exception("readiness check failed")
        return JSONResponse({"status": "not ready"}, status_code=503)


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/conversations")
def list_conversations(user: CurrentUser):
    """The signed-in user's chat threads (the sidebar), newest-activity first."""
    store = get_store()
    assert store is not None  # get_current_user 503s if Supabase isn't configured
    try:
        return {"conversations": store.list_conversations(user.id)}
    except Exception:
        logger.exception("list_conversations failed")
        raise HTTPException(status_code=500, detail="internal error")


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user: CurrentUser):
    """Full message history for a thread the user owns. 404 if missing or not theirs
    (same status for both, so ownership can't be probed)."""
    store = get_store()
    assert store is not None
    try:
        messages = store.get_messages(conversation_id, user.id)
    except Exception:
        logger.exception("get_conversation failed")
        raise HTTPException(status_code=500, detail="internal error")
    if messages is None:
        raise HTTPException(status_code=404, detail="not found")
    return {"conversation_id": conversation_id, "messages": messages}


@app.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, req: RenameRequest, user: CurrentUser):
    """Rename a thread the user owns. 404 if missing/not theirs."""
    store = get_store()
    assert store is not None
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    try:
        ok = store.rename_conversation(conversation_id, user.id, title)
    except Exception:
        logger.exception("rename_conversation failed")
        raise HTTPException(status_code=500, detail="internal error")
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return {"id": conversation_id, "title": title[:120]}


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user: CurrentUser):
    """Delete a thread the user owns (messages cascade). 404 if missing/not theirs."""
    store = get_store()
    assert store is not None
    try:
        ok = store.delete_conversation(conversation_id, user.id)
    except Exception:
        logger.exception("delete_conversation failed")
        raise HTTPException(status_code=500, detail="internal error")
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True}


@app.post("/stream")
def stream(req: StreamRequest, user: CurrentUser):
    pipeline = get_pipeline()
    store = get_store()
    assert store is not None  # get_current_user 503s if Supabase isn't configured

    # Resolve conversation + history from the DB (source of truth), scoped to the verified user.
    # Continuing an existing thread requires owning it; a new thread is created for this user.
    conversation_id = req.conversation_id
    if conversation_id is not None and not store.owns_conversation(conversation_id, user.id):
        raise HTTPException(status_code=404, detail="not found")
    try:
        if conversation_id is None:
            conversation_id = store.create_conversation(title=req.question, user_id=user.id)
        history = store.get_history(conversation_id)
        store.add_message(conversation_id, "user", req.question)
    except Exception:
        logger.exception("persistence (load/user-message) failed; continuing without it")
        history = []

    # Scope retrieval: to the verified user's own documents, and optionally to one document.
    # user_id/document come from the token + request, never trusted for identity beyond scoping.
    filter_dict: dict = {}
    if SCOPE_BY_USER:
        filter_dict["user_id"] = user.id
    if req.document:
        filter_dict["filename"] = req.document

    def event_stream():
        answer_parts: list[str] = []
        citations: list[dict] = []
        meta: dict = {}
        try:
            if conversation_id:
                yield _sse("conversation", {"conversation_id": conversation_id})
            for event in pipeline.stream(req.question, history, filter_dict=filter_dict or None):
                if event["type"] == "token":
                    answer_parts.append(event["data"])
                elif event["type"] == "citations":
                    citations = event["data"]
                elif event["type"] == "meta":
                    meta = event["data"]
                yield _sse(event["type"], event["data"])
            if store is not None and conversation_id:
                try:
                    # Persist run metrics alongside the message so the UI can show them on reload.
                    store.add_message(conversation_id, "assistant", "".join(answer_parts),
                                      citations=citations,
                                      metadata={"pipeline": "production", **meta})
                except Exception:
                    logger.exception("persistence (assistant-message) failed")
        except Exception:
            logger.exception("stream generation failed")
            yield _sse("error", "internal error")  # generic — no details leaked to client

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _ingest_bytes(data: bytes, ext: str, filename: str, user_id: str) -> int:
    """Parse → chunk → embed → upsert one document into the chat index. Returns the chunk count.
    Raises ValueError for user-fixable problems (no text, too many chunks)."""
    from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_PROVIDER
    from src.ingestion.ChunkCreator import get_chunker
    from src.ingestion.DBIngestion import get_vector_store
    from src.ingestion.DocumentParsers import parse_document
    from src.ingestion.EmbeddingCreator import get_embedder
    from src.ingestion.metadata import extract_metadata

    parsed = parse_document(data, ext, filename)
    text = (parsed.get("text") or "").strip()
    if not text:
        raise ValueError("No extractable text found in the document.")

    # Financial-doc hygiene (Win 18): drop legal/disclosure boilerplate before anything
    # downstream (metadata, chunking, contextualization) sees the text.
    from config import STRIP_BOILERPLATE
    if STRIP_BOILERPLATE:
        from src.ingestion.boilerplate import strip_boilerplate
        text = strip_boilerplate(text)

    # Doc-level metadata (company/ticker/doc_type/date/period/rating/topics) — best-effort,
    # nullable, non-null keys only. Rides on every chunk so retrieval can filter on it.
    doc_meta = extract_metadata(filename, text)
    metadata = {**parsed.get("metadata", {}), **doc_meta, "user_id": user_id,
                "source": filename, "filename": filename}
    # Structure-aware: split on the markdown headings pymupdf4llm produced (falls back to
    # recursive for header-less text). Each chunk carries its `section` path.
    chunks = get_chunker("markdown", chunk_size=CHUNK_SIZE,
                         chunk_overlap=CHUNK_OVERLAP).chunk_text(text, metadata)
    if len(chunks) > MAX_INGEST_CHUNKS:
        raise ValueError(
            f"Document produced {len(chunks)} chunks (max {MAX_INGEST_CHUNKS}). "
            "Upload a smaller file or split it."
        )

    # Contextual Retrieval (Win 1): prepend a situating header to each chunk before embedding.
    from config import CONTEXTUAL_RETRIEVAL
    if CONTEXTUAL_RETRIEVAL:
        from src.ingestion.contextualizer import contextualize_chunks
        chunks = contextualize_chunks(chunks, doc_text=text)

    embedder = get_embedder(provider=EMBEDDING_PROVIDER)
    get_vector_store().add_documents(RAG_INDEX, chunks, embedder)
    return len(chunks)


@app.post("/ingest")
async def ingest(file: UploadFile, user: CurrentUser):
    """Ingest one uploaded document into the chat index (auth required). Enforces file-type,
    size, and chunk-count guardrails; records the outcome in the documents table."""
    store = get_store()
    assert store is not None

    filename = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or '?'}'. Allowed: "
                   f"{', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Read at most the cap + 1 byte, so an oversized file is rejected without buffering all of it.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    file_type = ext.lstrip(".")

    # Dedup: skip a file this user already ingested (same bytes) so re-uploads don't create
    # duplicate vectors that would double-count in aggregation queries.
    content_hash = hashlib.sha256(data).hexdigest()[:16]
    try:
        if store.document_exists(user.id, content_hash):
            return {"document": None, "num_chunks": 0, "duplicate": True}
    except Exception:
        logger.exception("dedup check failed; continuing with ingest")

    t0 = time.time()
    try:
        num_chunks = _ingest_bytes(data, ext, filename, user.id)
    except ValueError as exc:
        store.save_document(filename, file_type, len(data), None, RAG_INDEX, "failed",
                            error=str(exc)[:500], user_id=user.id, content_hash=content_hash)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("ingest failed")
        store.save_document(filename, file_type, len(data), None, RAG_INDEX, "failed",
                            error="ingestion error", user_id=user.id, content_hash=content_hash)
        raise HTTPException(status_code=500, detail="internal error")

    row = store.save_document(filename, file_type, len(data), num_chunks, RAG_INDEX, "success",
                              ingestion_time_s=time.time() - t0, user_id=user.id,
                              content_hash=content_hash)
    reset_pipeline()  # next chat rebuilds BM25 including the new chunks
    return {"document": row, "num_chunks": num_chunks}


@app.get("/documents")
def list_documents(user: CurrentUser):
    """The signed-in user's ingested documents (the Ingest tab table)."""
    store = get_store()
    assert store is not None
    try:
        return {"documents": store.list_documents(user.id)}
    except Exception:
        logger.exception("list_documents failed")
        raise HTTPException(status_code=500, detail="internal error")
