"""FastAPI service exposing the RAG pipeline over Server-Sent Events (WIN 7a).

    POST /stream            -> SSE: `token` events, then a `citations` event, then `done`
    GET  /api/healthCheck   -> liveness
    GET  /api/readinessCheck-> readiness (builds the pipeline once)

Security (OX python-server-hardening): all secrets come from env via `config`; exception
details are logged server-side and NEVER returned to the client; CSP/HSTS + nosniff headers
are set on every response. Run: `uvicorn api.main:app`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from api.auth import AuthUser, get_current_user
from src.rag_pipeline import RagPipeline
from src.storage.supabase_store import get_store

logger = logging.getLogger("rag.api")

# The authenticated caller, resolved from the validated Supabase bearer token.
CurrentUser = Annotated[AuthUser, Depends(get_current_user)]

RAG_INDEX = os.getenv("RAG_INDEX", "beir-scifact")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

# Ingestion guardrails (env-overridable). Bound file size and chunk count so a single upload
# can't run up unbounded embedding cost or memory.
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
MAX_INGEST_CHUNKS = int(os.getenv("MAX_INGEST_CHUNKS", "300"))

app = FastAPI(title="RAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
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
        cache = Path("data/bm25_cache") / f"{RAG_INDEX}.json"
        cache.unlink(missing_ok=True)
    except OSError:
        logger.exception("failed clearing corpus cache")


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


class StreamRequest(BaseModel):
    question: str
    conversation_id: str | None = None       # continue an existing thread
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

    def event_stream():
        answer_parts: list[str] = []
        citations: list[dict] = []
        try:
            if conversation_id:
                yield _sse("conversation", {"conversation_id": conversation_id})
            for event in pipeline.stream(req.question, history):
                if event["type"] == "token":
                    answer_parts.append(event["data"])
                elif event["type"] == "citations":
                    citations = event["data"]
                yield _sse(event["type"], event["data"])
            if store is not None and conversation_id:
                try:
                    store.add_message(conversation_id, "assistant", "".join(answer_parts),
                                      citations=citations, metadata={"pipeline": "production"})
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

    parsed = parse_document(data, ext, filename)
    text = (parsed.get("text") or "").strip()
    if not text:
        raise ValueError("No extractable text found in the document.")

    metadata = {**parsed.get("metadata", {}), "user_id": user_id, "source": filename,
                "filename": filename}
    chunks = get_chunker("recursive", chunk_size=CHUNK_SIZE,
                         chunk_overlap=CHUNK_OVERLAP).chunk_text(text, metadata)
    if len(chunks) > MAX_INGEST_CHUNKS:
        raise ValueError(
            f"Document produced {len(chunks)} chunks (max {MAX_INGEST_CHUNKS}). "
            "Upload a smaller file or split it."
        )

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
    t0 = time.time()
    try:
        num_chunks = _ingest_bytes(data, ext, filename, user.id)
    except ValueError as exc:
        store.save_document(filename, file_type, len(data), None, RAG_INDEX, "failed",
                            error=str(exc)[:500], user_id=user.id)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("ingest failed")
        store.save_document(filename, file_type, len(data), None, RAG_INDEX, "failed",
                            error="ingestion error", user_id=user.id)
        raise HTTPException(status_code=500, detail="internal error")

    row = store.save_document(filename, file_type, len(data), num_chunks, RAG_INDEX,
                              "success", ingestion_time_s=time.time() - t0, user_id=user.id)
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
