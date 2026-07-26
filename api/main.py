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
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
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
