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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.rag_pipeline import RagPipeline

logger = logging.getLogger("rag.api")

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
    history: list[dict] | None = None  # [{"role": "...", "content": "..."}]


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


@app.post("/stream")
def stream(req: StreamRequest):
    pipeline = get_pipeline()
    history = [
        (m["role"], m["content"])
        for m in (req.history or [])
        if m.get("role") and m.get("content")
    ]

    def event_stream():
        try:
            for event in pipeline.stream(req.question, history):
                yield _sse(event["type"], event["data"])
        except Exception:
            logger.exception("stream generation failed")
            yield _sse("error", "internal error")  # generic — no details leaked to client

    return StreamingResponse(event_stream(), media_type="text/event-stream")
