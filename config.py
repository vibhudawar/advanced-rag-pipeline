# Reads from .env file
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# SERVICE SELECTION
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
# Generation model (OpenAI). gpt-5.4-nano: lite + cost-efficient, good enough for grounded RAG
# (retrieval + gate do the hard part). Override to gpt-5.4-mini if evals show quality slipping.
OPENAI_GENERATION_MODEL = os.getenv("OPENAI_GENERATION_MODEL", "gpt-5.4-nano")

# OpenAI specific
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Gemini specific
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# HuggingFace specific
HF_TOKEN = os.getenv("HF_TOKEN")

# MCP Server Endpoint
MCP_SERVER_ENDPOINT = os.getenv("MCP_SERVER_ENDPOINT", "http://localhost:8000/log")

# Pinecone specific
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive")

# Contextual Retrieval (Win 1): at ingest, prepend an LLM-generated context header to each
# chunk before embedding + BM25-indexing, so a chunk retrieved in isolation still carries its
# place in the document. Header generation uses a cheap model with the doc as a cached prefix;
# very long docs are summarised first (CONTEXT_MAX_DOC_CHARS). Toggle off to fall back to plain
# chunks. See docs/win-1-contextual-retrieval.md.
CONTEXTUAL_RETRIEVAL = os.getenv("CONTEXTUAL_RETRIEVAL", "true").lower() == "true"
CONTEXT_MODEL = os.getenv("CONTEXT_MODEL", "gpt-4o-mini")
CONTEXT_MAX_DOC_CHARS = int(os.getenv("CONTEXT_MAX_DOC_CHARS", "40000"))

# Financial-doc hygiene (Win 18): strip legal/disclosure boilerplate (per-page licensing
# watermark, front-cover Reg-AC footer, and the trailing disclosures/jurisdiction/other-ticker
# rating-table section) before chunking, so the report's thesis isn't crowded out and answers
# can't be misattributed to a coverage-disclosure table. Conservative + best-effort — never
# touches financial tables or the page-1 rating. See src/ingestion/boilerplate.py.
STRIP_BOILERPLATE = os.getenv("STRIP_BOILERPLATE", "true").lower() == "true"

# Web Search Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Reranking Configuration  
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "auto")  # auto, cohere, huggingface, score, none

# RAG Pipeline Configuration
INCLUDE_WEB_SEARCH = os.getenv("INCLUDE_WEB_SEARCH", "true").lower() == "true"
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "3"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))

# Phase 2: multi-query diversity retrieval. A query planner (src/retrieval/nlu.py) routes each
# question to simple / comparison / aggregation. For the latter two it fans out one retrieval
# pass per sub-query, reranks the union, then selects with a per-document cap so several source
# documents reach the generator (which then synthesizes across them). Simple queries are
# untouched — identical to the pre-Phase-2 single-query path. Toggle off to disable planning.
RAG_MULTI_QUERY = os.getenv("RAG_MULTI_QUERY", "true").lower() == "true"
RAG_UNION_RERANK_K = int(os.getenv("RAG_UNION_RERANK_K", "12"))  # rerank depth over the fan-out union
RAG_PER_DOC_CAP = int(os.getenv("RAG_PER_DOC_CAP", "3"))         # max chunks from any one document
RAG_MULTI_FINAL_K = int(os.getenv("RAG_MULTI_FINAL_K", "8"))     # context size for multi-doc answers

# Supabase (WIN 7b). URL is shared with the Next.js client (NEXT_PUBLIC_*); the SECRET key
# is server-side only (service role, bypasses RLS) — never expose it to the frontend.
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")
# Direct/pooled Postgres URLs are only needed for migrations (scripts/apply_migration.py),
# not at runtime — the app talks to Supabase over PostgREST via supabase-py.
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_POOLED_URL = os.getenv("DATABASE_POOLED_URL")

# Observability (WIN 8). LangChain/LangSmith read these from the environment directly; set
# LANGCHAIN_TRACING_V2=true + LANGSMITH_API_KEY to turn tracing on. We default the project name
# so traces land somewhere sensible when tracing is enabled. Absent the key, tracing is off and
# @traceable is a near-no-op.
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    os.environ.setdefault("LANGCHAIN_PROJECT", "rag-production")