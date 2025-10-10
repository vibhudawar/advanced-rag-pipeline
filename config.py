# Reads from .env file
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# SERVICE SELECTION
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto") 

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