"""
Main RAG Application
====================
This module provides the complete RAG pipeline with two main functions:
1. ingest_documents() - Document ingestion pipeline
2. retrieve_and_generate() - Retrieval and generation pipeline with conversation memory

All components are reused from existing modules with zero code duplication.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from pathlib import Path
import time
import sqlite3

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable

# Import existing components
from src.ingestion.DocumentParsers import parse_document
from src.ingestion.ChunkCreator import get_chunker
from src.ingestion.EmbeddingCreator import get_embedder
from src.ingestion.DBIngestion import get_vector_store
from src.generation.llm_generator import expand_query, get_llm_generator
from src.reranking.reranker import get_reranker
from src.utils.Helpers import ensure_index_exists, save_document_metadata
from config import EMBEDDING_PROVIDER, LLM_PROVIDER


# ============================================================================
# LangGraph State Definition
# ============================================================================

class GraphState(TypedDict):
    """State for the RAG conversation graph"""
    messages: Annotated[list[BaseMessage], add_messages]
    sources: List[Dict[str, Any]]
    retrieved_documents: List[Document]  # Added missing field!
    reranked_documents: List[Document]
    expanded_queries: List[str]  # Added missing field!
    query_type: str  # "greeting", "rag_query", "irrelevant"
    is_rag_query: bool


# ============================================================================
# Ingestion Pipeline
# ============================================================================

@traceable(name="ingest_documents")
def ingest_documents(
    uploaded_files: List[Any],
    index_name: str,
    chunking_strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_provider: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete document ingestion pipeline.

    Flow: Files -> Parse -> Chunk -> Embed -> Vector DB

    Args:
        uploaded_files: List of Streamlit UploadedFile objects or file paths
        index_name: Name of the Pinecone index
        chunking_strategy: 'recursive', 'semantic', or 'agentic'
        chunk_size: Chunk size for recursive chunking
        chunk_overlap: Overlap for recursive chunking
        embedding_provider: Override default embedding provider

    Returns:
        Dictionary with status, num_chunks, and file_results
    """
    try:
        print(f"\n{'='*60}")
        print(f"[INGESTION] Starting Document Ingestion Pipeline")
        print(f"{'='*60}\n")

        # Initialize components
        print(f"[INIT] Initializing components...")
        embedder = get_embedder(provider=embedding_provider or EMBEDDING_PROVIDER)
        vector_store = get_vector_store()
        ensure_index_exists(vector_store, index_name, embedder)

        # Initialize chunker based on strategy
        if chunking_strategy == "agentic":
            llm_gen = get_llm_generator()
            chunker = get_chunker(strategy=chunking_strategy, llm=llm_gen.llm)
        elif chunking_strategy == "semantic":
            chunker = get_chunker(strategy=chunking_strategy, embedder=embedder.embeddings)
        else:
            chunker = get_chunker(strategy=chunking_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        all_chunks = []
        file_results = []

        # Process each file
        for uploaded_file in uploaded_files:
            file_start_time = time.time()

            # Extract file info (handles both file paths and Streamlit UploadedFile objects)
            if isinstance(uploaded_file, str):
                filename = Path(uploaded_file).name
                file_extension = Path(uploaded_file).suffix
                with open(uploaded_file, 'rb') as f:
                    file_bytes = f.read()
                file_size = len(file_bytes)
            else:
                filename = uploaded_file.name
                file_extension = Path(filename).suffix
                file_bytes = uploaded_file.read()
                file_size = uploaded_file.size

            print(f"\n[FILE] Processing: {filename}")

            try:
                # Parse and chunk document
                parsed_doc = parse_document(file_bytes, file_extension, filename)
                print(f"   [OK] Parsed ({parsed_doc['metadata']['file_type']})")

                chunks = chunker.chunk_text(parsed_doc['text'], parsed_doc['metadata'])
                print(f"   [OK] Created {len(chunks)} chunks")

                all_chunks.extend(chunks)
                file_time = time.time() - file_start_time

                # Save success metadata to database
                save_document_metadata(
                    filename=filename,
                    file_size=file_size,
                    file_type=parsed_doc['metadata']['file_type'],
                    num_chunks=len(chunks),
                    status="success",
                    ingestion_time=file_time
                )

                file_results.append({
                    "filename": filename,
                    "status": "success",
                    "num_chunks": len(chunks),
                    "time": file_time
                })

            except Exception as e:
                file_time = time.time() - file_start_time
                print(f"   [ERROR] Failed to process {filename}: {str(e)}")

                # Save failure metadata to database
                save_document_metadata(
                    filename=filename,
                    file_size=file_size,
                    file_type=file_extension.lstrip('.'),
                    num_chunks=0,
                    status="failed",
                    ingestion_time=file_time
                )

                file_results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": str(e),
                    "time": file_time
                })

        # Ingest all chunks into vector database
        print(f"\n[VECTORDB] Ingesting {len(all_chunks)} chunks into Pinecone...")
        vector_store.add_documents(index_name, all_chunks, embedder)

        print(f"\n{'='*60}")
        print(f"[SUCCESS] Ingestion Complete!")
        print(f"{'='*60}\n")

        return {
            "status": "success",
            "num_chunks": len(all_chunks),
            "index_name": index_name,
            "files_processed": len(uploaded_files),
            "file_results": file_results
        }

    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# RAG Pipeline with LangGraph
# ============================================================================

def create_rag_graph(
    index_name: str,
    embedder,
    vector_store,
    llm_generator,
    reranker,
    top_k: int = 10,
    rerank_top_k: int = 5
) -> StateGraph:
    """
    Create LangGraph workflow for RAG pipeline.

    Graph Flow:
    START -> classify_query -> [greeting -> simple_response OR rag_query -> expand_query -> retrieve -> rerank -> generate] -> END
    """

    # Node 0: Query Classification
    def classify_query_node(state: GraphState) -> GraphState:
        """Classify query as greeting, rag_query, or irrelevant"""
        try:
            query = state["messages"][-1].content

            # Simple classification prompt
            classification_prompt = f"""Classify the following user query into ONE of these categories:
- "greeting": Hi, Hello, Thanks, How are you, etc.
- "rag_query": Questions requiring document lookup or knowledge base search
- "irrelevant": Off-topic or inappropriate queries

User query: "{query}"

Respond with ONLY ONE WORD: greeting, rag_query, or irrelevant"""

            # Use LLM to classify
            from langchain_core.messages import HumanMessage as HM
            classification_response = llm_generator.llm.invoke([HM(content=classification_prompt)])
            query_type = classification_response.content.strip().lower()

            # Validate classification
            if query_type not in ["greeting", "rag_query", "irrelevant"]:
                query_type = "rag_query"  # Default to RAG query if unclear

            state["query_type"] = query_type
            state["is_rag_query"] = (query_type == "rag_query")

        except Exception as e:
            print(f"[WARN] Classification failed: {e}, defaulting to rag_query")
            state["query_type"] = "rag_query"
            state["is_rag_query"] = True

        return state

    # Node 0.5: Simple Response for Greetings
    def simple_response_node(state: GraphState) -> GraphState:
        """Generate simple response for non-RAG queries"""
        try:
            query = state["messages"][-1].content

            # Simple conversational response
            from langchain_core.messages import HumanMessage as HM
            simple_prompt = f"""You are a helpful AI assistant. Respond naturally to the user's message.
If it's a greeting, respond warmly. If it's a thank you, acknowledge it kindly.
Keep your response brief and friendly.

User message: "{query}"

Response:"""

            response = llm_generator.llm.invoke([HM(content=simple_prompt)])
            state["messages"].append(AIMessage(content=response.content))
            state["sources"] = []  # No sources for simple responses

        except Exception as e:
            print(f"[ERROR] Simple response failed: {e}")
            state["messages"].append(AIMessage(content="Hello! How can I help you today?"))

        return state

    # Node 1: Query Expansion
    def expand_query_node(state: GraphState) -> GraphState:
        """Expand user query into multiple variants"""
        try:
            # Get the last user message
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                query = last_message.content
                expanded_queries = expand_query(query)

                # Store expanded queries in state (we'll use them for retrieval)
                state["expanded_queries"] = expanded_queries
        except Exception as e:
            print(f"[WARN] Query expansion failed: {e}")
            # Fallback: use original query
            state["expanded_queries"] = [state["messages"][-1].content]

        return state

    # Node 2: Retrieve Documents
    @traceable(name="retrieve_documents_node")
    def retrieve_documents_node(state: GraphState) -> GraphState:
        """Retrieve documents from vector store"""
        try:
            expanded_queries = state.get("expanded_queries", [state["messages"][-1].content])
            all_documents = []
            seen_contents = set()

            # Retrieve for each expanded query
            for query in expanded_queries:
                docs = vector_store.similarity_search(
                    index_name=index_name,
                    query=query,
                    embedder=embedder,
                    top_k=top_k
                )

                # Deduplicate based on content
                for doc in docs:
                    if doc.page_content not in seen_contents:
                        all_documents.append(doc)
                        seen_contents.add(doc.page_content)

            state["retrieved_documents"] = all_documents

        except Exception as e:
            print(f"[ERROR] Retrieval failed: {e}")
            state["retrieved_documents"] = []

        return state

    # Node 3: Rerank Documents
    @traceable(name="rerank_node")
    def rerank_node(state: GraphState) -> GraphState:
        """Rerank documents using cross-encoder or Cohere"""
        try:
            documents = state.get("retrieved_documents", [])

            if not documents:
                state["reranked_documents"] = []
                state["sources"] = []
                return state

            query = state["messages"][-1].content

            reranked_docs = reranker.rerank(
                query=query,
                documents=documents,
                top_k=rerank_top_k
            )

            state["reranked_documents"] = reranked_docs

            # Extract sources with document content
            sources = []
            for i, doc in enumerate(reranked_docs, 1):
                source = {
                    "position": i,
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "file_type": doc.metadata.get("file_type", "Unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "score": doc.metadata.get("rerank_score", doc.metadata.get("score", 0)),
                    "content": doc.page_content  # Include document content for frontend display
                }
                sources.append(source)

            state["sources"] = sources

        except Exception as e:
            print(f"[WARN] Reranking failed: {e}")
            state["reranked_documents"] = state.get("retrieved_documents", [])[:rerank_top_k]
            state["sources"] = []

        return state

    # Node 4: Generate Response
    @traceable(name="generate_node")
    def generate_node(state: GraphState) -> GraphState:
        """Generate response using LLM with streaming"""
        try:
            query = state["messages"][-1].content
            context_docs = state.get("reranked_documents", [])

            # Generate streaming response
            response_chunks = []
            for chunk in llm_generator.generate_stream(query, context_docs):
                response_chunks.append(chunk)
                # In real Streamlit app, yield these chunks

            full_response = "".join(response_chunks)

            # Add AI response to messages
            state["messages"].append(AIMessage(content=full_response))

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            state["messages"].append(AIMessage(content=error_msg))

        return state

    # Routing function for conditional edges
    def route_query(state: GraphState) -> str:
        """Route query based on classification"""
        query_type = state.get("query_type", "rag_query")

        if query_type == "greeting":
            return "simple_response"
        elif query_type == "irrelevant":
            return "simple_response"  # Handle irrelevant queries politely
        else:
            return "expand_query"  # RAG flow

    # Build the graph
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("simple_response", simple_response_node)
    workflow.add_node("expand_query", expand_query_node)
    workflow.add_node("retrieve", retrieve_documents_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)

    # Add edges
    workflow.add_edge(START, "classify_query")

    # Conditional routing after classification
    workflow.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "simple_response": "simple_response",
            "expand_query": "expand_query"
        }
    )

    # Simple response goes directly to END
    workflow.add_edge("simple_response", END)

    # RAG flow continues
    workflow.add_edge("expand_query", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)

    return workflow


def retrieve_and_generate(
    query: str,
    index_name: str,
    conversation_id: str = "default",
    top_k: int = 10,
    rerank_top_k: int = 5
) -> Dict[str, Any]:
    """
    Complete RAG pipeline with conversation memory.

    Flow: Query -> Expand -> Retrieve -> Rerank -> Generate

    Args:
        query: User query
        index_name: Pinecone index name
        conversation_id: Unique ID for conversation thread
        top_k: Number of documents to retrieve
        rerank_top_k: Number of documents after reranking

    Returns:
        Dictionary with response, sources, and conversation state
    """
    try:
        # Initialize RAG components
        embedder = get_embedder(provider=EMBEDDING_PROVIDER)
        vector_store = get_vector_store()

        # Check if index exists
        if index_name not in vector_store.list_indexes():
            return {
                "status": "error",
                "error": "No documents uploaded. Please upload documents first before querying.",
                "response": "I don't have any documents to search through yet. Please upload some documents first!",
                "sources": []
            }

        llm_generator = get_llm_generator(provider=LLM_PROVIDER)
        reranker = get_reranker()

        # Create conversation memory
        conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
        memory = SqliteSaver(conn=conn)

        workflow = create_rag_graph(
            index_name=index_name,
            embedder=embedder,
            vector_store=vector_store,
            llm_generator=llm_generator,
            reranker=reranker,
            top_k=top_k,
            rerank_top_k=rerank_top_k
        )

        graph = workflow.compile(checkpointer=memory)

        # Prepare initial state with conversation history
        config = {"configurable": {"thread_id": conversation_id}}

        try:
            current_state = graph.get_state(config)
            existing_messages = current_state.values.get("messages", [])
        except:
            existing_messages = []

        initial_state = {
            "messages": existing_messages + [HumanMessage(content=query)],
            "sources": [],
            "retrieved_documents": [],
            "reranked_documents": [],
            "expanded_queries": [],
            "query_type": "rag_query",
            "is_rag_query": True
        }

        # Run the graph and extract results
        final_state = graph.invoke(initial_state, config)
        ai_response = final_state["messages"][-1].content
        sources = final_state.get("sources", [])

        return {
            "status": "success",
            "response": ai_response,
            "sources": sources,
            "conversation_id": conversation_id,
            "message_count": len(final_state["messages"])
        }

    except Exception as e:
        print(f"[ERROR] RAG pipeline failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


