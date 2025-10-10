"""
Helper utilities for RAG application.

This module contains helper functions used across the RAG pipeline.
"""

from typing import List, Dict, Any
import sqlite3
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver


def ensure_index_exists(vector_store, index_name: str, embedder) -> None:
    """
    Ensure Pinecone index exists, create if it doesn't.

    Args:
        vector_store: Vector store instance
        index_name: Name of the index
        embedder: Embedder instance (to get dimension)
    """
    existing_indexes = vector_store.list_indexes()

    if index_name not in existing_indexes:
        print(f"   [CREATE] Creating new index: {index_name}")
        dimension = embedder.get_embedding_dimension()
        vector_store.create_index(
            index_name=index_name,
            dimension=dimension,
            metric="cosine"
        )
    else:
        print(f"   [OK] Index '{index_name}' already exists")


def get_conversation_history(
    conversation_id: str,
    db_path: str = "chatbot.db"
) -> List[BaseMessage]:
    """
    Retrieve conversation history for a given thread.

    Args:
        conversation_id: Conversation thread ID
        db_path: Path to SQLite database file

    Returns:
        List of messages in the conversation
    """
    try:
        conn = sqlite3.connect(database=db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)

        # Get state for the specific thread
        config = {"configurable": {"thread_id": conversation_id}}

        # Get the latest checkpoint for this thread
        checkpoint_tuple = checkpointer.get_tuple(config)

        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            # Access the channel_values from the checkpoint
            channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
            messages = channel_values.get("messages", [])
            return messages

        return []
    except Exception as e:
        print(f"[WARN] Could not retrieve conversation history: {e}")
        import traceback
        traceback.print_exc()
        return []


def retrieve_all_threads(db_path: str = "chatbot.db") -> List[str]:
    """
    Retrieve all conversation thread IDs from the database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        List of unique thread IDs
    """
    try:
        conn = sqlite3.connect(database=db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)

        all_threads = set()
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
            if thread_id:
                all_threads.add(thread_id)

        return list(all_threads)
    except Exception as e:
        print(f"[WARN] Could not retrieve threads: {e}")
        return []


# ============================================================================
# Document Metadata Management
# ============================================================================

def get_all_documents(vector_store, index_name: str) -> List[Dict[str, Any]]:
    """
    Retrieve document statistics from Pinecone index.

    This is suitable for cloud deployments where filesystem is ephemeral.
    Returns basic statistics about the indexed documents.

    Args:
        vector_store: Vector store instance
        index_name: Name of the Pinecone index

    Returns:
        List with a single dict containing index statistics
    """
    try:
        # Check if index exists
        if index_name not in vector_store.list_indexes():
            return []

        # Get index statistics
        stats = vector_store.get_index_stats(index_name)

        # Return basic info about the index
        if stats.get("total_vector_count", 0) > 0:
            return [{
                "filename": f"{index_name} (Pinecone Index)",
                "file_type": "vector_db",
                "file_size": 0,
                "num_chunks": stats.get("total_vector_count", 0),
                "status": "success",
                "created_at": datetime.now().isoformat()
            }]

        return []
    except Exception as e:
        print(f"[WARN] Failed to retrieve documents: {e}")
        return []
