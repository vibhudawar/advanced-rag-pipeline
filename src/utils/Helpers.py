"""
Helper utilities for RAG application.

This module contains helper functions used across the RAG pipeline.
"""

from typing import List, Dict, Any
import sqlite3
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
) -> Dict[str, Any]:
    """
    Retrieve conversation history with sources for a given thread.

    Args:
        conversation_id: Conversation thread ID
        db_path: Path to SQLite database file

    Returns:
        Dict with 'messages' (List[BaseMessage]) and 'sources' (List[Dict])
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
            sources = channel_values.get("sources", [])

            return {"messages": messages, "sources": sources}

        return {"messages": [], "sources": []}
    except Exception as e:
        print(f"[WARN] Could not retrieve conversation history: {e}")
        import traceback
        traceback.print_exc()
        return {"messages": [], "sources": []}


def retrieve_all_threads(db_path: str = "chatbot.db") -> List[str]:
    """
    Retrieve all conversation thread IDs sorted by most recent first.

    Args:
        db_path: Path to SQLite database file

    Returns:
        List of thread IDs sorted by timestamp (newest first)
    """
    try:
        conn = sqlite3.connect(database=db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)

        # Collect threads with their most recent timestamp
        threads_with_ts = {}
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
            if thread_id:
                # Get timestamp from checkpoint
                ts = checkpoint.checkpoint.get("ts", 0)
                # Keep only the latest timestamp for each thread
                if thread_id not in threads_with_ts or ts > threads_with_ts[thread_id]:
                    threads_with_ts[thread_id] = ts

        # Sort by timestamp descending (newest first)
        sorted_threads = sorted(threads_with_ts.items(), key=lambda x: x[1], reverse=True)
        return [thread_id for thread_id, _ in sorted_threads]

    except Exception as e:
        print(f"[WARN] Could not retrieve threads: {e}")
        return []


# ============================================================================
# Document Metadata Management (SQLite)
# ============================================================================

def init_documents_table(db_path: str = "chatbot.db") -> None:
    """Initialize documents metadata table in SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                file_size INTEGER NOT NULL,
                file_type TEXT NOT NULL,
                num_chunks INTEGER NOT NULL,
                status TEXT NOT NULL,
                ingestion_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_created
            ON documents(created_at DESC)
        """)

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Failed to initialize documents table: {e}")


def save_document_metadata(
    filename: str,
    file_size: int,
    file_type: str,
    num_chunks: int,
    status: str,
    ingestion_time: float,
    db_path: str = "chatbot.db"
) -> bool:
    """
    Save document metadata to SQLite database.

    Args:
        filename: Name of the document
        file_size: Size in bytes
        file_type: Type of file (pdf, docx, txt, etc.)
        num_chunks: Number of chunks created
        status: 'success' or 'failed'
        ingestion_time: Time taken in seconds
        db_path: Path to SQLite database

    Returns:
        True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO documents
            (filename, file_size, file_type, num_chunks, status, ingestion_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (filename, file_size, file_type, num_chunks, status, ingestion_time))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save document metadata: {e}")
        return False


def get_all_documents(db_path: str = "chatbot.db") -> List[Dict[str, Any]]:
    """
    Retrieve all documents from SQLite database.

    Args:
        db_path: Path to SQLite database

    Returns:
        List of document metadata dictionaries
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT filename, file_size, file_type, num_chunks, status,
                   ingestion_time, created_at
            FROM documents
            ORDER BY created_at DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        documents = []
        for row in rows:
            documents.append({
                "filename": row[0],
                "file_size": row[1],
                "file_type": row[2],
                "num_chunks": row[3],
                "status": row[4],
                "ingestion_time": row[5],
                "created_at": row[6]
            })

        return documents
    except Exception as e:
        print(f"[ERROR] Failed to retrieve documents: {e}")
        return []
