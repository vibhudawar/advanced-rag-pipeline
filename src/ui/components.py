"""
UI Components for RAG Frontend
================================
This module contains reusable Streamlit UI components for the RAG application.
"""

import streamlit as st
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime


# ============================================================================
# File Upload Components
# ============================================================================

def create_file_uploader(accept_multiple_files: bool = True, key: str = "file_uploader") -> List[Any]:
    """
    Create a file uploader widget.

    Args:
        accept_multiple_files: Whether to accept multiple files
        key: Unique key for the widget

    Returns:
        List of uploaded files
    """
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=accept_multiple_files,
        key=key,
        help="Upload PDF, DOCX, TXT, or MD files"
    )

    if accept_multiple_files:
        return uploaded_files if uploaded_files else []
    else:
        return [uploaded_files] if uploaded_files else []


def display_uploaded_files(files: List[Any], on_remove: Optional[Callable] = None):
    """
    Display list of uploaded files with remove option.

    Args:
        files: List of uploaded files
        on_remove: Callback function when file is removed
    """
    if not files:
        return

    st.subheader(f"📁 Uploaded Files ({len(files)})")

    for idx, file in enumerate(files):
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.text(f"📄 {file.name}")

        with col2:
            file_size_kb = file.size / 1024
            if file_size_kb < 1024:
                st.text(f"{file_size_kb:.1f} KB")
            else:
                st.text(f"{file_size_kb/1024:.1f} MB")

        with col3:
            if st.button("🗑️ Remove", key=f"remove_{idx}_{file.name}"):
                if on_remove:
                    on_remove(idx)
                    st.rerun()


# ============================================================================
# Configuration Components
# ============================================================================

def create_chunking_config() -> Dict[str, Any]:
    """
    Create chunking configuration UI.

    Returns:
        Dictionary with chunking configuration
    """
    st.subheader("⚙️ Chunking Configuration")

    # Strategy selection - always shown
    strategy = st.selectbox(
        "Chunking Strategy",
        ["recursive", "semantic", "agentic"],
        help="Recursive: Fast, fixed-size chunks | Semantic: Context-aware | Agentic: AI-powered intelligent chunking"
    )

    # Conditionally show chunk size and overlap ONLY for recursive chunking
    chunk_size = 1000  # Default values
    chunk_overlap = 200

    if strategy == "recursive":
        col1, col2 = st.columns(2)

        with col1:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Number of characters per chunk (for recursive chunking)"
            )

        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Number of overlapping characters between chunks"
            )

    # Show strategy info
    strategy_info = {
        "recursive": "⚡ **Fast & Efficient**: Fixed-size chunks with character-level splitting. You can configure the chunk size and overlap.",
        "semantic": "🧠 **Context-Aware**: Automatically splits based on semantic meaning using embeddings. No manual configuration needed.",
        "agentic": "🤖 **AI-Powered**: Intelligent chunking using LLM to determine optimal boundaries. The AI decides chunk sizes automatically."
    }

    st.info(strategy_info[strategy])

    return {
        "strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }


# ============================================================================
# Results Display Components
# ============================================================================

def display_ingestion_results_table(file_results: List[Dict[str, Any]]):
    """
    Display ingestion results in a table.

    Args:
        file_results: List of file processing results
    """
    st.subheader("📊 Ingestion Results")

    for result in file_results:
        filename = result.get("filename", "Unknown")
        status = result.get("status", "unknown")

        if status == "success":
            num_chunks = result.get("num_chunks", 0)
            time_taken = result.get("time", 0)

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.success(f"✅ {filename}")
            with col2:
                st.text(f"{num_chunks} chunks")
            with col3:
                st.text(f"{time_taken:.2f}s")
        else:
            error = result.get("error", "Unknown error")
            st.error(f"❌ {filename}: {error}")


def display_documents_table(documents: List[Dict[str, Any]]):
    """
    Display existing documents in a table.

    Args:
        documents: List of document metadata
    """
    st.subheader("📚 Existing Documents")

    if not documents:
        st.info("No documents ingested yet.")
        return

    # Create table headers
    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 2])

    with col1:
        st.markdown("**Filename**")
    with col2:
        st.markdown("**Type**")
    with col3:
        st.markdown("**Size**")
    with col4:
        st.markdown("**Chunks**")
    with col5:
        st.markdown("**Uploaded**")

    st.divider()

    # Display documents
    for doc in documents:
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 2])

        status = doc.get("status", "unknown")
        filename = doc.get("filename", "Unknown")

        with col1:
            if status == "success":
                st.text(f"✅ {filename}")
            else:
                st.text(f"❌ {filename}")

        with col2:
            st.text(doc.get("file_type", "N/A").upper())

        with col3:
            size_kb = doc.get("file_size", 0) / 1024
            if size_kb < 1024:
                st.text(f"{size_kb:.0f} KB")
            else:
                st.text(f"{size_kb/1024:.1f} MB")

        with col4:
            st.text(str(doc.get("num_chunks", 0)))

        with col5:
            created_at = doc.get("created_at", "")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at)
                    st.text(dt.strftime("%Y-%m-%d %H:%M"))
                except:
                    st.text("N/A")
            else:
                st.text("N/A")


# ============================================================================
# Chat Components
# ============================================================================

def display_chat_message(role: str, content: str, sources: List[Dict[str, Any]] = None, message_id: str = None):
    """
    Display a chat message with optional sources.

    Args:
        role: Message role ('user' or 'assistant')
        content: Message content
        sources: List of source documents
        message_id: Unique identifier for this message to avoid key conflicts
    """
    import hashlib

    # Generate a unique message ID if not provided
    if message_id is None:
        message_id = hashlib.md5(f"{role}_{content[:50]}".encode()).hexdigest()[:8]

    with st.chat_message(role):
        st.write(content)

        if sources and len(sources) > 0:
            with st.expander(f"📚 View Sources ({len(sources)} documents)"):
                for idx, source in enumerate(sources, 1):
                    st.markdown(
                        f"**Source {idx}: {source.get('filename', 'Unknown')}** "
                        f"(Chunk {source.get('chunk_index', 0)}, "
                        f"Score: {source.get('score', 0):.3f})"
                    )

                    content_text = source.get('content', 'No content available')
                    # Create unique key using message_id to avoid conflicts
                    unique_key = f"source_{message_id}_{idx}_{source.get('filename', 'unknown')}_{source.get('chunk_index', 0)}"
                    st.text_area(
                        f"Content from {source.get('filename', 'Unknown')}",
                        value=content_text,
                        height=150,
                        key=unique_key,
                        disabled=True
                    )

                    if idx < len(sources):
                        st.divider()


# ============================================================================
# Sidebar Components
# ============================================================================

def create_sidebar(
    current_tab: str,
    chat_threads: List[str],
    on_tab_change: Callable,
    on_new_chat: Callable,
    on_thread_select: Callable,
    on_thread_delete: Callable
):
    """
    Create application sidebar with navigation and chat history.

    Args:
        current_tab: Currently active tab
        chat_threads: List of chat thread IDs
        on_tab_change: Callback for tab change
        on_new_chat: Callback for new chat
        on_thread_select: Callback for thread selection
        on_thread_delete: Callback for thread deletion
    """
    with st.sidebar:
        st.title("🤖 RAG Pipeline")
        st.divider()

        # Navigation
        st.subheader("📑 Navigation")

        if st.button("📤 Document Ingestion", use_container_width=True,
                     type="primary" if current_tab == "ingestion" else "secondary"):
            on_tab_change("ingestion")
            st.rerun()

        if st.button("💬 Chat", use_container_width=True,
                     type="primary" if current_tab == "chat" else "secondary"):
            on_tab_change("chat")
            st.rerun()

        st.divider()

        # Chat history (only show on chat tab)
        if current_tab == "chat":
            st.subheader("💬 Chat History")

            if st.button("➕ New Chat", use_container_width=True, type="primary"):
                on_new_chat()
                st.rerun()

            st.divider()

            if chat_threads:
                st.caption(f"Recent Conversations ({len(chat_threads)})")

                for idx, thread_id in enumerate(chat_threads[:10]):  # Show last 10
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        # Show shortened thread ID
                        short_id = thread_id[:8]
                        if st.button(f"💬 {short_id}...", key=f"thread_{idx}",
                                     use_container_width=True):
                            on_thread_select(thread_id)
                            st.rerun()

                    with col2:
                        if st.button("🗑️", key=f"delete_{idx}"):
                            on_thread_delete(thread_id)
                            st.rerun()
            else:
                st.info("No chat history yet")

        st.divider()

        # System info
        st.caption("Made with ❤️ using Streamlit")


# ============================================================================
# Message Components
# ============================================================================

def show_success_message(message: str):
    """
    Display a success message.

    Args:
        message: Success message to display
    """
    st.success(f"✅ {message}")


def show_error_message(message: str):
    """
    Display an error message.

    Args:
        message: Error message to display
    """
    st.error(f"❌ {message}")


def show_info_message(message: str):
    """
    Display an info message.

    Args:
        message: Info message to display
    """
    st.info(f"ℹ️ {message}")


def show_warning_message(message: str):
    """
    Display a warning message.

    Args:
        message: Warning message to display
    """
    st.warning(f"⚠️ {message}")


# ============================================================================
# Loading Components
# ============================================================================

def show_spinner(message: str = "Loading..."):
    """
    Context manager for showing a loading spinner.

    Args:
        message: Loading message

    Returns:
        Context manager
    """
    return st.spinner(message)


def create_progress_bar():
    """
    Create a progress bar.

    Returns:
        Progress bar object
    """
    return st.progress(0)
