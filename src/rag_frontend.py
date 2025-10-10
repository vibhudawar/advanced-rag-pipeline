"""
RAG Frontend Application
=========================
Streamlit-based frontend for the RAG pipeline.

Usage:
    streamlit run src/rag_frontend.py
"""

import streamlit as st
import uuid
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

# Import backend functions
from src.main_app import ingest_documents, retrieve_and_generate
from src.ingestion.DBIngestion import get_vector_store
from src.utils.Helpers import get_all_documents, retrieve_all_threads, get_conversation_history

# Import UI components
from src.ui.components import (
    create_file_uploader,
    display_uploaded_files,
    create_chunking_config,
    display_ingestion_results_table,
    display_documents_table,
    display_chat_message,
    create_sidebar,
    show_error_message,
    show_success_message
)

# Constants
INDEX_NAME = "rag-documents"


# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "ingestion"

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    if "chat_threads" not in st.session_state:
        st.session_state.chat_threads = retrieve_all_threads()

    if "ingestion_results" not in st.session_state:
        st.session_state.ingestion_results = None


# ============================================================================
# Helper Functions
# ============================================================================

def generate_thread_id() -> str:
    """Generate a new thread ID."""
    return str(uuid.uuid4())


def reset_chat():
    """Reset chat and create new conversation."""
    st.session_state.thread_id = generate_thread_id()
    st.session_state.message_history = []
    st.session_state.chat_threads = retrieve_all_threads()


def load_conversation(thread_id: str):
    """Load conversation from a thread."""
    st.session_state.thread_id = thread_id
    messages = get_conversation_history(thread_id)

    # Convert to session state format
    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({"role": "user", "content": msg.content, "sources": []})
        elif isinstance(msg, AIMessage):
            temp_messages.append({"role": "assistant", "content": msg.content, "sources": []})

    st.session_state.message_history = temp_messages


def remove_uploaded_file(index: int):
    """Remove a file from uploaded files list."""
    if 0 <= index < len(st.session_state.uploaded_files):
        st.session_state.uploaded_files.pop(index)


def handle_tab_change(new_tab: str):
    """Handle tab change."""
    st.session_state.current_tab = new_tab


def handle_thread_delete(thread_id: str):
    """Handle thread deletion."""
    if thread_id in st.session_state.chat_threads:
        st.session_state.chat_threads.remove(thread_id)

    if thread_id == st.session_state.thread_id:
        reset_chat()


# ============================================================================
# Document Ingestion Tab
# ============================================================================

def render_ingestion_tab():
    """Render the document ingestion tab."""
    st.header("Document Ingestion")
    st.info("**Supported file types:** PDF, DOCX, TXT, MD")

    # Chunking configuration
    config = create_chunking_config()
    st.divider()

    # File uploader
    st.subheader("Upload Documents")
    uploaded = create_file_uploader(accept_multiple_files=True, key="file_uploader")

    # Add uploaded files to session state
    if uploaded:
        for file in uploaded:
            if file not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(file)

    # Display uploaded files
    if st.session_state.uploaded_files:
        display_uploaded_files(
            st.session_state.uploaded_files,
            on_remove=remove_uploaded_file
        )
        st.divider()

        # Process button
        if st.button("Process Documents", type="primary", use_container_width=True):
            process_documents(config)

    st.divider()

    # Show existing documents
    try:
        vector_store = get_vector_store()
        documents = get_all_documents(vector_store, INDEX_NAME)
        if documents:
            display_documents_table(documents)
    except Exception as e:
        st.info(f"Unable to retrieve document statistics: {str(e)}")


def process_documents(config: Dict[str, Any]):
    """Process uploaded documents through ingestion pipeline with live logs."""
    import sys
    from io import StringIO
    import contextlib

    if not st.session_state.uploaded_files:
        show_error_message("No files to process!")
        return

    st.subheader("📤 Processing Documents...")

    # Create UI elements
    log_expander = st.expander("📋 Processing Logs", expanded=True)

    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Start processing indicator
    status_text.info("🔄 Initializing document processing pipeline...")

    # Capture stdout to show logs
    captured_output = StringIO()

    try:
        # Context manager to capture stdout
        with contextlib.redirect_stdout(captured_output):
            # Call backend
            result = ingest_documents(
                uploaded_files=st.session_state.uploaded_files,
                index_name=INDEX_NAME,
                chunking_strategy=config["strategy"],
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"]
            )

        # Get captured logs
        logs = captured_output.getvalue()

        # Display logs in the expander with syntax highlighting
        with log_expander:
            if logs:
                # Parse and colorize logs for better readability
                st.text_area(
                    "Console Output",
                    value=logs,
                    height=400,
                    key="ingestion_logs"
                )
            else:
                st.warning("No logs captured during processing.")

        # Update progress
        progress_bar.progress(1.0)

        # Display results
        if result["status"] == "success":
            status_text.success(f"✅ Successfully processed {result['files_processed']} files! ({result['num_chunks']} chunks created)")

            # Show detailed file results
            if "file_results" in result:
                st.divider()
                st.subheader("📊 Processing Results")
                display_ingestion_results_table(result["file_results"])

                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files", result['files_processed'])
                with col2:
                    st.metric("Total Chunks", result['num_chunks'])
                with col3:
                    successful = sum(1 for f in result['file_results'] if f['status'] == 'success')
                    st.metric("Success Rate", f"{(successful/len(result['file_results'])*100):.0f}%")

            # Clear uploaded files
            st.session_state.uploaded_files = []
            st.session_state.ingestion_results = result

        else:
            error_msg = result.get('error', 'Unknown error')
            status_text.error(f"❌ Ingestion failed: {error_msg}")

            # Show error details in logs if available
            with log_expander:
                st.error(f"**Error Details:** {error_msg}")

    except Exception as e:
        # Get any captured logs
        logs = captured_output.getvalue()

        # Display logs even on error
        with log_expander:
            if logs:
                st.text_area(
                    "Console Output (with errors)",
                    value=logs,
                    height=400,
                    key="ingestion_logs_error"
                )
            st.error(f"**Exception:** {str(e)}")

        status_text.error(f"❌ Error during ingestion: {str(e)}")

        # Show detailed error
        with st.expander("🐛 Error Traceback", expanded=False):
            import traceback
            st.code(traceback.format_exc(), language="python")


# ============================================================================
# Chat Tab
# ============================================================================

def render_chat_tab():
    """Render the RAG query chat tab."""
    st.header("Chat with Your Documents")

    # Display chat history
    for idx, message in enumerate(st.session_state.message_history):
        # Generate unique message ID based on index
        message_id = f"msg_{idx}"
        display_chat_message(
            role=message["role"],
            content=message["content"],
            sources=message.get("sources", []),
            message_id=message_id
        )

    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Add user message
        st.session_state.message_history.append({
            "role": "user",
            "content": user_input,
            "sources": []
        })

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = retrieve_and_generate(
                        query=user_input,
                        index_name=INDEX_NAME,
                        conversation_id=st.session_state.thread_id
                    )

                    if result["status"] == "success":
                        response = result["response"]
                        sources = result.get("sources", [])

                        # Display response
                        st.write(response)

                        # Display sources
                        if sources and len(sources) > 0:
                            with st.expander(f"View Sources ({len(sources)} documents)"):
                                for idx, source in enumerate(sources, 1):
                                    st.markdown(
                                        f"**Source {idx}: {source.get('filename', 'Unknown')}** "
                                        f"(Chunk {source.get('chunk_index', 0)}, "
                                        f"Score: {source.get('score', 0):.3f})"
                                    )

                                    content_text = source.get('content', 'No content available')
                                    st.text_area(
                                        f"Content from {source.get('filename', 'Unknown')}",
                                        value=content_text,
                                        height=150,
                                        key=f"source_{idx}_{source.get('filename', 'unknown')}",
                                        disabled=True
                                    )
                                    st.divider()

                        # Add to history
                        st.session_state.message_history.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })

                    else:
                        error_msg = result.get("error", "Unknown error occurred")
                        response = result.get("response", f"Error: {error_msg}")

                        st.error(error_msg)
                        st.write(response)

                        st.session_state.message_history.append({
                            "role": "assistant",
                            "content": response,
                            "sources": []
                        })

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)

                    st.session_state.message_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })

        # Update threads
        if st.session_state.thread_id not in st.session_state.chat_threads:
            st.session_state.chat_threads.append(st.session_state.thread_id)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="RAG Pipeline",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    # Render sidebar
    create_sidebar(
        current_tab=st.session_state.current_tab,
        chat_threads=st.session_state.chat_threads,
        on_tab_change=handle_tab_change,
        on_new_chat=reset_chat,
        on_thread_select=load_conversation,
        on_thread_delete=handle_thread_delete
    )

    # Render current tab
    if st.session_state.current_tab == "ingestion":
        render_ingestion_tab()
    else:
        render_chat_tab()


if __name__ == "__main__":
    main()
