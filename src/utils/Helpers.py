"""Helper utilities for the RAG pipeline.

Only `ensure_index_exists` remains — the legacy SQLite conversation-memory helpers were removed
with the Streamlit app (history now lives in Supabase; see src/storage/supabase_store.py)."""


def ensure_index_exists(vector_store, index_name: str, embedder) -> None:
    """Ensure a Pinecone index exists, creating it (at the embedder's dimension) if not."""
    if index_name not in vector_store.list_indexes():
        print(f"   [CREATE] Creating new index: {index_name}")
        vector_store.create_index(
            index_name=index_name,
            dimension=embedder.get_embedding_dimension(),
            metric="cosine",
        )
    else:
        print(f"   [OK] Index '{index_name}' already exists")
