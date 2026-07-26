"""Supabase persistence for chat threads, messages, and ingestion metadata (WIN 7b).

Talks to Supabase over PostgREST via supabase-py using the server-side SECRET key (service
role → bypasses RLS; must never reach the frontend). This is the app's durable memory: the
API loads recent messages as conversation history and appends each turn, so the pipeline
stays stateless per request (the "external store + stateless graph" pattern). Vectors remain
in Pinecone; this only stores display/metadata rows.

If Supabase isn't configured (keys absent), `get_store()` returns None and the API degrades
to no persistence rather than failing — handy for tests/CI.
"""

from __future__ import annotations

from config import SUPABASE_SECRET_KEY, SUPABASE_URL

_store: ConversationStore | None = None
_initialised = False


class ConversationStore:
    def __init__(self, url: str, key: str):
        from supabase import create_client
        self.client = create_client(url, key)

    def create_conversation(self, title: str | None = None, user_id: str | None = None) -> str:
        row = {"title": (title or "New chat")[:120], "user_id": user_id}
        res = self.client.table("conversations").insert(row).execute()
        return res.data[0]["id"]

    def get_history(self, conversation_id: str, limit: int = 12) -> list[tuple[str, str]]:
        res = (
            self.client.table("messages")
            .select("role,content")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .limit(limit)
            .execute()
        )
        return [(m["role"], m["content"]) for m in res.data]

    def owns_conversation(self, conversation_id: str, user_id: str) -> bool:
        """True if `conversation_id` belongs to `user_id`. Application-level isolation while
        RLS is deferred and the backend uses the service-role key."""
        res = (
            self.client.table("conversations")
            .select("id")
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return bool(res.data)

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]:
        """The current user's threads, newest-activity first (the sidebar)."""
        res = (
            self.client.table("conversations")
            .select("id,title,created_at,updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data

    def get_messages(self, conversation_id: str, user_id: str) -> list[dict] | None:
        """Full turns for a thread the user owns, oldest-first. None if not owned/missing."""
        if not self.owns_conversation(conversation_id, user_id):
            return None
        res = (
            self.client.table("messages")
            .select("role,content,citations,created_at")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .execute()
        )
        return res.data

    def add_message(self, conversation_id: str, role: str, content: str,
                    citations: list[dict] | None = None, metadata: dict | None = None) -> None:
        self.client.table("messages").insert({
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "citations": citations,
            "metadata": metadata,
        }).execute()

    def save_document(self, filename: str, file_type: str | None, file_size: int | None,
                      num_chunks: int | None, pinecone_index: str | None, status: str,
                      ingestion_time_s: float | None = None, error: str | None = None,
                      user_id: str | None = None) -> dict | None:
        res = self.client.table("documents").insert({
            "filename": filename, "file_type": file_type, "file_size": file_size,
            "num_chunks": num_chunks, "pinecone_index": pinecone_index, "status": status,
            "ingestion_time_s": ingestion_time_s, "error": error, "user_id": user_id,
        }).execute()
        return res.data[0] if res.data else None

    def list_documents(self, user_id: str, limit: int = 100) -> list[dict]:
        """The current user's ingested documents, newest first (the Ingest tab table)."""
        res = (
            self.client.table("documents")
            .select("id,filename,file_type,file_size,num_chunks,status,ingestion_time_s,error,created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data


def get_store() -> ConversationStore | None:
    """Return a shared store, or None if Supabase isn't configured."""
    global _store, _initialised
    if not _initialised:
        _initialised = True
        if SUPABASE_URL and SUPABASE_SECRET_KEY:
            _store = ConversationStore(SUPABASE_URL, SUPABASE_SECRET_KEY)
    return _store
