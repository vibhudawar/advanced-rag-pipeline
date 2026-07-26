# Supabase (data layer)

Postgres schema for the RAG app's **application data** — chat threads, messages, and
ingestion metadata. Designed now so WIN 7 (FastAPI backend + Next.js UI) is just wiring.

## What lives where

| Data | Store | Why |
|---|---|---|
| Chat threads + messages + citations | **Supabase** (`conversations`, `messages`) | queryable, powers the UI; survives restarts |
| Ingestion metadata | **Supabase** (`documents`) | migrated out of the local SQLite `chatbot.db` |
| Document **vectors** | **Pinecone** | already built; Supabase is not the vector store here |
| Agent working memory / resumability | **LangGraph checkpointer** (its own tables) | `AsyncPostgresSaver.setup()` creates them; not modeled here |

The app tables (for display) and the LangGraph checkpointer (for the agent's state) are kept
separate on purpose — the UI queries clean rows, while LangGraph owns its serialized state.

## Apply the schema

Once you've created a Supabase project:

```bash
# Option A — Supabase CLI (recommended)
supabase link --project-ref <your-project-ref>
supabase db push                     # applies supabase/migrations/*.sql

# Option B — paste supabase/migrations/0001_init_schema.sql into the SQL Editor and run it
```

## Auth is deferred (by design)

`user_id` columns already exist (nullable, FK to Supabase's built-in `auth.users`). Until
Google login is added, the backend talks to Postgres with the **service_role** key (bypasses
RLS). When auth lands: start setting `user_id`, then uncomment the RLS block at the bottom of
the migration to enforce per-user isolation. No schema restructuring needed.

## Next (WIN 7)

- Backend reads/writes these tables via `supabase-py` (service role) or SQLAlchemy.
- LangGraph `AsyncPostgresSaver` pointed at the same database (`SqliteSaver` → Postgres).
- Pydantic models mirroring these tables for the FastAPI request/response layer.
