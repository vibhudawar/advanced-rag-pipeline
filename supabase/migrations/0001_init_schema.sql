-- RAG app schema (Supabase / Postgres).
--
-- Scope: the APPLICATION's own data — chat threads, messages, and ingestion metadata — that
-- the UI reads. NOT in scope here:
--   * Vectors/embeddings  -> live in Pinecone, not Postgres.
--   * LangGraph memory     -> AsyncPostgresSaver.setup() creates its own tables
--                             (checkpoints, checkpoint_writes, checkpoint_blobs) at WIN 7.
--                             Don't hand-model those; they are the agent's runtime state.
--
-- Auth is deferred: user_id columns exist now (nullable, FK to Supabase's built-in auth.users),
-- so when Google login lands we just start populating them and enable the RLS policies at the
-- bottom — no restructuring. Until then the FastAPI backend uses the service_role key.

create extension if not exists "pgcrypto";  -- for gen_random_uuid()

-- ---------------------------------------------------------------------------
-- conversations: one row per chat thread (what the sidebar lists)
-- ---------------------------------------------------------------------------
create table if not exists conversations (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid references auth.users (id) on delete cascade,  -- nullable until auth
    title       text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- messages: user / assistant turns (what the chat renders)
--   citations: assistant grounding, e.g. [{"n":1,"doc_id":"...","snippet":"...","score":0.87}]
--   metadata : run info, e.g. {"model":"gpt-4o-mini","pipeline":"snippet_gate",
--                              "latency_ms":3300,"cost_usd":0.0012,"abstained":false}
-- ---------------------------------------------------------------------------
create table if not exists messages (
    id               uuid primary key default gen_random_uuid(),
    conversation_id  uuid not null references conversations (id) on delete cascade,
    role             text not null check (role in ('user', 'assistant', 'system')),
    content          text not null,
    citations        jsonb,
    metadata         jsonb,
    created_at       timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- documents: ingestion metadata (migrated out of the SQLite chatbot.db `documents` table)
-- ---------------------------------------------------------------------------
create table if not exists documents (
    id                uuid primary key default gen_random_uuid(),
    user_id           uuid references auth.users (id) on delete cascade,  -- nullable until auth
    filename          text not null,
    file_type         text,
    file_size         bigint,
    num_chunks        integer,
    pinecone_index    text,                                       -- which index holds the chunks
    status            text not null default 'pending'
                          check (status in ('pending', 'success', 'failed')),
    ingestion_time_s  real,
    error             text,
    created_at        timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- indexes for the common access patterns
-- ---------------------------------------------------------------------------
create index if not exists idx_messages_conversation      on messages (conversation_id, created_at);
create index if not exists idx_conversations_user_updated on conversations (user_id, updated_at desc);
create index if not exists idx_documents_user_created     on documents (user_id, created_at desc);

-- keep conversations.updated_at fresh when a message is added
create or replace function touch_conversation() returns trigger as $$
begin
    update conversations set updated_at = now() where id = new.conversation_id;
    return new;
end;
$$ language plpgsql;

drop trigger if exists trg_touch_conversation on messages;
create trigger trg_touch_conversation
    after insert on messages
    for each row execute function touch_conversation();

-- ---------------------------------------------------------------------------
-- Row Level Security — now ENABLED in 0003_enable_rls.sql (per-user isolation).
-- The backend uses the service_role key, which bypasses RLS, so RLS is defense-in-depth
-- rather than the primary control (the API enforces ownership in application code).
-- ---------------------------------------------------------------------------
