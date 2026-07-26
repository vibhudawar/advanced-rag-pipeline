-- Row Level Security — per-user isolation (WIN: security hardening).
--
-- Defense-in-depth. The backend talks to Postgres as service_role (the SECRET key), which has
-- the BYPASSRLS attribute, so enabling RLS does NOT change how the app works — the API still
-- reads/writes every row and enforces ownership in application code (owns_conversation, etc.).
-- What this adds: if these tables are ever reached via the anon/authenticated role instead of
-- the backend (e.g. someone querying with the public publishable key), a user can only see and
-- mutate their own rows. Belt-and-suspenders on top of the "frontend goes through the backend"
-- design and the service_role-only GRANTs in 0002.
--
-- Idempotent: safe to re-run (enable is a no-op if already on; policies are dropped first).

alter table conversations enable row level security;
alter table messages      enable row level security;
alter table documents     enable row level security;

-- conversations / documents carry user_id directly.
drop policy if exists "own conversations" on conversations;
create policy "own conversations" on conversations
    for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists "own documents" on documents;
create policy "own documents" on documents
    for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

-- messages have no user_id column; ownership is derived from the parent conversation.
drop policy if exists "own messages" on messages;
create policy "own messages" on messages
    for all
    using (
        auth.uid() = (select c.user_id from conversations c where c.id = conversation_id)
    )
    with check (
        auth.uid() = (select c.user_id from conversations c where c.id = conversation_id)
    );
