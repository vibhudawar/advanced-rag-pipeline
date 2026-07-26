-- Grant the backend's service_role access to the app tables.
--
-- Tables created via a direct psql/pooler connection do NOT inherit Supabase's default
-- privileges, so service_role (used by the backend's SECRET key) gets "permission denied"
-- without explicit grants. Note: service_role bypasses RLS but still needs table-level
-- GRANTs — the two are separate.
--
-- We grant to service_role only. The frontend reads chat data through the backend API (not
-- the anon/publishable key), so no anon grants until auth + RLS land in WIN 7d.

grant usage on schema public to service_role;
grant all privileges on public.conversations to service_role;
grant all privileges on public.messages to service_role;
grant all privileges on public.documents to service_role;
