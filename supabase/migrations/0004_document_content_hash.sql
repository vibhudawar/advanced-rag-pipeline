-- Content-hash for ingest deduplication (Phase 1).
--
-- Stores a hash of the uploaded file so re-uploading the same document is skipped instead of
-- creating duplicate vectors (which would double-count in aggregation queries). Nullable +
-- idempotent, so it applies cleanly to existing rows and re-runs.

alter table documents add column if not exists content_hash text;

create index if not exists idx_documents_user_hash on documents (user_id, content_hash);
