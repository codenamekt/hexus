-- 004_enhancements.sql — Support Headroom compression and deduplication.
--
-- Adds compressed and content_hash columns for reversible caching and SHA-256 deduplication.

ALTER TABLE memory_entries
  ADD COLUMN IF NOT EXISTS compressed TEXT,
  ADD COLUMN IF NOT EXISTS content_hash BYTEA;

CREATE INDEX IF NOT EXISTS idx_memory_compressed_fts
  ON memory_entries USING gin(to_tsvector('english', COALESCE(compressed, '')));
