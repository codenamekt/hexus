-- 003_entities.sql — Index metadata->'entities' JSONB array for fast co-occurrence and graph lookups.
--
-- Adds a GIN index specifically on the entities array inside the metadata jsonb column.

CREATE INDEX IF NOT EXISTS ix_memory_entries_entities_gin
  ON memory_entries USING gin ((metadata->'entities'));

CREATE INDEX IF NOT EXISTS ix_conversations_entities_gin
  ON conversations USING gin ((metadata->'entities'));

