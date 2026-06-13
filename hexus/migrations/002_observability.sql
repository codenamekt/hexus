-- 002_observability.sql — hexus observability schema for agent-of-agents delegations.
--
-- ONE new table:
--   delegations     → records when a parent agent delegates work to a child agent/session
--
-- Scoped by agent_identity and parent_session_id. Includes 384-dim embeddings and HNSW/ FTS indexing.

CREATE TABLE IF NOT EXISTS delegations (
  id                BIGSERIAL PRIMARY KEY,
  parent_session_id TEXT NOT NULL,
  child_session_id  TEXT NOT NULL,
  agent_identity    TEXT NOT NULL DEFAULT 'default',
  task              TEXT NOT NULL,
  result            TEXT NOT NULL,
  ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
  embedding         vector(384),
  metadata          JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Indexing for lookup by parent/child sessions and ordering by timeline
CREATE INDEX IF NOT EXISTS ix_delegations_parent_session
  ON delegations (parent_session_id, ts DESC);

CREATE INDEX IF NOT EXISTS ix_delegations_child_session
  ON delegations (child_session_id, ts DESC);

CREATE INDEX IF NOT EXISTS ix_delegations_agent_ts
  ON delegations (agent_identity, ts DESC);

-- Semantic recall over delegated tasks/results. Same HNSW tuning as memory_entries.
CREATE INDEX IF NOT EXISTS ix_delegations_embedding_hnsw
  ON delegations USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Full-text search index for hybrid search
CREATE INDEX IF NOT EXISTS ix_delegations_task_result_tsvector
  ON delegations USING gin (to_tsvector('english', task || ' ' || result));
