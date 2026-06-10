# memory-pgvector

**Full Project Plan: Fork + Local BERT + MCP Shared Knowledge Base Adapter**
**For Hermes Agent + Any MCP Client (Claude, Cursor, etc.)**

---

> **FORK NOTICE**
> This is **Toby's fork** of `andreab67/hermes-memory-pgvector` v0.3.1, kept under the same repo name (`memory-pgvector`) by design — the upstream's package name, plugin path, schema, and Hermes integration points are deliberately preserved for drop-in compatibility. The work described in this plan adds local BERT embeddings and an MCP server without breaking the upstream contract.
>
> - **Upstream:** `https://github.com/andreab67/hermes-memory-pgvector` (BSD-3-Clause © 2026 Andrea Borghi)
> - **This fork:** `git@github.com:codenamekt/memory-pgvector.git`
> - **Working copy:** `/opt/data/workspace/memory-pgvector/`
> - **Date:** June 2026
> - **Target Hardware:** Intel NUC6i7KYK (i7, 16 GB RAM, CPU-only)
>
> All Python files in this fork carry a `# Forked from andreab67/hermes-memory-pgvector (BSD-3-Clause)` header per upstream license requirements.

---

## Goal

A drop-in Hermes memory plugin and a reusable MCP server that turns Postgres + pgvector into a fully local, shared knowledge base for documents + session data across agents. The whole thing runs offline on a NUC, no LLM in the hot path, no third-party services.

---

## 1. Why This Project Makes Sense (Recap)

- You want a Hermes-native memory provider that is fully offline and lightweight.
- You want the same vector store exposed as a standard MCP server so other agents (Claude, Cursor, etc.) can read/write the shared KB.
- The existing repo already solves 90% of the hard parts (async writer, connection pooling, multi-tenant scoping, Hermes hooks, schema, tests, migrations).
- **Decision:** Fork `https://github.com/andreab67/hermes-memory-pgvector` (not just inspiration). It is the exact building block we need.

## 2. License Confirmation

- **License:** BSD 3-Clause ("New BSD")
- **Copyright:** © 2026 Andrea Borghi
- **Commercial use:** Fully allowed (including closed-source derivatives, SaaS, internal tools, selling products).
- **Obligations:** Keep the original copyright notice + full BSD license text in any distributed copies. Add a clear "Forked from andreab67/hermes-memory-pgvector" note in README.
- **Per-file attribution:** Add a one-line header to the docstring of every Python file we touch or copy: `# Forked from andreab67/hermes-memory-pgvector (BSD-3-Clause)`.

## 3. Hardware & Embedding Model Choice

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Why perfect for the NUC:**
  - ~23M parameters, ~90MB on disk, <500MB RAM resident.
  - Excellent CPU performance (ONNX runtime optional for +20-30% speed, but skip initially).
  - Batch embedding ~10-20 sentences/sec on the i7.
  - Proven quality for semantic search; no GPU required.
- **Pre-warm strategy:** first run downloads the model into `~/.cache/huggingface/`. After that, set `HF_HUB_OFFLINE=1` in production containers for air-gapped reliability.
- **Dependencies to add:**
  - `sentence-transformers`
  - `modelcontextprotocol` (Phase 3, MCP Python SDK)
  - `httpx` (test client for the MCP server)

## 4. High-Level Architecture (Shared Core)

```
memory-pgvector/                  (the fork, repo root)
├── pgvector/                       (kept name for the plugin package, upstream compat)
│   ├── __init__.py                 ← PgvectorMemoryProvider (Hermes hooks)
│   ├── store.py                    ← MemoryStore (Postgres ops, shared with MCP)
│   ├── writer.py                   ← AsyncWriter (daemon drain thread, shared)
│   ├── embed.py                    ← module-level embed() dispatch (local OR HTTP)
│   ├── embedder.py                 ← NEW: LocalBertEmbedder (MiniLM-L6-v2)
│   ├── plugin.yaml
│   └── migrations/001_schema.sql   ← updated to vector(384)
├── mcp_server/                     ← NEW
│   ├── server.py                   ← MCP SDK tools wrapper around MemoryStore
│   └── cli.py                      ← `memory-pgvector-mcp serve --transport stdio|http`
├── tests/                          ← updated for 384-dim
├── docker/                         ← NEW: Docker scaffolding (see Section 5)
│   ├── Dockerfile                  ← multi-stage: deps → runtime
│   ├── compose.yml                 ← profiles: test, dev, mcp
│   ├── entrypoint-test.sh          ← wait for pg → run pytest
│   ├── entrypoint-mcp.sh           ← start the MCP server
│   └── preload-model.py            ← pre-download MiniLM-L6-v2 into image
├── pyproject.toml                  ← updated name, deps, optional [mcp] extra
├── PLAN.md                         ← this file
└── README.md                       ← updated with BERT default + docker quickstart
```

One `MemoryStore` + one `LocalBertEmbedder` instance powers both the Hermes plugin and the MCP server. Cold start hits exactly once per process.

---

## 5. Docker-First Development Environment (NEW)

**Principle:** every test, every smoke check, every MCP server invocation runs through docker. The host never installs Python deps directly. Sibling containers are the dev/test loop. This is the same pattern as the `docker-development` skill.

### 5.1 Image design (multi-stage Dockerfile)

```
Stage 1 — `deps` (builder)
  Base: python:3.11-slim
  • pip install build tooling
  • pip install all runtime + dev deps into a venv
  • pre-download the MiniLM-L6-v2 model into /opt/hf-cache

Stage 2 — `runtime` (production)
  Base: python:3.11-slim
  • Copy the venv from `deps`
  • Copy the preloaded HF cache from `deps`
  • Copy the installed plugin package
  • Non-root user, HF_HUB_OFFLINE=1, TINI as PID 1
  • Entrypoint dispatches on $CMD_PROFILE: test | mcp | shell

Stage 3 — `dev` (local iteration, optional)
  Base: runtime
  • Mount-bind the working tree at /app for live reload
  • pip install -e /app on container start
```

Build targets:
- `docker build --target runtime -t memory-pgvector:latest .`
- `docker build --target dev     -t memory-pgvector:dev .`

### 5.2 docker compose profiles

A single `docker/compose.yml` with three profiles. Operators pick one per command:

| Profile | Services                  | Use case                                   |
|---------|---------------------------|--------------------------------------------|
| `test`  | `pg` (postgres+pgvector), `test` (one-shot) | CI + local test runs                  |
| `dev`   | `pg`                      | Local dev: host runs pytest, container runs DB |
| `mcp`   | `pg`, `mcp`               | Run the MCP server for Claude/Cursor to connect to |

### 5.3 Canonical docker commands

```bash
# Run the full test suite (CI mode, one-shot)
docker compose -f docker/compose.yml --profile test up --abort-on-container-exit --exit-code-from test

# Local dev: bring up Postgres+pgvector, run pytest from host
# `--service-ports` is required to publish 5432 to the host (the compose uses
# `expose:`, not `ports:`, to avoid a host-port conflict with homelab-db).
docker compose -f docker/compose.yml --profile dev up --service-ports -d pg
PG_TEST_DSN="dbname=hermes_test user=postgres host=localhost" \
  pytest tests/

# Start the MCP server
docker compose -f docker/compose.yml --profile mcp up -d
docker compose -f docker/compose.yml logs -f mcp    # tail MCP server logs

# Build the image (cold)
docker build -f docker/Dockerfile -t memory-pgvector:latest .

# Build with the model pre-downloaded (slower build, faster runtime)
docker build -f docker/Dockerfile --target runtime --build-arg PRELOAD_MODEL=1 -t memory-pgvector:latest .
```

### 5.4 Image hygiene

- Non-root user (`hermes`, uid 1000).
- `HF_HUB_OFFLINE=1` in the runtime image to prevent accidental downloads.
- `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`.
- Healthcheck on the `mcp` service.
- Named volume for `pg` data so test runs are isolated but persistent across runs.
- `.dockerignore` excludes `.git`, `.venv`, `__pycache__`, `.pytest_cache`, `*.egg-info`.

### 5.5 CI integration

GitHub Actions workflow (`.github/workflows/test.yml`):
- Service: `pgvector/pgvector:pg16`
- Build: `docker build --target runtime -t memory-pgvector:test .`
- Run: `docker run --rm --network container:<pg> -e PG_TEST_DSN=... memory-pgvector:test pytest`
- No host Python needed in CI; the image is the test environment.

### 5.6 Implementation notes (from Phase 0 build-out)

Three adjustments made during the initial implementation, kept here so future readers know why the plan and the code diverge on these points:

1. **`expose:` not `ports:` for the `pg` service.** The host's port 5432 is already used by `homelab-db` in the homelab compose. Using `expose: ["5432"]` makes the port reachable to sibling containers via the internal docker network (as `pg:5432`) without claiming a host port. Dev profile users opt into publishing with `--service-ports`.
2. **Migration applied by the test entrypoint, not via `/docker-entrypoint-initdb.d/`.** The pgvector base image ships with a pre-existing `/docker-entrypoint-initdb.d/001_schema.sql/` *directory* (not a file) that conflicts with our file bind-mount — `psql` inside the container reports "Is a directory" when the entrypoint sources the file. Mounting the whole `migrations/` directory has the same symptom. Fix: drop the volume mount, rely on the Dockerfile `COPY` of the package (which includes the migrations dir) into the test image, and have the entrypoint apply the SQL via `psql -f /app/pgvector/migrations/001_schema.sql` with a `to_regclass('memory_entries')` guard so re-runs are no-ops.
3. **Entrypoint shebang is `#!/bin/bash`, not `#!/bin/sh`.** The TCP connectivity check uses `/dev/tcp/host/port` which is a bash built-in; `sh`/`dash` silently treat it as a regular file path and the check never actually opens a connection. `python:3.11-slim-bookworm` ships with bash at `/bin/bash`, so no extra package is needed.

---

## 6. Detailed Phased Implementation Plan

### Phase 0 — Setup & Fork (1 day) [DONE]

- [x] Fork the repo on GitHub → `codenamekt/memory-pgvector`
- [ ] Update `pyproject.toml` (name, version, description, dependencies)
- [ ] Update `README.md` (add BERT default, MCP instructions, NUC notes, fork notice)
- [ ] `LICENSE` stays unchanged (it's the original BSD-3-Clause, not our copyright)
- [ ] Add fork attribution header to every Python file's docstring
- [ ] Clone, install in editable mode on NUC, run existing tests
- [ ] **NEW:** Add the docker scaffolding skeleton (see Section 5)
  - `docker/Dockerfile`, `docker/compose.yml`, `docker/entrypoint-*.sh`, `docker/preload-model.py`
  - `.dockerignore`
  - Verify: `docker compose --profile test up` passes against the current 768-dim code (no behavior change yet, just proves the harness works)

### Phase 1 — Local BERT Embedder Swap (2-3 days)

**Sub-step 1a — Add the embedder (no breaking changes)**
- Add `sentence-transformers` to `pyproject.toml` dependencies
- Create `pgvector/embedder.py` with `LocalBertEmbedder` class
- Add a smoke test that loads the model and embeds 3 strings
- Add a sibling-container docker test: `docker compose --profile test up --abort-on-container-exit`

**Sub-step 1b — Wire it in (dual-path)**
- Update `pgvector/embed.py:embed()` to dispatch to `LocalBertEmbedder()` when `base_url` is unset, else fall through to the existing HTTP path
- Run the full smoke test suite in docker — confirm both paths still work
- This step is fully reversible: flip a config key to switch back to HTTP

**Sub-step 1c — Migrate to 384 dims (coordinated single change)**
- `pgvector/embed.py`: change dim guard from 768 to 384 (lines 92-93)
- `pgvector/migrations/001_schema.sql`: `vector(768)` → `vector(384)` (lines 40, 80)
- `tests/test_smoke.py`: update all fake-embedding tests from `[0.1] * 768` to `[0.1] * 384` (~6 sites)
- `pgvector/__init__.py`: update module docstring (line 30) and `DEFAULTS['embed_model']` default
- `README.md`: update config docs (line 152) — embed_url now optional
- `ROADMAP.md`: update milestone table reference (line 23)
- Add new migration `002_bert_384.sql` (idempotent, optional, for existing 768-dim installations that want to migrate)
- Verify: `docker compose --profile test up --abort-on-container-exit` passes
- Verify: `docker run --rm memory-pgvector:latest python -c "from pgvector.embedder import LocalBertEmbedder; e=LocalBertEmbedder(); v=e.embed(['hello']); assert len(v[0])==384"`

**Why this step takes 2-3 days, not 1-2:** the dim constant is in 5+ files. A single coordinated migration is safer than 5 partial commits, and the docker-based verification loop adds a half-day of setup before we get the green light.

### Phase 2 — Hermes Plugin Polish (1 day)

- Keep all existing hooks: `initialize`, `on_memory_write`, `sync_turn`, `prefetch`, `recall_memory` (schema), `recall_conversation` (schema), `shutdown`, `system_prompt_block`, `on_session_switch`.
- **Verify `plugin.yaml` hook list** against hermes-agent's discovery code. Current yaml only declares `on_session_end` (line 8-9) but the class implements more — either it's a hint and class methods are auto-discovered, or it's a real bug. Confirm before shipping.
- Update `DEFAULTS` in `pgvector/__init__.py`:
  - `embed_url`: drop or make optional (default None → use local BERT)
  - `embed_model`: default `"sentence-transformers/all-MiniLM-L6-v2"` (or keep the legacy default and require explicit override — decide before coding)
- Ensure `hermes memory setup` works unchanged — same tool surfaces, new default model.
- Update `pgvector/embedder.py` to load the model lazily on first use (so plugin import is fast).

### Phase 3 — MCP Adapter / Shared KB Server (3-5 days)

- Add dependency: `modelcontextprotocol` (official Python SDK: https://github.com/modelcontextprotocol/python-sdk).
- Create `mcp_server/server.py` that wraps `MemoryStore` directly (not the provider class — the MCP server has no notion of Hermes hooks).
- Expose MCP tools:
  - `memory_retain(items: list[dict])` — add documents/sessions with metadata
  - `memory_recall(query: str, top_k: int = 10, filters: dict = None, session_id: str = None)`
  - `memory_search(docs_only: bool = False, ...)` — pagination, tenant scoping
  - `memory_forget(item_id: int)` — delete by id (operator-confirmed)
- Support stdio / HTTP / SSE transports. Pick stdio + streamable-http as the v1 pair; SSE for legacy clients.
- Add CLI: `memory-pgvector-mcp serve --transport stdio --dsn ...` and the docker compose `mcp` profile as the deployment shape.
- Metadata fields: `source_url`, `project_tag`, `session_id`, `timestamp`, `tenant_id`, `doc_type` (`document` | `session` | `note`).
- Add `tests/test_mcp_server.py` that spins up the MCP server in a sibling docker container, connects via stdio, and round-trips retain/recall/forget.

### Phase 4 — Packaging, Production Hardening & Benchmarks (2-3 days)

- Production docker image is already built in Phase 0 — just verify final size, scan, sign.
- `docker compose --profile mcp up -d` is the production shape: pg + mcp on the same user-defined network.
- Benchmark on the NUC:
  - Cold start: time to first embed (with vs without preloaded model)
  - Throughput: embed rate at batch sizes 1, 8, 32, 128
  - RAM: peak RSS during embed loop with 100k and 1M rows
  - Recall latency: p50/p95 for top-10 search with HNSW at 100k and 1M rows
- Update `tests/` to cover the MCP server round-trip.
- Update `ROADMAP.md` to reflect the new milestones (MCP server, BERT swap).
- One-liner install instructions for Hermes (docker-based).

**Total realistic effort:** 1-2 weeks solo.

---

## 7. Key Technical Notes & Gotchas

- **Async non-blocking writes** → existing `AsyncWriter` (pgvector/writer.py:51).
- **Connection pooling & leak fixes** → v0.3.1, `psycopg_pool.ConnectionPool` with `min=0`, `max=4`, `max_idle=30s`, `max_lifetime=300s` (pgvector/store.py:37).
- **Multi-tenant / per-minion scoping** → `agent_identity` priority chain: header > profile > workspace > 'default' (pgvector/__init__.py:228).
- **No LLM in hot path** → your BERT swap preserves this.
- **HNSW index + hybrid search** → already in schema (m=16, ef_construction=64).
- **768→384 dim swap is invasive** — see Phase 1c, 5+ files.
- **plugin.yaml hooks list is suspect** — verify against hermes-agent upstream (Phase 2).
- **`to_pgvector_literal()` belongs in store.py** — refactor opportunity, low priority.

## 8. Optional Future Extensions (Nice-to-Haves)

- Lightweight reflect step (tiny local LLM reranker, optional).
- Graph-like metadata relations (via JSONB queries).
- Bulk import from files/folders.
- Veracity-style proof counts (simple metadata counter).
- Vector quantization (int8 binary) for >10M row scale.

## 9. Open Questions to Resolve Before Phase 1

- [ ] Should the existing 768-dim HTTP path remain as a fallback, or do we hard-cut to local BERT? (Plan assumes dual-path in 1b, hard-cut in 1c.)
- [ ] Renaming the package: keep `pgvector/` directory or rename to `pgbert/`? (Plan assumes `pgvector/` for backward compat with existing installs.)
- [ ] PyPI publish: yes/no? If yes, `memory-pgvector` as the package name (different from import name `pgvector` for clarity).
- [ ] Should the docker image be published to Docker Hub / GHCR? (Plan assumes local-only for now, publish in Phase 4 if desired.)

## 10. Decision Log

- **2026-06-10:** Plan drafted; review against current fork confirmed 90% accuracy. Identified three under-specified risks: 768→384 invasiveness, plugin.yaml hooks list, model pre-warm strategy. All three are now called out in Phases 1, 2, 5 respectively.
- **2026-06-10:** Project name normalized to `memory-pgvector` everywhere (title, docker image, CLI binary, PyPI package, repo tree label) — no codename, no qualifier.
- **2026-06-10:** Phase 0 docker scaffolding implemented and verified (15/16 tests pass in sibling container, 1 skipped by design). Three adjustments vs the plan-as-written, all documented in §5.6: `expose:` not `ports:`, entrypoint-applied migration, bash shebang.
