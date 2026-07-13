"""hexus — Postgres + hexus memory provider for hermes-agent.
#
# Forked from andreab67/hermes-memory-pgvector (BSD-3-Clause).
#
# Mirrors hermes-agent's built-in `memory` tool entries (MEMORY.md / USER.md
# in tools/memory_tool.py) into a single Postgres table, adds 384-dim
# embeddings for semantic recall, and scopes by `agent_identity` so each
# named agent (marketing / sales / trading / incident / …) has its own
# theme.
#
# Design philosophy: this is a STORAGE LAYER for hermes-agent's native
# memory model, not a new memory model. We don't invent facts, entities,
# trust scores, deriver pipelines, or dialectic synthesis. We give the
# built-in `memory` tool a durable Postgres backing + semantic search,
# nothing more. Honcho went heavy and exploded; this stays lean.
#
# v0.4.0 (hexus fork) — embeddings are produced locally by
# sentence-transformers all-MiniLM-L6-v2 (see hexus.embedder) by
# default. The HTTP-embed path from upstream is preserved as a fallback
# for operators with an existing Ollama / OpenAI-compatible endpoint
# (configure `embed_url` in plugin config to use it).
#
# Config in $HERMES_HOME/config.yaml under plugins.hexus:
#
#     plugins:
#       hexus:
#         dsn:         "dbname=hermes_memory user=hermes host=/var/run/postgresql"
#         # No embed_url → use the local sentence-transformers model
#         embed_url:   null
#         embed_model: "sentence-transformers/all-MiniLM-L6-v2"
#         prefetch_limit: 5
#         min_similarity: 0.30
#         embed_on_write: true
#         scope_default: "current"   # 'current' | 'all'
#         embed_eager_load: false    # set true to load BERT at init
#
# Tools exposed: `recall_memory` (one explicit search tool). All built-in
# memory writes (add/replace/remove) are mirrored automatically via the
# on_memory_write hook — no agent-facing change.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from agent.memory_provider import MemoryProvider
    from tools.registry import tool_error
    from hermes_cli.config import cfg_get
except (
    ImportError
):  # pragma: no cover - standalone smoke tests do not install hermes-agent
    MemoryProvider = None  # type: ignore[assignment]
    tool_error = None  # type: ignore[assignment]
    cfg_get = None  # type: ignore[assignment]

from .embed import embed, EmbeddingError
from .store import MemoryStore
import hashlib
from .writer import AsyncWriter, _PendingWrite
from .pipeline.router import ContentRouter


# Boilerplate / acknowledgement-only turns that are not worth embedding or
# storing. Case-insensitive whole-string match after strip. Combined with
# a length floor (default 40 chars) in _is_noise.
_NOISE_RE = re.compile(
    r"^("
    r"ok(ay)?|thanks?( you)?|thx|ty|np|"
    r"yes|no|sure|got it|done|cool|nice|great|"
    r"continue|please|exit|cancel|stop|quit|"
    r"yeah|yep|nope|alright"
    r")[\s\.\!\?]*$",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

if os.environ.get("HEXUS_DEBUG", "").lower() in ("1", "true", "yes", "on"):
    logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Tool schema — one explicit search over memory_entries
# ---------------------------------------------------------------------------

RECALL_CONVERSATION_SCHEMA = {
    "name": "recall_conversation",
    "description": (
        "Semantic search over past chat turns (every substantive "
        "user/assistant exchange across all sessions). Use this when "
        "the user references something you discussed earlier — last week, "
        "yesterday, in another session — and you need the actual turn "
        "text, not just a durable memory entry. Returns top-K matching "
        "turns with role, content, session_id, and timestamp.\n\n"
        "SCOPES: 'current' (your theme — default), 'session' (current "
        "session only), 'all' (every theme).\n\n"
        "Skip for in-session continuity (already in your context). Skip "
        "for durable facts (use recall_memory instead — that's the "
        "MEMORY.md / USER.md entries the agent decided to remember)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text query describing what to recall.",
            },
            "scope": {
                "type": "string",
                "description": "Theme scope: 'current', 'session', 'all', or a named agent.",
                "default": "current",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (1-20, default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


RECALL_MEMORY_SCHEMA = {
    "name": "recall_memory",
    "description": (
        "Semantic search over durable memory entries (the same entries the "
        "built-in `memory` tool writes to MEMORY.md / USER.md, stored "
        "durably in Postgres with embeddings).\n\n"
        "WHEN TO USE: when the answer might be in a past memory entry that "
        "is NOT already in your system prompt's memory block — older "
        "entries, or entries from another named agent. The current scope's "
        "recent entries are already injected ambient; only use this tool "
        "for deeper / cross-scope recall.\n\n"
        "SCOPES:\n"
        "  'current' — your own theme (default; e.g. 'marketing')\n"
        "  'all'     — across all agent themes\n"
        "  '<name>'  — a specific theme: 'marketing', 'sales', 'trading', 'incident', …"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text query describing what to recall.",
            },
            "scope": {
                "type": "string",
                "description": "Theme scope: 'current', 'all', or a named agent.",
                "default": "current",
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user", "both"],
                "description": "Which store to search. Default 'both'.",
                "default": "both",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (1-20, default 5).",
                "default": 5,
            },
            "min_confidence": {
                "type": "number",
                "description": "Minimum confidence ratio of confirm/(confirm+reject) (0..1). Default 0.",
                "default": 0.0,
            },
        },
        "required": ["query"],
    },
}


RECALL_DELEGATION_SCHEMA = {
    "name": "recall_delegation",
    "description": (
        "Semantic search over subagent delegation tasks and results. "
        "Use this when you need to recall what tasks were delegated to subagents "
        "and what results they returned in previous steps or sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text query describing what delegation task/result to recall.",
            },
            "scope": {
                "type": "string",
                "description": "Theme scope: 'current', 'all', or a named agent.",
                "default": "current",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (1-20, default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


ENTITY_GRAPH_SCHEMA = {
    "name": "entity_graph",
    "description": (
        "Retrieve entities that co-occur with a target entity. "
        "Helps you find related topics, systems, or concepts in memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity_type": {
                "type": "string",
                "description": "The type of starting entity (e.g. 'docker_image', 'url').",
            },
            "entity_value": {
                "type": "string",
                "description": "The specific entity value (e.g. 'postgres', 'github.com').",
            },
            "scope": {
                "type": "string",
                "description": "Theme scope: 'current', 'all', or a named agent.",
                "default": "current",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (1-20, default 5).",
                "default": 5,
            },
        },
        "required": ["entity_type", "entity_value"],
    },
}

GRAPH_WALK_SCHEMA = {
    "name": "graph_walk",
    "description": (
        "Traverse the co-occurrence graph up to N hops away from a start entity. "
        "Lets you trace multi-hop connections and related concepts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity_type": {
                "type": "string",
                "description": "Type of starting entity.",
            },
            "entity_value": {
                "type": "string",
                "description": "Value of starting entity.",
            },
            "scope": {
                "type": "string",
                "description": "Theme scope: 'current', 'all', or a named agent.",
                "default": "current",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum hops to walk (1-5, default 2).",
                "default": 2,
            },
            "limit": {
                "type": "integer",
                "description": "Max results (1-20, default 5).",
                "default": 5,
            },
        },
        "required": ["entity_type", "entity_value"],
    },
}

COMMON_TOPICS_SCHEMA = {
    "name": "common_topics",
    "description": "Retrieve clusters/cliques of heavily co-occurring entities.",
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "description": "Theme scope: 'current', 'all', or a named agent.",
                "default": "current",
            },
            "min_strength": {
                "type": "integer",
                "description": "Minimum count of co-occurrences.",
                "default": 2,
            },
            "limit": {
                "type": "integer",
                "description": "Max results (1-20, default 10).",
                "default": 10,
            },
        },
        "required": [],
    },
}

CONFIRM_MEMORY_SCHEMA = {
    "name": "confirm_memory",
    "description": "Increment confirm_count in metadata JSONB for the given entry ID to confirm its relevance.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "The integer ID of the memory entry to confirm.",
            },
        },
        "required": ["id"],
    },
}

REJECT_MEMORY_SCHEMA = {
    "name": "reject_memory",
    "description": "Increment reject_count in metadata JSONB for the given entry ID to flag it as noise.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "The integer ID of the memory entry to reject.",
            },
        },
        "required": ["id"],
    },
}

SUMMARIZE_SESSION_SCHEMA = {
    "name": "summarize_session",
    "description": "Compute the vector centroid of a session's turns and find the K closest turns.",
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "The session identifier to summarize.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (1-20, default 5).",
                "default": 5,
            },
        },
        "required": ["session_id"],
    },
}

HEADROOM_RETRIEVE_SCHEMA = {
    "name": "headroom_retrieve",
    "description": (
        "Retrieve the original full content of a memory entry by its integer ID. "
        "Use this when a memory entry returned by recall_memory contains a compressed/truncated "
        "version of the content, and you need to inspect the full detail."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "The integer ID of the memory entry to retrieve.",
            },
        },
        "required": ["id"],
    },
}

MEMORY_STATS_SCHEMA = {
    "name": "memory_stats",
    "description": "Return metrics from Hexus database and background async queue stats.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


# ---------------------------------------------------------------------------
# Config

# ---------------------------------------------------------------------------

DEFAULTS = {
    "dsn": "dbname=hermes_memory user=hermes host=/var/run/postgresql connect_timeout=5",
    # embed_url=None means "use the local sentence-transformers model"
    # (see hexus.embedder.LocalBertEmbedder). Set to an HTTP URL
    # (e.g. "http://ollama:11434") to fall back to the OpenAI-compatible
    # /v1/embeddings + Ollama-native /api/embed dispatch in embed.py.
    "embed_url": None,
    # sentence-transformers checkpoint name. Default is MiniLM-L6-v2
    # (384-dim, ~90MB, <500MB RAM, ~20-50 sentences/sec on the NUC i7).
    # The HTTP path uses this only for the OpenAI-compat request body
    # (the Ollama-native path uses whatever the server is configured with).
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "prefetch_limit": 5,
    "min_similarity": 0.30,
    "embed_on_write": True,
    "scope_default": "current",
    "write_queue_maxsize": 256,
    # v0.1.1 — bulk sync MEMORY.md / USER.md on init
    "bulk_sync_on_init": True,
    # v0.2 — conversation turn capture
    "sync_turns": True,
    "turn_min_chars": 40,  # turns shorter than this are noise unless > 200 chars or contain tool refs
    # v0.4.0 — expected embedding dim. Local BERT is 384; HTTP path
    # must also produce 384-dim vectors (or the operator must override
    # this in their plugin config). The embed layer validates the dim
    # at HTTP-response time so a misconfigured model fails fast.
    "expected_dim": 384,
    # v0.4.0 — eagerly load the local embedder at plugin init?
    # Default False: keep import + init fast, pay the cold-start cost
    # on the first embed call. Set True if you want the model loaded
    # on a known thread with visible log output, e.g. on the NUC's
    # gateway boot path.
    "embed_eager_load": False,
    "entity_extractor_enabled": True,
    "entity_extractor_patterns": None,
    "webhook_url": None,
    "webhook_secret": None,
}


def _load_plugin_config() -> dict:
    try:
        from hermes_constants import get_hermes_home

        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return {}
        import yaml

        with open(config_path, encoding="utf-8-sig") as fh:
            data = yaml.safe_load(fh) or {}
        if cfg_get is None:
            return {}
        expanded = _expand_config_vars(
            cfg_get(data, "plugins", "hexus", default={}) or {}
        )
        return expanded if isinstance(expanded, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _expand_config_vars(obj):
    """Expand env references in plugin config values.

    Hermes's normal config loader expands plain ``${VAR}`` references, but
    this plugin reads the YAML directly so it can run before Hermes has
    necessarily loaded the expanded config. Support both plain references and
    the shell-style forms already used by the homelab config:

      ``${VAR}``        → env value or unchanged placeholder
      ``${VAR:-default}`` → env value or default
      ``${VAR:?message}`` → env value or ValueError
    """
    if isinstance(obj, str):
        return _ENV_REF_RE.sub(_expand_env_match, obj)
    if isinstance(obj, dict):
        return {key: _expand_config_vars(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_expand_config_vars(value) for value in obj]
    return obj


_ENV_REF_RE = re.compile(r"\$\{([^}:]+)(?::([\-?])((?:[^}]|\\})+))?\}")


def _expand_env_match(match: re.Match[str]) -> str:
    name = match.group(1)
    op = match.group(2)
    payload = match.group(3) or ""

    value = os.environ.get(name)
    if value is not None:
        return value
    if op == "-":
        return payload.replace("\\}", "}")
    if op == "?":
        message = payload.replace("\\}", "}")
        raise ValueError(f"missing required environment variable {name}: {message}")
    return match.group(0)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class HexusMemoryProvider(MemoryProvider or object):
    """Postgres mirror of built-in memory entries, with semantic recall."""

    def __init__(self, config: dict | None = None):
        if MemoryProvider is None:
            raise RuntimeError(
                "HexusMemoryProvider requires Hermes Agent internals; "
                "install this package inside Hermes Agent to use the provider."
            )
        self._config = {**DEFAULTS, **(config or {})}
        self._store: Optional[MemoryStore] = None
        self._writer: Optional[AsyncWriter] = None
        self._agent_identity: str = "default"
        self._session_id: str = ""
        self._healthy = False
        self._embed_warned = False
        self._last_md_mtimes: Dict[str, float] = {}
        self._hermes_home: Optional[str] = None
        self._content_router = ContentRouter()

    @property
    def name(self) -> str:
        return "hexus"

    # -- Lifecycle -----------------------------------------------------------

    def is_available(self) -> bool:
        try:
            import psycopg  # noqa: F401

            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home")
        # Per-agent theme scoping — priority order:
        #   1. gateway_session_key — from the `X-Hermes-Session-Key` header on
        #      API requests. This is the EXPLICIT minion-scope signal sent by
        #      systemd-run callers (marketing-daily, sales-daily, intraday
        #      workers, …) and takes precedence over the profile fallback
        #      because the gateway always sets agent_identity='default' for
        #      API traffic — without prioritising the header, every minion
        #      collapses to one shared 'default' scope.
        #   2. agent_identity ≠ 'default' — explicit profile name from CLI
        #      (`hermes --profile marketing`). Skipped when it's the
        #      auto-default sentinel to allow header (#1) to win.
        #   3. agent_workspace — shared workspace name from some platforms.
        #   4. agent_identity == 'default' — accept it now (no other source).
        #   5. 'default'        — last-resort bucket for unscoped traffic.
        explicit_identity = kwargs.get("agent_identity")
        if explicit_identity == "default":
            explicit_identity = None  # sentinel — let header take over
        self._agent_identity = (
            kwargs.get("gateway_session_key")
            or explicit_identity
            or kwargs.get("agent_workspace")
            or kwargs.get("agent_identity")  # accept 'default' if nothing else set
            or "default"
        )
        logger.debug(
            "hexus: initialized HexusMemoryProvider. session_id=%s, hermes_home=%s, agent_identity=%s, healthy_initially=%s",
            self._session_id,
            self._hermes_home,
            self._agent_identity,
            self._healthy,
        )

        # Re-initialization guard (v0.3.1): one registered provider instance
        # can have initialize() called again for a new session — the gateway
        # reuses the registered provider rather than constructing a fresh one
        # per session. Without tearing down the previous session's writer +
        # pool first, each re-init abandoned a ConnectionPool whose warm
        # connection lingered in Postgres until idle_session_timeout — the
        # v0.3.0 connection leak that saturated the server's slots under a
        # burst of concurrent sessions. shutdown() is idempotent and drains
        # in-flight writes, so calling it unconditionally here is safe.
        if self._store is not None or self._writer is not None:
            self.shutdown()

        self._store = MemoryStore(
            self._config["dsn"],
            entity_extractor_enabled=self._config.get("entity_extractor_enabled", True),
            entity_extractor_patterns=self._config.get("entity_extractor_patterns"),
        )
        try:
            # Schema is verify-only at runtime — admin applies the
            # migration out-of-band (see plugin README install step).
            self._store.ensure_schema()
            health = self._store.health()
            self._healthy = bool(health.get("ok"))
            if not self._healthy:
                logger.warning("hexus unhealthy on init: %s", health.get("error"))
        except MemoryStore.SchemaNotApplied as exc:
            logger.error("hexus schema not applied — %s", exc)
            self._healthy = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("hexus init failed: %s", exc)
            self._healthy = False

        # Background writer — bounded queue, lazy thread start. Decouples
        # on_memory_write + sync_turn from the (potentially slow) embed +
        # DB write so the agent loop never blocks on a stalled embed
        # endpoint.
        self._writer = AsyncWriter(
            self._worker,
            maxsize=int(self._config.get("write_queue_maxsize", 256)),
        )

        # v0.4.0 — optionally warm the local embedder now so the cold
        # start lands on a known thread with a visible log line, rather
        # than on the first user-facing embed call. Default False.
        if self._config.get("embed_eager_load", False) and not self._config.get(
            "embed_url"
        ):
            try:
                from .embedder import get_default_embedder, DEFAULT_MODEL

                get_default_embedder(
                    model_name=self._config.get("embed_model") or DEFAULT_MODEL
                ).ensure_loaded()
            except Exception as exc:  # noqa: BLE001
                logger.warning("hexus eager embed load failed: %s", exc)

        # v0.1.1: bulk import existing MEMORY.md / USER.md content so the
        # plugin sees pre-plugin entries + direct file edits, not just the
        # new writes captured via on_memory_write.
        if self._healthy and self._config.get("bulk_sync_on_init", True):
            self._bulk_sync_from_disk(self._hermes_home)

        # Dispatch session_new webhook event
        if self._config.get("webhook_url"):
            from .webhook.dispatcher import dispatch_webhook

            dispatch_webhook(
                url=self._config.get("webhook_url"),
                secret=self._config.get("webhook_secret"),
                event="session_new",
                payload={
                    "session_id": self._session_id,
                    "agent_identity": self._agent_identity,
                },
            )

    def shutdown(self) -> None:
        # Drain the in-flight writes first so we don't drop work...
        if self._writer:
            self._writer.shutdown(timeout=5.0)
            self._writer = None
        # ...then close the pool the writer was draining into.
        if self._store:
            self._store.close()
            self._store = None
        self._healthy = False

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        logger.debug("hexus: on_session_switch to session_id=%s", new_session_id)
        self._session_id = new_session_id

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Called at the start of each turn. Check if memory/skill files on disk have changed and re-sync."""
        logger.debug(
            "hexus: on_turn_start. turn_number=%d, message_len=%d",
            turn_number,
            len(message) if message else 0,
        )
        if self._healthy:
            self._check_and_sync_markdown_files()
            self._sync_skills_from_disk()

    def _check_and_sync_markdown_files(self) -> None:
        """Check modification times of MEMORY.md and USER.md on disk, and trigger sync if changed."""
        if not self._hermes_home:
            try:
                from hermes_constants import get_hermes_home

                self._hermes_home = str(get_hermes_home())
            except Exception:
                return

        memories_dir = Path(self._hermes_home) / "memories"
        changed = False
        for fname in ("MEMORY.md", "USER.md"):
            fpath = memories_dir / fname
            if fpath.exists():
                try:
                    mtime = os.path.getmtime(fpath)
                    if (
                        fname not in self._last_md_mtimes
                        or mtime > self._last_md_mtimes[fname]
                    ):
                        changed = True
                        self._last_md_mtimes[fname] = mtime
                except Exception as exc:
                    logger.debug("hexus failed to get mtime for %s: %s", fname, exc)

        if changed:
            logger.info(
                "hexus: detected local changes in memory markdown files, starting sync"
            )
            self._bulk_sync_from_disk(self._hermes_home)

    def _sync_skills_from_disk(self) -> None:
        if not self._store or not self._hermes_home:
            logger.debug(
                "hexus: skills sync skipped. store_exists=%s, hermes_home=%s",
                self._store is not None,
                self._hermes_home,
            )
            return

        skills_dir = Path(self._hermes_home) / "skills"
        logger.debug("hexus: scanning skills directory: %s", skills_dir)
        if not skills_dir.exists():
            logger.debug("hexus: skills directory does not exist: %s", skills_dir)
            return

        for skill_file in skills_dir.rglob("SKILL.md"):
            try:
                mtime = os.path.getmtime(skill_file)
                rel_path = str(skill_file.relative_to(skills_dir))

                # Check if we already processed this mtime
                if (
                    rel_path in self._last_md_mtimes
                    and mtime <= self._last_md_mtimes[rel_path]
                ):
                    logger.debug(
                        "hexus: skill '%s' unmodified (mtime: %f), skipping sync",
                        rel_path,
                        mtime,
                    )
                    continue

                # Parse skill_file
                content = skill_file.read_text(encoding="utf-8")
                if not content.strip():
                    logger.debug("hexus: skill '%s' content is empty", rel_path)
                    continue

                # Extract skill name from folder name
                skill_name = skill_file.parent.name
                logger.debug("hexus: syncing skill '%s' (mtime: %f)", skill_name, mtime)

                # Add to DB
                vec = self._maybe_embed(content)
                self._store.add(
                    agent_identity=self._agent_identity,
                    target="memory",
                    content=f"[Skill: {skill_name}] {content}",
                    embedding=vec,
                    metadata={
                        "source": "skill_sync",
                        "skill_name": skill_name,
                        "file_path": str(skill_file),
                    },
                    compressed=None,
                    content_hash=hashlib.sha256(content.encode("utf-8")).digest(),
                )

                self._last_md_mtimes[rel_path] = mtime
                logger.info("hexus: synced skill '%s' from disk", skill_name)
            except Exception as exc:
                logger.debug("hexus failed to sync skill %s: %s", skill_file, exc)

    # -- System prompt + ambient recall --------------------------------------

    def system_prompt_block(self) -> str:
        if not self._healthy or not self._store:
            return ""
        try:
            count_scoped = self._store.count(agent_identity=self._agent_identity)
            count_all = self._store.count()
        except Exception:  # noqa: BLE001
            count_scoped = count_all = 0
        if count_all == 0:
            return (
                "# hexus memory\n"
                "Active. Empty store. Use the built-in `memory` tool to save "
                "durable notes — entries are mirrored to Postgres with "
                "embeddings for semantic recall across sessions."
            )
        return (
            "# hexus memory\n"
            f"Active. {count_scoped} entries for '{self._agent_identity}', "
            f"{count_all} total across all themes. "
            "Use `recall_memory(query, scope='all'|'<theme>')` for deeper / "
            "cross-theme recall beyond what's in the built-in memory block."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._healthy or not self._store or not query:
            return ""
        try:
            vec = embed(
                query,
                base_url=self._config["embed_url"],
                model=self._config["embed_model"],
            )
        except EmbeddingError as exc:
            logger.debug("hexus prefetch embed failed: %s", exc)
            return ""

        # Ambient prefetch is scoped to the current agent_identity by
        # default — keeps marketing turns from polluting trading recall.
        try:
            rows = self._store.search(
                query_embedding=vec,
                agent_identity=self._agent_identity,
                limit=int(self._config.get("prefetch_limit", 5)),
                min_similarity=float(self._config.get("min_similarity", 0.30)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("hexus prefetch query failed: %s", exc)
            return ""
        if not rows:
            return ""

        lines = [f"## Recall (hexus, {self._agent_identity})"]
        for r in rows:
            score = r.get("score") or 0.0
            tgt = r.get("target") or "?"
            content = (r.get("content") or "").strip().replace("\n", " ")
            if len(content) > 280:
                content = content[:280] + "…"
            lines.append(f"- [{score:.2f}] ({tgt}) {content}")
        return "\n".join(lines)

    # -- Turn capture (v0.2) -------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Persist a (user, assistant) turn pair to the conversations table.

        Non-blocking — enqueues writes; the async writer drains, embeds,
        and INSERTs. Boilerplate / very short turns are filtered out so
        the recall table stays high-signal.
        """
        if not self._healthy or not self._writer:
            return
        if not self._config.get("sync_turns", True):
            return

        sid = session_id or self._session_id or "default"
        min_chars = int(self._config.get("turn_min_chars", 40))

        for role, content in (("user", user_content), ("assistant", assistant_content)):
            if not content:
                continue
            if self._is_noise(content, min_chars=min_chars):
                continue
            self._writer.enqueue(
                action="turn",
                agent_identity=self._agent_identity,
                target="conversations",  # synthetic; worker dispatches on action
                content=content,
                extra={"role": role, "session_id": sid},
                metadata={"session_id": sid},
            )

    @staticmethod
    def _is_noise(content: str, *, min_chars: int) -> bool:
        """True for short / boilerplate content we don't want in recall."""
        stripped = (content or "").strip()
        if not stripped:
            return True
        if len(stripped) < min_chars:
            return True
        if _NOISE_RE.match(stripped):
            return True
        return False

    # -- Built-in memory mirror (THE main integration point) ----------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in `memory` tool writes to Postgres (non-blocking).

        Built-in tool fires this on every add/replace/remove. We enqueue
        the write; the background thread drains, embeds, and INSERTs.
        Returns instantly so the agent loop never blocks on the embed
        endpoint or the DB.
        """
        if not self._healthy or not self._writer:
            return
        if target not in ("memory", "user"):
            logger.debug("hexus ignoring unsupported target: %r", target)
            return
        if action not in ("add", "replace", "remove"):
            logger.debug("hexus ignoring unknown action: %r", action)
            return

        meta = dict(metadata or {})
        meta.setdefault("session_id", self._session_id)
        old_text = meta.get("old_text") or meta.get("replaces")

        self._writer.enqueue(
            action=action,
            agent_identity=self._agent_identity,
            target=target,
            content=content,
            extra={"old_text": str(old_text)} if old_text else {},
            metadata=meta,
        )

    def _worker(self, item: "_PendingWrite") -> None:
        """Drain-thread worker: embed + DB write for a single queued item.

        Must NOT raise — the AsyncWriter logs + survives if we do, but
        we still want failures to degrade gracefully (drop the write,
        keep the queue moving).
        """
        if not self._store:
            return
        try:
            if item.action == "add":
                compressed = self._content_router.maybe_compress(item.content)
                hash_target = compressed if compressed is not None else item.content
                content_hash = hashlib.sha256(hash_target.encode("utf-8")).digest()
                vec = self._maybe_embed(item.content)
                self._store.add(
                    agent_identity=item.agent_identity,
                    target=item.target,
                    content=item.content,
                    embedding=vec,
                    metadata=item.metadata,
                    compressed=compressed,
                    content_hash=content_hash,
                )
                if self._config.get("webhook_url"):
                    from .webhook.dispatcher import dispatch_webhook

                    dispatch_webhook(
                        url=self._config.get("webhook_url"),
                        secret=self._config.get("webhook_secret"),
                        event="memory_retain",
                        payload={
                            "agent_identity": item.agent_identity,
                            "target": item.target,
                            "content": item.content,
                            "metadata": item.metadata,
                        },
                    )
            elif item.action == "replace":
                compressed = self._content_router.maybe_compress(item.content)
                hash_target = compressed if compressed is not None else item.content
                content_hash = hashlib.sha256(hash_target.encode("utf-8")).digest()
                old_text = item.extra.get("old_text")
                vec = self._maybe_embed(item.content)
                if old_text:
                    n = self._store.replace(
                        agent_identity=item.agent_identity,
                        target=item.target,
                        old_text=old_text,
                        new_content=item.content,
                        new_embedding=vec,
                        compressed=compressed,
                        content_hash=content_hash,
                    )
                    if n == 0:
                        # Nothing matched — degrade to add (built-in wrote
                        # the new entry to disk; mirror it).
                        self._store.add(
                            agent_identity=item.agent_identity,
                            target=item.target,
                            content=item.content,
                            embedding=vec,
                            metadata=item.metadata,
                            compressed=compressed,
                            content_hash=content_hash,
                        )
                else:
                    # No old_text in metadata → can't locate prior row;
                    # add the new content so we don't lose it.
                    self._store.add(
                        agent_identity=item.agent_identity,
                        target=item.target,
                        content=item.content,
                        embedding=vec,
                        metadata=item.metadata,
                        compressed=compressed,
                        content_hash=content_hash,
                    )
                if self._config.get("webhook_url"):
                    from .webhook.dispatcher import dispatch_webhook

                    dispatch_webhook(
                        url=self._config.get("webhook_url"),
                        secret=self._config.get("webhook_secret"),
                        event="memory_retain",
                        payload={
                            "agent_identity": item.agent_identity,
                            "target": item.target,
                            "content": item.content,
                            "metadata": item.metadata,
                        },
                    )
            elif item.action == "remove":
                self._store.remove(
                    agent_identity=item.agent_identity,
                    target=item.target,
                    old_text=item.content,
                )
                if self._config.get("webhook_url"):
                    from .webhook.dispatcher import dispatch_webhook

                    dispatch_webhook(
                        url=self._config.get("webhook_url"),
                        secret=self._config.get("webhook_secret"),
                        event="memory_forget",
                        payload={
                            "agent_identity": item.agent_identity,
                            "target": item.target,
                            "content": item.content,
                        },
                    )
            elif item.action == "turn":
                role = item.extra.get("role") or "user"
                sid = item.extra.get("session_id") or "default"
                vec = self._maybe_embed(item.content)
                self._store.append_turn(
                    session_id=sid,
                    agent_identity=item.agent_identity,
                    role=role,
                    content=item.content,
                    embedding=vec,
                    metadata=item.metadata,
                )
            elif item.action == "delegation":
                parent_sid = item.metadata.get("parent_session_id") or "default"
                child_sid = item.extra.get("child_session_id") or "default"
                combined_text = (
                    f"Task: {item.content}\nResult: {item.extra.get('result', '')}"
                )
                vec = self._maybe_embed(combined_text)
                self._store.record_delegation(
                    parent_session_id=parent_sid,
                    child_session_id=child_sid,
                    agent_identity=item.agent_identity,
                    task=item.content,
                    result=item.extra.get("result") or "",
                    embedding=vec,
                    metadata=item.metadata,
                )
            elif item.action == "summarize_session_hook":
                summary_text = self._generate_session_summary(item.content)
                if summary_text:
                    vec = self._maybe_embed(summary_text)
                    self._store.add(
                        agent_identity=item.agent_identity,
                        target="memory",
                        content=f"[Session Summary for {item.metadata.get('session_id', '')}] {summary_text}",
                        embedding=vec,
                        metadata={
                            "session_id": item.metadata.get("session_id"),
                            "source": "session_summarizer",
                        },
                        compressed=None,
                        content_hash=hashlib.sha256(
                            summary_text.encode("utf-8")
                        ).digest(),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "hexus worker (%s/%s/%s) failed: %s",
                item.action,
                item.agent_identity,
                item.target,
                str(exc)[:200],
            )

    def _generate_session_summary(self, messages_json: str) -> Optional[str]:
        import urllib.request

        try:
            messages = json.loads(messages_json)
            if not messages:
                return None

            # Format history
            history = []
            for m in messages:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                if content:
                    history.append(f"{role.upper()}: {content}")

            if not history:
                return None

            history_str = "\n".join(history)

            api_base = os.environ.get("LLM_API_BASE") or "http://headroom:8787/v1"
            # Try to get LITELLM_MASTER_KEY or HEADROOM_INTERNAL_TOKEN
            api_key = os.environ.get("HEADROOM_INTERNAL_TOKEN") or os.environ.get(
                "LITELLM_MASTER_KEY"
            )

            url = f"{api_base.rstrip('/')}/chat/completions"
            summary_model = os.environ.get("HEXUS_SUMMARY_MODEL")
            if not summary_model:
                raise ValueError(
                    "HEXUS_SUMMARY_MODEL environment variable is not configured."
                )
            payload = {
                "model": summary_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that summarizes conversation transcripts. "
                            "Write a highly concise, 1-2 sentence summary of the key outcomes, "
                            "user preferences, and facts learned during this session. "
                            "Do not include boilerplate like 'In this session...' or 'The user...'"
                        ),
                    },
                    {"role": "user", "content": f"Transcript:\n{history_str}"},
                ],
                "temperature": 0.3,
                "max_tokens": 150,
            }

            logger.debug(
                "hexus: generating session summary. url=%s, model=%s, api_key_configured=%s, history_turns=%d",
                url,
                summary_model,
                api_key is not None,
                len(history),
            )

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")

            with urllib.request.urlopen(req, timeout=15) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                summary_content = resp_data["choices"][0]["message"]["content"].strip()
                logger.debug(
                    "hexus: session summary successfully generated: %s", summary_content
                )
                return summary_content
        except Exception as e:
            logger.debug("hexus failed to generate session summary via LLM: %s", e)
            return None

    # -- Observability Hooks (M4) --------------------------------------------

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs,
    ) -> None:
        """Called on the parent agent when a subagent finishes."""
        if not self._healthy or not self._writer:
            return

        meta = {
            "parent_session_id": self._session_id,
            "child_session_id": child_session_id,
        }

        self._writer.enqueue(
            action="delegation",
            agent_identity=self._agent_identity,
            target="delegations",
            content=task,
            extra={"result": result, "child_session_id": child_session_id},
            metadata=meta,
        )

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Called before context compaction discards older messages."""
        if not self._healthy or not self._writer or not messages:
            return ""

        compaction_info = f"[CONTEXT COMPACTION] Compacting {len(messages)} turns from session {self._session_id}."
        self._writer.enqueue(
            action="turn",
            agent_identity=self._agent_identity,
            target="conversations",
            content=compaction_info,
            extra={"role": "system", "session_id": self._session_id},
            metadata={"is_compaction": True, "message_count": len(messages)},
        )
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called when a session ends."""
        if not self._healthy or not self._writer:
            return

        end_info = f"[SESSION ENDED] Session {self._session_id} has ended."
        self._writer.enqueue(
            action="turn",
            agent_identity=self._agent_identity,
            target="conversations",
            content=end_info,
            extra={"role": "system", "session_id": self._session_id},
            metadata={"session_end": True},
        )

        if messages and len(messages) >= 2:
            self._writer.enqueue(
                action="summarize_session_hook",
                agent_identity=self._agent_identity,
                target="memory",
                content=json.dumps(messages),
                metadata={"session_id": self._session_id},
            )

        # Dispatch session_end webhook event
        if self._config.get("webhook_url"):
            from .webhook.dispatcher import dispatch_webhook

            dispatch_webhook(
                url=self._config.get("webhook_url"),
                secret=self._config.get("webhook_secret"),
                event="session_end",
                payload={
                    "session_id": self._session_id,
                    "agent_identity": self._agent_identity,
                },
            )

    # -- Bulk sync (v0.1.1) --------------------------------------------------

    def _bulk_sync_from_disk(self, hermes_home: Optional[str]) -> None:
        """Import MEMORY.md + USER.md entries from disk into memory_entries.

        Called by initialize(). Runs synchronously (not via async writer)
        so the table is warm before the first turn's prefetch. Cheap on
        re-init: an existence pre-check skips already-imported entries
        without re-embedding.
        """
        if not self._store:
            return
        if not hermes_home:
            # Fall back to hermes_constants if the runtime didn't pass it.
            try:
                from hermes_constants import get_hermes_home

                hermes_home = str(get_hermes_home())
            except Exception:  # noqa: BLE001
                return

        memories_dir = Path(hermes_home) / "memories"
        embed_fn = self._make_embed_fn()

        for target, fname in (("memory", "MEMORY.md"), ("user", "USER.md")):
            fpath = memories_dir / fname
            try:
                result = self._store.bulk_upsert_md(
                    agent_identity=self._agent_identity,
                    target=target,
                    file_path=fpath,
                    embed_fn=embed_fn,
                )
                # Store modification time to prevent immediate re-sync
                if fpath.exists():
                    self._last_md_mtimes[fname] = os.path.getmtime(fpath)
                if result.get("inserted"):
                    logger.info(
                        "hexus bulk-sync %s: parsed=%d inserted=%d skipped=%d",
                        fname,
                        result.get("parsed", 0),
                        result.get("inserted", 0),
                        result.get("skipped", 0),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("hexus bulk-sync %s failed: %s", fname, exc)

    def _make_embed_fn(self):
        """Return a closure over the configured embed endpoint, or None."""
        if not self._config.get("embed_on_write", True):
            return None
        base_url = self._config["embed_url"]
        model = self._config["embed_model"]

        def _fn(text: str):
            return embed(text, base_url=base_url, model=model)

        return _fn

    # -- Tool surface --------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            RECALL_MEMORY_SCHEMA,
            RECALL_CONVERSATION_SCHEMA,
            RECALL_DELEGATION_SCHEMA,
            ENTITY_GRAPH_SCHEMA,
            GRAPH_WALK_SCHEMA,
            COMMON_TOPICS_SCHEMA,
            CONFIRM_MEMORY_SCHEMA,
            REJECT_MEMORY_SCHEMA,
            SUMMARIZE_SESSION_SCHEMA,
            HEADROOM_RETRIEVE_SCHEMA,
            MEMORY_STATS_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "recall_conversation":
            return self._handle_recall_conversation(args)
        if tool_name == "recall_delegation":
            return self._handle_recall_delegation(args)
        if tool_name == "entity_graph":
            return self._handle_entity_graph(args)
        if tool_name == "graph_walk":
            return self._handle_graph_walk(args)
        if tool_name == "common_topics":
            return self._handle_common_topics(args)
        if tool_name == "confirm_memory":
            return self._handle_confirm_memory(args)
        if tool_name == "reject_memory":
            return self._handle_reject_memory(args)
        if tool_name == "summarize_session":
            return self._handle_summarize_session(args)
        if tool_name == "headroom_retrieve":
            return self._handle_headroom_retrieve(args)
        if tool_name == "memory_stats":
            return self._handle_memory_stats(args)
        if tool_name != "recall_memory":
            return tool_error(f"Unknown tool: {tool_name}")

        if not self._healthy or not self._store:
            return json.dumps({"results": [], "count": 0, "error": "hexus unavailable"})

        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required arg: query")

        try:
            limit = max(1, min(int(args.get("limit", 5)), 20))
        except (TypeError, ValueError):
            limit = 5

        # Scope resolution: 'current' → my agent_identity; 'all' → no filter;
        # anything else → treat as explicit theme name.
        scope = (
            args.get("scope") or self._config.get("scope_default") or "current"
        ).strip()
        if scope == "current":
            agent_filter: Optional[str] = self._agent_identity
        elif scope == "all":
            agent_filter = None
        else:
            agent_filter = scope

        # Target resolution: 'memory'/'user'/'both'.
        target_arg = (args.get("target") or "both").strip()
        target_filter: Optional[str] = None if target_arg == "both" else target_arg
        if target_filter not in (None, "memory", "user"):
            return tool_error(f"Invalid target: {target_arg!r}")

        try:
            min_confidence = float(args.get("min_confidence", 0.0))
        except (TypeError, ValueError):
            min_confidence = 0.0

        try:
            vec = embed(
                query,
                base_url=self._config["embed_url"],
                model=self._config["embed_model"],
            )
        except EmbeddingError as exc:
            return json.dumps({"results": [], "count": 0, "error": f"embed: {exc}"})

        try:
            rows = self._store.search(
                query_embedding=vec,
                agent_identity=agent_filter,
                target=target_filter,
                limit=limit,
                min_confidence=min_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"results": [], "count": 0, "error": f"db: {exc}"})

        results = []
        for r in rows:
            ts = r.get("updated_at") or r.get("created_at")
            results.append(
                {
                    "id": r.get("id"),
                    "agent_identity": r.get("agent_identity"),
                    "target": r.get("target"),
                    "ts": ts.isoformat() if ts else None,
                    "score": round(float(r.get("score") or 0.0), 4),
                    "content": (r.get("content") or "")[:2000],
                }
            )
        return json.dumps({"results": results, "count": len(results)})

    def _handle_recall_conversation(self, args: Dict[str, Any]) -> str:
        """Tool handler for recall_conversation over the conversations table."""
        if not self._healthy or not self._store:
            return json.dumps({"results": [], "count": 0, "error": "hexus unavailable"})

        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required arg: query")
        try:
            limit = max(1, min(int(args.get("limit", 5)), 20))
        except (TypeError, ValueError):
            limit = 5

        scope = (args.get("scope") or "current").strip()
        agent_filter: Optional[str] = None
        session_filter: Optional[str] = None
        if scope == "current":
            agent_filter = self._agent_identity
        elif scope == "session":
            session_filter = self._session_id or None
        elif scope == "all":
            pass  # no filters
        else:
            agent_filter = scope  # treat as a specific theme name

        try:
            vec = embed(
                query,
                base_url=self._config["embed_url"],
                model=self._config["embed_model"],
            )
        except EmbeddingError as exc:
            return json.dumps({"results": [], "count": 0, "error": f"embed: {exc}"})

        try:
            rows = self._store.search_turns(
                query_embedding=vec,
                agent_identity=agent_filter,
                session_id=session_filter,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"results": [], "count": 0, "error": f"db: {exc}"})

        results = []
        for r in rows:
            ts = r.get("ts")
            results.append(
                {
                    "id": r.get("id"),
                    "agent_identity": r.get("agent_identity"),
                    "session_id": r.get("session_id"),
                    "role": r.get("role"),
                    "ts": ts.isoformat() if ts else None,
                    "score": round(float(r.get("score") or 0.0), 4),
                    "content": (r.get("content") or "")[:2000],
                }
            )
        return json.dumps({"results": results, "count": len(results)})

    def _handle_recall_delegation(self, args: Dict[str, Any]) -> str:
        """Tool handler for recall_delegation over the delegations table."""
        if not self._healthy or not self._store:
            return json.dumps({"results": [], "count": 0, "error": "hexus unavailable"})

        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required arg: query")
        try:
            limit = max(1, min(int(args.get("limit", 5)), 20))
        except (TypeError, ValueError):
            limit = 5

        scope = (args.get("scope") or "current").strip()
        agent_filter: Optional[str] = None
        if scope == "current":
            agent_filter = self._agent_identity
        elif scope == "all":
            pass  # no filters
        else:
            agent_filter = scope  # treat as a specific theme name

        try:
            vec = embed(
                query,
                base_url=self._config["embed_url"],
                model=self._config["embed_model"],
            )
        except EmbeddingError as exc:
            return json.dumps({"results": [], "count": 0, "error": f"embed: {exc}"})

        try:
            rows = self._store.search_delegations(
                query_embedding=vec,
                agent_identity=agent_filter,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"results": [], "count": 0, "error": f"db: {exc}"})

        results = []
        for r in rows:
            ts = r.get("ts")
            results.append(
                {
                    "id": r.get("id"),
                    "parent_session_id": r.get("parent_session_id"),
                    "child_session_id": r.get("child_session_id"),
                    "agent_identity": r.get("agent_identity"),
                    "task": r.get("task"),
                    "result": r.get("result"),
                    "ts": ts.isoformat() if ts else None,
                    "score": round(float(r.get("score") or 0.0), 4),
                }
            )
        return json.dumps({"results": results, "count": len(results)})

    def _handle_entity_graph(self, args: Dict[str, Any]) -> str:
        """Tool handler for entity_graph."""
        if not self._healthy or not self._store:
            return json.dumps({"error": "hexus unavailable"})

        entity_type = (args.get("entity_type") or "").strip()
        if not entity_type:
            return tool_error("Missing required arg: entity_type")
        entity_value = (args.get("entity_value") or "").strip()
        if not entity_value:
            return tool_error("Missing required arg: entity_value")

        try:
            limit = max(1, min(int(args.get("limit", 5)), 20))
        except (TypeError, ValueError):
            limit = 5

        scope = (args.get("scope") or "current").strip()
        agent_filter: Optional[str] = None
        if scope == "current":
            agent_filter = self._agent_identity
        elif scope == "all":
            pass
        else:
            agent_filter = scope

        try:
            res = self._store.entity_graph(
                entity_type=entity_type,
                entity_value=entity_value,
                agent_identity=agent_filter,
                limit=limit,
            )
            return json.dumps(res)
        except Exception as exc:
            return json.dumps({"error": f"db: {exc}"})

    def _handle_graph_walk(self, args: Dict[str, Any]) -> str:
        """Tool handler for graph_walk."""
        if not self._healthy or not self._store:
            return json.dumps({"error": "hexus unavailable"})

        entity_type = (args.get("entity_type") or "").strip()
        if not entity_type:
            return tool_error("Missing required arg: entity_type")
        entity_value = (args.get("entity_value") or "").strip()
        if not entity_value:
            return tool_error("Missing required arg: entity_value")

        try:
            max_depth = max(1, min(int(args.get("max_depth", 2)), 5))
        except (TypeError, ValueError):
            max_depth = 2

        try:
            limit = max(1, min(int(args.get("limit", 5)), 20))
        except (TypeError, ValueError):
            limit = 5

        scope = (args.get("scope") or "current").strip()
        agent_filter: Optional[str] = None
        if scope == "current":
            agent_filter = self._agent_identity
        elif scope == "all":
            pass
        else:
            agent_filter = scope

        try:
            res = self._store.graph_walk(
                entity_type=entity_type,
                entity_value=entity_value,
                agent_identity=agent_filter,
                max_depth=max_depth,
                limit=limit,
            )
            return json.dumps({"results": res})
        except Exception as exc:
            return json.dumps({"error": f"db: {exc}"})

    def _handle_common_topics(self, args: Dict[str, Any]) -> str:
        """Tool handler for common_topics."""
        if not self._healthy or not self._store:
            return json.dumps({"error": "hexus unavailable"})

        try:
            min_strength = max(1, int(args.get("min_strength", 2)))
        except (TypeError, ValueError):
            min_strength = 2

        try:
            limit = max(1, min(int(args.get("limit", 10)), 20))
        except (TypeError, ValueError):
            limit = 10

        scope = (args.get("scope") or "current").strip()
        agent_filter: Optional[str] = None
        if scope == "current":
            agent_filter = self._agent_identity
        elif scope == "all":
            pass
        else:
            agent_filter = scope

        try:
            res = self._store.common_topics(
                agent_identity=agent_filter,
                min_strength=min_strength,
                limit=limit,
            )
            return json.dumps({"results": res})
        except Exception as exc:
            return json.dumps({"error": f"db: {exc}"})

    def _handle_confirm_memory(self, args: Dict[str, Any]) -> str:
        """Tool handler for confirm_memory."""
        if not self._healthy or not self._store:
            return json.dumps({"error": "hexus unavailable"})

        entry_id = args.get("id")
        if entry_id is None:
            return tool_error("Missing required arg: id")
        try:
            entry_id = int(entry_id)
        except (TypeError, ValueError):
            return tool_error("id must be an integer")

        try:
            success = self._store.confirm_entry(entry_id, self._agent_identity)
            return json.dumps({"id": entry_id, "success": success})
        except Exception as exc:
            return json.dumps({"error": f"db: {exc}"})

    def _handle_reject_memory(self, args: Dict[str, Any]) -> str:
        """Tool handler for reject_memory."""
        if not self._healthy or not self._store:
            return json.dumps({"error": "hexus unavailable"})

        entry_id = args.get("id")
        if entry_id is None:
            return tool_error("Missing required arg: id")
        try:
            entry_id = int(entry_id)
        except (TypeError, ValueError):
            return tool_error("id must be an integer")

        try:
            success = self._store.reject_entry(entry_id, self._agent_identity)
            return json.dumps({"id": entry_id, "success": success})
        except Exception as exc:
            return json.dumps({"error": f"db: {exc}"})

    def _handle_summarize_session(self, args: Dict[str, Any]) -> str:
        """Tool handler for summarize_session."""
        if not self._healthy or not self._store:
            return json.dumps({"error": "hexus unavailable"})

        session_id = (args.get("session_id") or "").strip()
        if not session_id:
            return tool_error("Missing required arg: session_id")

        try:
            limit = max(1, min(int(args.get("limit", 5)), 20))
        except (TypeError, ValueError):
            limit = 5

        try:
            res = self._store.summarize_session(
                session_id=session_id,
                limit=limit,
                agent_identity=self._agent_identity,
            )
            return json.dumps(res)
        except Exception as exc:
            return json.dumps({"error": f"db: {exc}"})

    def _handle_headroom_retrieve(self, args: Dict[str, Any]) -> str:
        """Tool handler for headroom_retrieve."""
        if not self._healthy or not self._store:
            return json.dumps({"error": "hexus unavailable"})

        entry_id = args.get("id")
        if entry_id is None:
            return tool_error("Missing required arg: id")
        try:
            entry_id = int(entry_id)
        except (TypeError, ValueError):
            return tool_error("id must be an integer")

        try:
            content = self._store.fetch_full(entry_id, self._agent_identity)
            if content is None:
                return json.dumps({"id": entry_id, "found": False, "content": None})
            return json.dumps({"id": entry_id, "found": True, "content": content})
        except Exception as exc:
            return json.dumps({"error": f"db: {exc}"})

    def _handle_memory_stats(self, args: Dict[str, Any]) -> str:
        """Tool handler for memory_stats."""
        if not self._healthy or not self._store:
            return json.dumps({"error": "hexus unavailable"})
        try:
            from mcp_server.tools import memory_stats

            res = memory_stats(self._store, args)
            return json.dumps(res)
        except Exception as exc:
            return json.dumps({"error": f"stats check failed: {exc}"})

    # -- Setup hooks ---------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "dsn",
                "description": "Postgres DSN (psycopg connection string)",
                "default": DEFAULTS["dsn"],
                "required": True,
            },
            {
                "key": "embed_url",
                "description": "Embedding endpoint base URL (OpenAI-compatible or Ollama native)",
                "default": DEFAULTS["embed_url"],
                "required": True,
            },
            {
                "key": "embed_model",
                "description": "Embedding model name (must return 768-dim vectors)",
                "default": DEFAULTS["embed_model"],
            },
            {
                "key": "prefetch_limit",
                "description": "Max ambient recall results injected per turn",
                "default": str(DEFAULTS["prefetch_limit"]),
            },
            {
                "key": "min_similarity",
                "description": "Cosine similarity cutoff for ambient prefetch (0.0–1.0)",
                "default": str(DEFAULTS["min_similarity"]),
            },
            {
                "key": "embed_on_write",
                "description": "Compute embedding on each write; turn off for text-only mode",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "scope_default",
                "description": "Default scope for recall_memory when caller omits it",
                "default": DEFAULTS["scope_default"],
                "choices": ["current", "all"],
            },
            {
                "key": "write_queue_maxsize",
                "description": "Bounded async-writer queue size; full = oldest writes drop with a warning",
                "default": str(DEFAULTS["write_queue_maxsize"]),
            },
            {
                "key": "bulk_sync_on_init",
                "description": "Import MEMORY.md / USER.md content from disk on agent init (v0.1.1)",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "sync_turns",
                "description": "Capture every substantive (user, assistant) turn pair into the conversations table",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "turn_min_chars",
                "description": "Turns shorter than this (after strip) are treated as boilerplate and skipped",
                "default": str(DEFAULTS["turn_min_chars"]),
            },
            {
                "key": "webhook_url",
                "description": "Custom HTTP URL to dispatch webhook payloads upon lifecycle events",
                "default": "",
                "required": False,
            },
            {
                "key": "webhook_secret",
                "description": "Secret token for signing webhook payloads with HMAC-SHA256",
                "default": "",
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from pathlib import Path

        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml

            existing: Dict[str, Any] = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as fh:
                    existing = yaml.safe_load(fh) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["hexus"] = values
            with open(config_path, "w", encoding="utf-8") as fh:
                yaml.dump(existing, fh, default_flow_style=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hexus save_config failed: %s", exc)

    # -- Helpers -------------------------------------------------------------

    def _maybe_embed(self, content: str) -> Optional[List[float]]:
        if not self._config.get("embed_on_write", True):
            return None
        try:
            return embed(
                content,
                base_url=self._config["embed_url"],
                model=self._config["embed_model"],
            )
        except EmbeddingError as exc:
            if not self._embed_warned:
                logger.warning("hexus embed failed (degrading to text-only): %s", exc)
                self._embed_warned = True
            return None


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register the hexus memory provider with the plugin system."""
    provider = HexusMemoryProvider(config=_load_plugin_config())
    ctx.register_memory_provider(provider)
