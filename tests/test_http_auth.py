"""tests/test_http_auth.py — bearer-token auth wrapper for the MCP HTTP transport.

Covers `mcp_server.server._wrap_with_bearer_auth`: HTTP requests must present
`Authorization: Bearer <token>`, while lifespan/websocket scopes pass through
untouched (so the streamable-http session manager still starts). Pure ASGI —
no live server, no DB connection (importing the module only needs psycopg present).
"""

from __future__ import annotations

import asyncio

import pytest

# The module imports hexus.store (psycopg) at import time, but constructs no
# DB connection just to reach the auth wrapper.
pytest.importorskip("psycopg")

from mcp_server.server import _wrap_with_bearer_auth  # noqa: E402


def _make_downstream():
    """A minimal ASGI app that records the scope types it was called with."""
    calls = []

    async def app(scope, receive, send):
        calls.append(scope["type"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    app.calls = calls
    return app


def _drive(app, scope):
    """Run an ASGI app once and return (status_or_None, sent_messages)."""
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    asyncio.run(app(scope, receive, send))
    status = next(
        (m["status"] for m in sent if m["type"] == "http.response.start"), None
    )
    return status, sent


def _http_scope(auth_header=None):
    headers = []
    if auth_header is not None:
        headers.append((b"authorization", auth_header.encode("latin-1")))
    return {"type": "http", "headers": headers, "method": "POST", "path": "/mcp"}


def test_missing_token_is_rejected():
    downstream = _make_downstream()
    app = _wrap_with_bearer_auth(downstream, "s3cret")
    status, _ = _drive(app, _http_scope(auth_header=None))
    assert status == 401
    assert downstream.calls == []  # downstream never reached


def test_wrong_token_is_rejected():
    downstream = _make_downstream()
    app = _wrap_with_bearer_auth(downstream, "s3cret")
    status, _ = _drive(app, _http_scope(auth_header="Bearer nope"))
    assert status == 401
    assert downstream.calls == []


def test_correct_token_passes_through():
    downstream = _make_downstream()
    app = _wrap_with_bearer_auth(downstream, "s3cret")
    status, _ = _drive(app, _http_scope(auth_header="Bearer s3cret"))
    assert status == 200
    assert downstream.calls == ["http"]


def test_401_sets_www_authenticate_header():
    downstream = _make_downstream()
    app = _wrap_with_bearer_auth(downstream, "s3cret")
    _, sent = _drive(app, _http_scope(auth_header=None))
    start = next(m for m in sent if m["type"] == "http.response.start")
    header_names = {k.lower() for k, _ in start["headers"]}
    assert b"www-authenticate" in header_names


def test_lifespan_scope_passes_through_without_auth():
    """Non-http scopes must not be gated, or the streamable-http lifespan
    (which starts the session manager) would never run."""
    downstream = _make_downstream()
    app = _wrap_with_bearer_auth(downstream, "s3cret")

    sent = []

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(msg):
        sent.append(msg)

    asyncio.run(app({"type": "lifespan"}, receive, send))
    assert downstream.calls == ["lifespan"]  # forwarded, no 401


# -----------------------------------------------------------------------
# Server-derived caller identity (issue #19 item A)
# -----------------------------------------------------------------------

from mcp_server.server import _wrap_with_identity  # noqa: E402
from mcp_server import tools  # noqa: E402


def _identity_scope(session_key=None):
    headers = []
    if session_key is not None:
        headers.append((b"x-hermes-session-key", session_key.encode("latin-1")))
    return {"type": "http", "headers": headers, "method": "POST", "path": "/mcp"}


def test_identity_header_published_to_contextvar():
    """The X-Hermes-Session-Key header is visible via tools.current_caller for
    the duration of the request, and reset afterwards."""
    seen = {}

    async def app(scope, receive, send):
        seen["caller"] = tools.current_caller.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    wrapped = _wrap_with_identity(app)
    _drive(wrapped, _identity_scope(session_key="marketing"))
    assert seen["caller"] == "marketing"
    # No leak across requests.
    assert tools.current_caller.get() is None


def test_identity_absent_header_is_none():
    seen = {}

    async def app(scope, receive, send):
        seen["caller"] = tools.current_caller.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    wrapped = _wrap_with_identity(app)
    _drive(wrapped, _identity_scope(session_key=None))
    assert seen["caller"] is None


def test_identity_contextvar_beats_client_arg_for_writes():
    """When a transport identity is set, it is authoritative for writes even
    if the client passes a different agent_identity arg."""
    token = tools.current_caller.set("real-agent")
    try:
        assert tools._write_identity({"agent_identity": "spoofed"}) == "real-agent"
        assert tools._scope_identity({"agent_identity": "spoofed"}) == "real-agent"
    finally:
        tools.current_caller.reset(token)


def test_write_identity_falls_back_to_arg_then_env(monkeypatch):
    """Without a transport identity, writes fall back to the arg, then env."""
    assert tools.current_caller.get() is None
    assert tools._write_identity({"agent_identity": "sales"}) == "sales"
    monkeypatch.setenv("HEXUS_AGENT_IDENTITY", "fallback")
    assert tools._write_identity({}) == "fallback"


def test_scope_identity_none_when_nothing_known():
    """By-id ops stay unscoped for direct/stdio callers with no identity."""
    assert tools.current_caller.get() is None
    assert tools._scope_identity({}) is None


class _FakeStore:
    def __init__(self, isolation):
        self.isolation = isolation


def test_read_identity_shared_empty_means_all_agents():
    store = _FakeStore("shared")
    assert tools._read_identity(store, {"agent_identity": ""}) is None
    # Explicit arg is honored as a filter.
    assert tools._read_identity(store, {"agent_identity": "sales"}) == "sales"


def test_read_identity_strict_confines_to_caller(monkeypatch):
    store = _FakeStore("strict")
    monkeypatch.setenv("HEXUS_AGENT_IDENTITY", "me")
    # Empty → caller (env default here).
    assert tools._read_identity(store, {"agent_identity": ""}) == "me"
    # Even an explicit other-agent arg is overridden by the caller identity.
    token = tools.current_caller.set("real")
    try:
        assert tools._read_identity(store, {"agent_identity": "other"}) == "real"
    finally:
        tools.current_caller.reset(token)


# -----------------------------------------------------------------------
# SQL-safety helpers + isolation policy (issue #19), no DB required
# -----------------------------------------------------------------------

from hexus.store import _escape_like, _resolve_isolation  # noqa: E402


def test_escape_like_neutralizes_wildcards():
    assert _escape_like("%") == r"\%"
    assert _escape_like("_") == r"\_"
    assert _escape_like("a_b%c") == r"a\_b\%c"
    # Backslash escaped first so it can't double-escape a following wildcard.
    assert _escape_like("\\%") == r"\\\%"
    # Plain text is untouched.
    assert _escape_like("hello world") == "hello world"


def test_resolve_isolation_default_shared(monkeypatch):
    monkeypatch.delenv("HEXUS_MEMORY_ISOLATION", raising=False)
    assert _resolve_isolation() == "shared"


def test_resolve_isolation_strict_from_env(monkeypatch):
    monkeypatch.setenv("HEXUS_MEMORY_ISOLATION", "STRICT")
    assert _resolve_isolation() == "strict"
    monkeypatch.setenv("HEXUS_MEMORY_ISOLATION", "shared")
    assert _resolve_isolation() == "shared"
    # Explicit arg wins over env.
    assert _resolve_isolation("strict") == "strict"
    # Unknown value falls back to shared.
    monkeypatch.setenv("HEXUS_MEMORY_ISOLATION", "banana")
    assert _resolve_isolation() == "shared"
