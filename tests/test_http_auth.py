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
