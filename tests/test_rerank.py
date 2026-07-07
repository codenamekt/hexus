"""tests/test_rerank.py — long-document reranking (issue #7).

Query-side companion to the write-side embedder handling. The cross-encoder
scores a (query, doc) pair jointly and truncates at its context window, so a
long `compressed`/`content` doc loses its tail before scoring. These tests
exercise the truncate/warn/maxp handling in hexus.store.rerank_scores with a
fake cross-encoder — no model load, no DB.
"""

from __future__ import annotations

import logging

import pytest

# rerank_scores and friends are pure module-level helpers (no DB), but the
# module imports psycopg at import time — same requirement as the other
# store-backed tests in this suite.
psycopg = pytest.importorskip("psycopg")

from hexus.store import (  # noqa: E402
    RERANK_MAX_PASSAGES,
    rerank_scores,
    get_rerank_stats,
    reset_rerank_stats,
    _resolve_rerank_mode,
    _cross_encoder_max_len,
)


QUERY = "hi there"  # 2 tokens → budget = 32 - 2 - 3 = 27


class _FakeTok:
    model_max_length = 32

    def encode(self, text, add_special_tokens=False, verbose=False):
        return list(range(len(text.split())))

    def num_special_tokens_to_add(self, pair=True):
        return 3

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)


class _FakeCE:
    """Fake CrossEncoder. Scores each pair by its doc word count so tests can
    verify order and the max-over-passages reduction deterministically."""

    def __init__(self, max_length=32):
        self.max_length = max_length
        self.tokenizer = _FakeTok()
        self.calls = []

    def predict(self, pairs):
        self.calls.append([list(p) for p in pairs])
        return [float(len(p[1].split())) for p in pairs]


def _words(n):
    return " ".join(f"w{i}" for i in range(n))


@pytest.fixture(autouse=True)
def _clean_stats():
    reset_rerank_stats()
    yield
    reset_rerank_stats()


def test_mode_resolution():
    assert _resolve_rerank_mode(None) == "warn"
    assert _resolve_rerank_mode("bogus") == "warn"
    assert _resolve_rerank_mode("MaxP") == "maxp"


def test_mode_env_override(monkeypatch):
    monkeypatch.setenv("HEXUS_RERANK_LONG_DOC_MODE", "truncate")
    assert _resolve_rerank_mode(None) == "truncate"


def test_max_len_from_model():
    assert _cross_encoder_max_len(_FakeCE(64)) == 64


def test_empty_docs_returns_empty():
    assert rerank_scores(_FakeCE(), QUERY, []) == []


def test_short_docs_are_passthrough():
    ce = _FakeCE()
    scores = rerank_scores(ce, QUERY, [_words(5), _words(3)], mode="warn")
    assert scores == [5.0, 3.0]
    assert len(ce.calls[0]) == 2  # one pair per doc
    assert get_rerank_stats().docs_over_limit == 0


def test_warn_mode_truncates_and_counts(caplog):
    ce = _FakeCE()
    with caplog.at_level(logging.WARNING, logger="hexus.store"):
        scores = rerank_scores(ce, QUERY, [_words(60), _words(5)], mode="warn")
    assert len(scores) == 2
    assert scores[0] == 60.0 and scores[1] == 5.0  # order preserved
    assert len(ce.calls[0]) == 2  # still one pair per doc (truncated)
    s = get_rerank_stats()
    assert s.docs_over_limit == 1
    assert s.docs_truncated == 1
    assert s.docs_split == 0
    assert s.tokens_dropped == 60 - 27
    assert s.max_tokens_seen == 60
    assert any("exceeds cross-encoder" in r.message for r in caplog.records)


def test_truncate_mode_is_silent(caplog):
    ce = _FakeCE()
    with caplog.at_level(logging.WARNING, logger="hexus.store"):
        rerank_scores(ce, QUERY, [_words(60)], mode="truncate")
    assert get_rerank_stats().docs_truncated == 1
    assert not any("exceeds cross-encoder" in r.message for r in caplog.records)


def test_maxp_splits_and_max_reduces():
    ce = _FakeCE()
    scores = rerank_scores(ce, QUERY, [_words(60), _words(5)], mode="maxp")
    assert len(scores) == 2
    assert len(ce.calls[0]) > 2  # long doc expanded into passages
    assert scores[1] == 5.0  # short doc untouched, order preserved
    assert isinstance(scores[0], float) and scores[0] > 0
    s = get_rerank_stats()
    assert s.docs_split == 1
    assert s.passages_scored > 1
    assert s.docs_truncated == 0


def test_maxp_caps_passages_and_records_tail():
    ce = _FakeCE()
    rerank_scores(ce, QUERY, [_words(300)], mode="maxp")
    s = get_rerank_stats()
    assert s.passages_scored == RERANK_MAX_PASSAGES  # bounded
    assert s.docs_capped == 1
    assert s.tokens_dropped > 0  # tail accounted for, not silently dropped


def test_no_tokenizer_falls_back():
    class _NoTokCE:
        max_length = 32
        tokenizer = None

        def __init__(self):
            self.calls = []

        def predict(self, pairs):
            self.calls.append(list(pairs))
            return [1.0 for _ in pairs]

    ce = _NoTokCE()
    scores = rerank_scores(ce, QUERY, [_words(999), _words(5)], mode="maxp")
    assert len(scores) == 2
    assert len(ce.calls[0]) == 2  # no split without a tokenizer
    assert get_rerank_stats().docs_over_limit == 0


def test_warn_is_throttled(caplog):
    ce = _FakeCE()
    with caplog.at_level(logging.WARNING, logger="hexus.store"):
        rerank_scores(ce, QUERY, [_words(60)] * 101, mode="warn")
    warns = [r for r in caplog.records if "exceeds cross-encoder" in r.message]
    assert len(warns) == 2  # 1st and 100th
    assert get_rerank_stats().docs_over_limit == 101
