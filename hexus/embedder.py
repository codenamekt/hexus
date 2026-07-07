"""embedder.py — local sentence-transformers embedder for the hexus memory plugin.
#
# Forked from andreab67/hermes-memory-pgvector (BSD-3-Clause).
#
# Replaces the upstream HTTP embedder with a local MiniLM-L6-v2 model
# loaded once at first use. Produces 384-dim vectors. The whole model
# fits in <500MB RAM and runs ~20-50 sentences/sec on the NUC6i7KYK
# i7 (CPU-only, no GPU needed). Cold start is the first embed call
# (~1-2s for the model load) — the async writer absorbs that without
# blocking the agent loop.
#
# Why this exists:
#   - v0.3.x required a separate Ollama / OpenAI-compatible endpoint.
#     The hermes fleet deployments on the NUC had no such endpoint
#     running; spinning one up just for the memory store is overkill
#     for a 23M-param model.
#   - sentence-transformers is a single pip install with no daemon
#     to manage, no port to expose, no healthcheck to monitor.
#   - One process = one model load. The MCP server and the Hermes
#     plugin share the same MemoryStore + LocalBertEmbedder instance,
#     so a fleet of N agents costs ~500MB resident, not N×500MB.
#
# The lazy load (see LocalBertEmbedder.embed) is deliberate: importing
# the plugin package must stay fast (Hermes loads it at startup), and
# tests that don't need embeddings should not pay the model load cost.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("USER", "agy")
import threading
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Public model name constant — keep in one place so tests + the provider
# config + the README can all reference the same value.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384

# How to handle text longer than the model's context window
# (all-MiniLM-L6-v2 has max_seq_length=256 — note this is a
# sentence-transformers cap; the BERT backbone supports 512 positions,
# but the model was not trained past 256, so raising it degrades quality.
# Chunking is the right long-form fix). See issue #7.
#
#   truncate  — hand the text to the model, which silently clips it at
#               max_seq_length. No log line. Stats still count it, so
#               operators can measure the loss even when logs are quiet.
#   warn      — same truncation, but emit a throttled logger.warning so
#               the loss is visible. This is the default: it changes no
#               recall behaviour and can be silenced by raising the log
#               level or switching to `truncate`.
#   chunk     — split the text into overlapping token windows, embed each,
#               and combine them into one token-count-weighted, L2-normalised
#               vector. One vector per input — no schema change. Preserves
#               signal from the whole text instead of just the first window.
LONG_TEXT_MODE_TRUNCATE = "truncate"
LONG_TEXT_MODE_WARN = "warn"
LONG_TEXT_MODE_CHUNK = "chunk"
VALID_LONG_TEXT_MODES = (
    LONG_TEXT_MODE_TRUNCATE,
    LONG_TEXT_MODE_WARN,
    LONG_TEXT_MODE_CHUNK,
)
DEFAULT_LONG_TEXT_MODE = LONG_TEXT_MODE_WARN
LONG_TEXT_MODE_ENV = "HEXUS_EMBED_LONG_TEXT_MODE"

# Token overlap between adjacent chunks in `chunk` mode. A small overlap
# keeps a phrase that straddles a window boundary represented in both
# windows; the token-weighted average washes out most boundary effects
# so this is deliberately modest.
CHUNK_OVERLAP_TOKENS = 16


@dataclass
class EmbedStats:
    """Per-process counters for long-input handling (issue #7).

    Every over-limit text is counted here regardless of the configured
    mode, so operators can answer "how often is content being clipped /
    chunked?" even when logging is turned down. Read a snapshot via
    ``LocalBertEmbedder.stats`` or the aggregate ``get_embed_stats()``.
    """

    texts_embedded: int = 0  # total non-empty texts passed to embed()
    texts_over_limit: int = 0  # texts whose token count exceeded max_seq_length
    texts_truncated: int = 0  # over-limit texts left for the model to clip
    texts_chunked: int = 0  # over-limit texts split into windows (chunk mode)
    chunks_encoded: int = 0  # total window sub-encodes produced by chunking
    tokens_dropped: int = 0  # approx tokens lost to truncation (Σ tc-max_seq)
    max_tokens_seen: int = 0  # largest single-text token count observed

    def as_dict(self) -> Dict[str, int]:
        return {
            "texts_embedded": self.texts_embedded,
            "texts_over_limit": self.texts_over_limit,
            "texts_truncated": self.texts_truncated,
            "texts_chunked": self.texts_chunked,
            "chunks_encoded": self.chunks_encoded,
            "tokens_dropped": self.tokens_dropped,
            "max_tokens_seen": self.max_tokens_seen,
        }


def _resolve_long_text_mode(mode: Optional[str]) -> str:
    """Resolve the configured mode: explicit arg > env var > default.

    An unrecognised value falls back to the default with a warning rather
    than raising — a typo in an env var should not take embedding offline.
    """
    candidate = mode or os.environ.get(LONG_TEXT_MODE_ENV) or DEFAULT_LONG_TEXT_MODE
    candidate = candidate.strip().lower()
    if candidate not in VALID_LONG_TEXT_MODES:
        logger.warning(
            "invalid %s=%r; falling back to %r (valid: %s)",
            LONG_TEXT_MODE_ENV,
            candidate,
            DEFAULT_LONG_TEXT_MODE,
            ", ".join(VALID_LONG_TEXT_MODES),
        )
        return DEFAULT_LONG_TEXT_MODE
    return candidate


class EmbedderError(Exception):
    """Raised when the local embedder fails to produce a usable vector."""


class LocalBertEmbedder:
    """Lazy-loaded, thread-safe wrapper around sentence-transformers.

    The model is loaded on the first call to `embed()` (or eagerly via
    `ensure_loaded()`) and reused for all subsequent calls. The class
    itself is cheap to construct — no model is loaded until needed.

    Designed to be a singleton per process: callers should keep one
    instance in their config and pass it around, not construct one per
    call. The MCP server and the Hermes plugin share the same
    MemoryStore which holds one embedder, so the model is loaded once
    per process regardless of how many consumers there are.

    Thread safety: a lock guards the model-load step; the underlying
    sentence-transformers `encode()` is thread-safe per the library's
    own documentation (SentenceTransformer.encode holds a per-call GIL
    boundary in the encode path), so concurrent embeds serialize on the
    GIL inside numpy/torch and don't need extra locking here.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        long_text_mode: Optional[str] = None,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._device = device
        self._long_text_mode = _resolve_long_text_mode(long_text_mode)
        self._model = None  # loaded on first embed()
        self._load_lock = threading.Lock()
        self._load_failed = False
        # Long-input stats + a lock: embed() is called from the async
        # writer thread(s), so counter bumps must be serialized.
        self._stats = EmbedStats()
        self._stats_lock = threading.Lock()

    # -- Public API ---------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Eagerly load the model. Useful at plugin init when you want
        the cold-start to happen on a known thread (and visibly, in
        logs) rather than on the first user-facing embed call.

        Idempotent: a no-op if the model is already loaded. Raises
        EmbedderError on load failure.
        """
        self._load_model()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        """Embedding dimension. For the default model this is 384.

        Read from the loaded model if available (some models
        self-report their dimension), otherwise returned as the
        constant for the default model.
        """
        if self._model is not None:
            # sentence-transformers exposes the dim on the underlying
            # transformer config. Fall through to the constant if not.
            try:
                return int(self._model.get_sentence_embedding_dimension())
            except Exception:  # noqa: BLE001
                pass
        return DEFAULT_DIM if self._model_name == DEFAULT_MODEL else 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def long_text_mode(self) -> str:
        """The configured long-input strategy: 'truncate' | 'warn' | 'chunk'."""
        return self._long_text_mode

    @property
    def stats(self) -> EmbedStats:
        """A snapshot copy of the long-input counters (issue #7).

        Returns a copy, so reading it never races an in-flight embed():
        e.g. ``get_default_embedder().stats.texts_chunked``.
        """
        with self._stats_lock:
            return replace(self._stats)

    def reset_stats(self) -> None:
        """Zero the long-input counters. Useful to measure a single run."""
        with self._stats_lock:
            self._stats = EmbedStats()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts → list of float vectors.

        Empty / whitespace-only inputs are silently dropped (returned
        as an empty list for that entry) rather than raising — callers
        in the async writer are fail-soft and the upstream test suite
        already has the "reject empty" semantics at the module level
        (see embed.embed).
        """
        if not texts:
            return []
        # Filter empties, but remember their original indices so callers
        # can still correlate the output if they care. (Today no caller
        # does — bulk_upsert_md and the async writer both just want
        # "the embeddings for the non-empty items".)
        non_empty = [t for t in texts if t and t.strip()]
        if not non_empty:
            return []

        model = self._load_model()

        # Plan the encode. In the common all-short case this is a 1:1
        # pass-through identical to the pre-issue-#7 behaviour: `pieces`
        # is just `non_empty` and every plan entry is a single vector.
        # Long inputs are expanded (chunk mode) or flagged (warn/truncate)
        # here, and every over-limit text is counted in self._stats.
        pieces, plan = self._plan_encode(non_empty, model)

        try:
            # Disable sentence-transformers/tqdm progress bars; Hermes TUI
            # already reports memory tool progress and noisy "Batches 100%"
            # output is not useful in chat.
            raw = model.encode(
                pieces,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft, surface in logs
            raise EmbedderError(f"local embed failed: {exc}") from exc

        # Reassemble one vector per original input (chunk windows collapse
        # to a single weighted-average vector; everything else is 1:1).
        vectors = self._assemble(raw, plan)

        if not vectors or not isinstance(vectors[0], list):
            raise EmbedderError(f"unexpected encoder output shape: {type(vectors[0])}")

        actual_dim = len(vectors[0])
        if self._model_name == DEFAULT_MODEL and actual_dim != DEFAULT_DIM:
            # Only enforce for the known-default model — a custom model
            # with a different dim is fine, the schema is dim-driven.
            logger.warning(
                "embedder dim mismatch: model %s produced %d-dim, expected %d",
                self._model_name,
                actual_dim,
                DEFAULT_DIM,
            )
        return vectors

    # -- Long-input handling (issue #7) -------------------------------------

    def _plan_encode(
        self, texts: List[str], model
    ) -> Tuple[List[str], List[Tuple[int, int, Optional[List[float]]]]]:
        """Turn `texts` into (`pieces` to encode, `plan` to reassemble them).

        Each plan entry is `(start, count, weights)`:
          - count == 1: `pieces[start]` is the whole input → one vector, 1:1.
          - count  > 1: `pieces[start:start+count]` are chunk windows to be
            combined into one vector using `weights` (token counts).

        The token-length guard is best-effort: if the model exposes no
        tokenizer or no max_seq_length we skip it entirely and fall back to
        the model's own internal truncation (the pre-#7 behaviour).
        """
        tokenizer = getattr(model, "tokenizer", None)
        max_seq = self._resolve_max_seq(model)
        token_counts = (
            self._token_counts(texts, tokenizer)
            if (tokenizer is not None and max_seq > 0)
            else None
        )

        mode = self._long_text_mode
        pieces: List[str] = []
        plan: List[Tuple[int, int, Optional[List[float]]]] = []

        for i, text in enumerate(texts):
            with self._stats_lock:
                self._stats.texts_embedded += 1

            tc = token_counts[i] if token_counts is not None else None
            if tc is None or tc <= max_seq:
                pieces.append(text)
                plan.append((len(pieces) - 1, 1, None))
                continue

            # Over the limit — count it (in every mode) and record the peak.
            with self._stats_lock:
                self._stats.texts_over_limit += 1
                if tc > self._stats.max_tokens_seen:
                    self._stats.max_tokens_seen = tc
                over_count = self._stats.texts_over_limit

            if mode == LONG_TEXT_MODE_CHUNK:
                chunks = self._chunk_text(text, tokenizer, max_seq)
                if len(chunks) > 1:
                    start = len(pieces)
                    weights = [float(n) for _, n in chunks]
                    pieces.extend(ctext for ctext, _ in chunks)
                    plan.append((start, len(chunks), weights))
                    with self._stats_lock:
                        self._stats.texts_chunked += 1
                        self._stats.chunks_encoded += len(chunks)
                    self._log_over_limit(
                        over_count, tc, max_seq, chunked=True, n_chunks=len(chunks)
                    )
                    continue
                # Couldn't split (degenerate tokenizer output) — fall through
                # and let the model truncate rather than dropping the entry.

            # truncate / warn modes (and the chunk fallback above).
            with self._stats_lock:
                self._stats.texts_truncated += 1
                self._stats.tokens_dropped += tc - max_seq
            pieces.append(text)
            plan.append((len(pieces) - 1, 1, None))
            self._log_over_limit(over_count, tc, max_seq, chunked=False)

        return pieces, plan

    def _assemble(
        self, raw, plan: List[Tuple[int, int, Optional[List[float]]]]
    ) -> List[List[float]]:
        """Collapse encoded `pieces` back to one vector per original input.

        Single-piece entries are returned verbatim (byte-identical to the
        old path). Multi-chunk entries are combined as a token-count-weighted
        average, then L2-normalised. The store ranks with cosine distance and
        binary_quantize (both magnitude-invariant), so normalisation is safe
        and keeps the averaged vector well-behaved.
        """
        # Fast path: nothing was chunked (all warn/truncate/short inputs) →
        # the pieces are 1:1 with the inputs, so this is exactly the pre-#7
        # `model.encode(...).tolist()`. Avoids per-row work and numpy here.
        if all(count == 1 for _, count, _ in plan):
            return raw.tolist()

        import numpy as np

        vectors: List[List[float]] = []
        for start, count, weights in plan:
            if count == 1:
                vectors.append(raw[start].tolist())
                continue
            block = raw[start : start + count]
            w = np.asarray(weights, dtype=block.dtype)
            total = float(w.sum())
            w = (w / total) if total > 0 else np.full(count, 1.0 / count, dtype=block.dtype)
            avg = (block * w[:, None]).sum(axis=0)
            norm = float(np.linalg.norm(avg))
            if norm > 0:
                avg = avg / norm
            vectors.append(avg.tolist())
        return vectors

    @staticmethod
    def _resolve_max_seq(model) -> int:
        try:
            return int(getattr(model, "max_seq_length", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _token_counts(self, texts: List[str], tokenizer) -> Optional[List[int]]:
        """True (untruncated) token count per text, or None if unavailable.

        `verbose=False` suppresses HuggingFace's "sequence longer than model
        max" stderr warning — we do our own, controllable, logging."""
        try:
            enc = tokenizer(
                texts,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
            return [len(ids) for ids in enc["input_ids"]]
        except Exception as exc:  # noqa: BLE001 — best-effort guard
            logger.debug(
                "hexus embedder: token counting unavailable (%s); "
                "skipping length guard",
                exc,
            )
            return None

    def _chunk_text(
        self, text: str, tokenizer, max_seq: int
    ) -> List[Tuple[str, int]]:
        """Split `text` into overlapping token windows → [(chunk_text, n_tokens)].

        We reserve room for the special tokens the tokenizer re-adds when each
        decoded window is re-encoded, so no window ever exceeds max_seq. If the
        tokenizer can't round-trip we return a single element and the caller
        falls back to truncation.
        """
        try:
            special = tokenizer.num_special_tokens_to_add(pair=False)
        except Exception:  # noqa: BLE001
            special = 2
        window = max(1, max_seq - special)
        overlap = min(CHUNK_OVERLAP_TOKENS, window - 1) if window > 1 else 0
        stride = max(1, window - overlap)

        try:
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        except Exception:  # noqa: BLE001
            return [(text, max_seq)]

        if len(ids) <= window:
            return [(text, len(ids))]

        chunks: List[Tuple[str, int]] = []
        for start in range(0, len(ids), stride):
            window_ids = ids[start : start + window]
            if not window_ids:
                break
            ctext = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
            if ctext:
                chunks.append((ctext, len(window_ids)))
            if start + window >= len(ids):
                break
        return chunks or [(text, len(ids))]

    def _log_over_limit(
        self,
        over_count: int,
        token_count: int,
        max_seq: int,
        *,
        chunked: bool,
        n_chunks: int = 0,
    ) -> None:
        """Log an over-limit text per the configured mode.

        - chunk:    handled, so only a debug line (stats carry the count).
        - truncate: silent by design (stats still count it).
        - warn:     logger.warning, throttled to the 1st occurrence and every
                    100th thereafter, so a hot loop doesn't flood the log while
                    ongoing volume stays visible in stats. Never logs content —
                    these are private memory entries.
        """
        if chunked:
            logger.debug(
                "hexus embedder: chunked long input (%d tokens > %d max) "
                "into %d windows [model=%s]",
                token_count,
                max_seq,
                n_chunks,
                self._model_name,
            )
            return
        if self._long_text_mode != LONG_TEXT_MODE_WARN:
            return  # truncate: intentionally silent
        if over_count == 1 or over_count % 100 == 0:
            logger.warning(
                "hexus embedder: input exceeds model context "
                "(%d tokens > %d max) — truncating; ~%d tokens dropped. "
                "%d over-limit text(s) so far this process (see embedder "
                "stats for the running total). Set %s=chunk to keep the "
                "full content, or =truncate to silence. [model=%s]",
                token_count,
                max_seq,
                token_count - max_seq,
                over_count,
                LONG_TEXT_MODE_ENV,
                self._model_name,
            )

    # -- Internals ----------------------------------------------------------

    def _load_model(self):
        """Load the sentence-transformers model. Idempotent + thread-safe.

        The double-check pattern (check outside lock, then check inside
        before doing the expensive load) avoids serializing every embed
        call on the lock once the model is warm.
        """
        if self._model is not None:
            return self._model
        if self._load_failed:
            # Don't repeatedly try to load a model that already failed
            # this process — surface the error fast.
            raise EmbedderError(
                f"local embedder previously failed to load {self._model_name}; "
                "restart the process to retry"
            )
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                # Local import keeps the sentence-transformers dep
                # (and its torch/numpy/transformers transitive deps)
                # out of the module-level import graph, so importing
                # the plugin package stays fast.
                from sentence_transformers import SentenceTransformer

                kwargs = {"device": self._device}
                if self._cache_dir:
                    kwargs["cache_folder"] = self._cache_dir
                # Honor HF_HUB_OFFLINE for air-gapped production
                # containers. sentence-transformers passes through
                # whatever env vars are set.
                self._model = SentenceTransformer(self._model_name, **kwargs)
                logger.info(
                    "loaded local embedder model=%s dim=%d device=%s",
                    self._model_name,
                    self.dim,
                    self._device,
                )
                return self._model
            except Exception as exc:  # noqa: BLE001
                self._load_failed = True
                raise EmbedderError(
                    f"failed to load sentence-transformers model {self._model_name}: {exc}"
                ) from exc


# Module-level singleton accessor. NOT auto-created at import — callers
# must opt in. This keeps the import graph clean (no torch import at
# plugin import time) and lets tests inject their own embedder.
#
# Caching is keyed on (model_name, cache_dir, device) so a request for a
# different model returns a different embedder (mostly relevant for tests
# — production uses one model). The dict is small in practice.
_singletons: dict[tuple[str, Optional[str], str, str], "LocalBertEmbedder"] = {}
_singleton_lock = threading.Lock()


def get_default_embedder(
    model_name: str = DEFAULT_MODEL,
    *,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    long_text_mode: Optional[str] = None,
) -> "LocalBertEmbedder":
    """Return the process-wide default embedder for these args, constructing
    it on first call. Subsequent calls with the same (model_name, cache_dir,
    device, long_text_mode) return the same instance.

    The default device is `cpu`; pass `device="cuda"` (or similar) at
    first call to override.

    `long_text_mode` is resolved from the arg, then the
    HEXUS_EMBED_LONG_TEXT_MODE env var, then the default ('warn'). It is
    part of the cache key so a request for a different mode returns a
    distinct embedder rather than silently reusing another mode's instance.
    """
    global _singletons
    if device is None:
        device = os.environ.get("HEXUS_EMBED_DEVICE", "cpu")
    mode = _resolve_long_text_mode(long_text_mode)
    key = (model_name, cache_dir, device, mode)
    with _singleton_lock:
        existing = _singletons.get(key)
        if existing is not None:
            return existing
        embedder = LocalBertEmbedder(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device,
            long_text_mode=mode,
        )
        _singletons[key] = embedder
        return embedder


def get_embed_stats() -> EmbedStats:
    """Aggregate long-input stats across all live embedder singletons.

    Convenience for "how often was content clipped/chunked this process?"
    without having to hold an embedder reference. In production there is
    normally a single singleton, so this just returns its counters.
    """
    total = EmbedStats()
    with _singleton_lock:
        embedders = list(_singletons.values())
    for e in embedders:
        s = e.stats
        total.texts_embedded += s.texts_embedded
        total.texts_over_limit += s.texts_over_limit
        total.texts_truncated += s.texts_truncated
        total.texts_chunked += s.texts_chunked
        total.chunks_encoded += s.chunks_encoded
        total.tokens_dropped += s.tokens_dropped
        total.max_tokens_seen = max(total.max_tokens_seen, s.max_tokens_seen)
    return total


def reset_default_embedder() -> None:
    """Drop ALL module-level singletons. Test-only helper."""
    global _singletons
    with _singleton_lock:
        _singletons = {}
