# syntax=docker/dockerfile:1.6
# memory-pgvector — multi-stage Dockerfile (Phase 0: HTTP embed path)
#
# Stage 1 (deps): build a venv with runtime + test deps + the package
# Stage 2 (runtime): slim runtime image with venv, source, tests, entrypoint
#
# Build:
#   docker build -f docker/Dockerfile -t memory-pgvector:latest .
#
# Forked from andreab67/hermes-memory-pgvector (BSD-3-Clause)

ARG PYTHON_VERSION=3.11
ARG DEBIAN_RELEASE=bookworm

# ---------------------------------------------------------------------------
# Stage 1: deps — install the package + test deps into a venv
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY pgvector/ ./pgvector/
COPY tests/ ./tests/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip wheel \
    && pip install ".[test]"

# ---------------------------------------------------------------------------
# Stage 2: runtime — slim image with venv, source, tests, entrypoint
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CMD_PROFILE=shell \
    HF_HUB_OFFLINE=1

# Runtime system deps: libpq5 (psycopg), tini (PID 1), postgresql-client (psql for migration check)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        tini \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 hermes \
    && useradd -u 1000 -g 1000 -m -s /bin/bash hermes

COPY --from=deps /opt/venv /opt/venv
WORKDIR /app
COPY --chown=hermes:hermes pgvector/ /app/pgvector/
COPY --chown=hermes:hermes tests/ /app/tests/
COPY --chown=hermes:hermes docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER hermes
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["shell"]
