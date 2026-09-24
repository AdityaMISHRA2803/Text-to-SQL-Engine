# Multi-purpose image for the text2sql stack.
# The same image is reused by both the `api` and `frontend` services in
# docker-compose.yml; the role is selected via the CMD override.
#
# Build:    docker build -t text2sql:latest .
# Run api:  docker run --rm -p 8000:8000 text2sql:latest api
# Run ui:   docker run --rm -p 8501:8501 text2sql:latest frontend
#
# Notes:
#   * python:3.11-slim keeps the image small (~150 MB).
#   * libpq5 is required at runtime by psycopg[binary] (binary wheel only).
#   * We install the project with `pip install .` so the text2sql-api /
#     text2sql-ui console scripts become available on PATH.
#   * /app is the working directory; docker-compose bind-mounts nothing on top
#     of it, so the install in the image is what runs.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# System deps. libpq5 is needed by psycopg[binary]; build-base is left out
# intentionally — every dependency in pyproject.toml ships a manylinux wheel.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what `pip install` needs first to leverage Docker layer caching.
COPY pyproject.toml README.md ./
COPY text2sql ./text2sql
COPY frontend ./frontend
COPY evals ./evals

RUN pip install --upgrade pip \
 && pip install .

# Quick healthcheck target. docker-compose wires this into the api service.
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
  CMD curl --fail --silent http://localhost:8000/health || exit 1

# Default command (overridden by docker-compose per service).
CMD ["text2sql-api"]
