FROM python:3.11-slim AS builder

# Keep Python/pip lean and deterministic in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build-time deps:
# - build-essential: compile wheels when needed
# - git: required because pyproject depends on tvdatafeed via git URL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package into an isolated venv we can copy to runtime.
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install -U pip && \
    /opt/venv/bin/pip install .


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_TECH_HOME=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Runtime deps:
# - curl: used by docker-compose healthcheck (and optional debugging)
# - libgomp1: OpenMP runtime commonly needed by CatBoost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Create a non-root user and required writable dirs.
RUN adduser --disabled-password --gecos "" appuser && \
    mkdir -p /app/data /app/artifacts && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f "http://localhost:8000/v1/health" || exit 1

CMD ["model-tech", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


