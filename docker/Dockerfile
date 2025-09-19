# Multi-stage build with UV
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    git \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Development stage
FROM base as development

# Install all dependencies
RUN uv sync --all-extras

# Copy source code
COPY . .

# Development entrypoint
COPY scripts/dev_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
CMD ["/entrypoint.sh"]

# Research stage
FROM development as research

# Additional research tools
RUN uv sync --extra research

# Jupyter configuration
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.ip = '0.0.0.0'" > /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_server_config.py

COPY scripts/research_entrypoint.sh /research_entrypoint.sh
RUN chmod +x /research_entrypoint.sh

EXPOSE 8888
CMD ["/research_entrypoint.sh"]

