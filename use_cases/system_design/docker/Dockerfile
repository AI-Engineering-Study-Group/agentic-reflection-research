# Multi-stage build using pip with pyproject.toml (compatible approach)
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./

# Install dependencies using pip (which can read pyproject.toml)
RUN pip install --no-cache-dir -e .[dev]

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_CASE=system_design
ENV API_PORT=8001

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set the working directory
WORKDIR /app

# Copy the source code
COPY . .

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting System Design Use Case API Server"\n\
echo "Port: ${API_PORT:-8001}"\n\
echo "Use Case: ${USE_CASE:-system_design}"\n\
\n\
# Set use case specific environment\n\
export USE_CASE=system_design\n\
export API_PORT=${API_PORT:-8001}\n\
\n\
# Verify setup\n\
echo "Python: $(python --version)"\n\
echo "Checking uvicorn availability..."\n\
\n\
if command -v uvicorn >/dev/null 2>&1; then\n\
    echo "Uvicorn: $(uvicorn --version)"\n\
    UVICORN_CMD="uvicorn"\n\
elif python -c "import uvicorn" >/dev/null 2>&1; then\n\
    echo "Uvicorn available via python -m"\n\
    UVICORN_CMD="python -m uvicorn"\n\
else\n\
    echo "ERROR: uvicorn not available"\n\
    echo "Installed packages:"\n\
    pip list | grep -E "(fastapi|uvicorn)" || echo "FastAPI/Uvicorn not found"\n\
    exit 1\n\
fi\n\
\n\
# Run the API server\n\
echo "Starting System Design API server on port ${API_PORT}..."\n\
exec $UVICORN_CMD api.use_case_server:app \\\n\
    --host 0.0.0.0 \\\n\
    --port ${API_PORT} \\\n\
    --log-level info \\\n\
    --access-log' > /entrypoint.sh && chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

EXPOSE ${API_PORT}
CMD ["/entrypoint.sh"]