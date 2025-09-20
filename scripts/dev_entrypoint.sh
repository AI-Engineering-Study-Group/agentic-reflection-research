#!/bin/bash
set -e

echo "Starting Agentic Research Framework - Development Mode"

# Wait for database to be ready
echo "Waiting for database..."
while ! nc -z shared_research_db 5432; do
  sleep 0.1
done
echo "Database is ready!"

# Redis removed for simplicity

# Run any database migrations or setup here
# python -m alembic upgrade head

echo "Starting research framework API server..."
exec uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

