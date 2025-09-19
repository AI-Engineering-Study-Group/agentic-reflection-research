#!/bin/bash
set -e

echo "Starting Agentic Research Framework - Research Mode"

# Wait for database to be ready
echo "Waiting for database..."
while ! nc -z research_db 5432; do
  sleep 0.1
done
echo "Database is ready!"

# Start Jupyter Lab
echo "Starting Jupyter Lab..."
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token=${JUPYTER_TOKEN:-research_token} \
    --notebook-dir=/app

