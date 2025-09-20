#!/bin/bash
set -e

echo "Starting System Design Use Case API Server"
echo "Port: ${API_PORT:-8001}"
echo "Use Case: ${USE_CASE:-system_design}"

# Wait for shared database to be ready (if using external DB)
if [ ! -z "$DATABASE_URL" ]; then
    echo "Waiting for database..."
    while ! nc -z shared_research_db 5432; do
        sleep 0.1
    done
    echo "Database is ready!"
fi

# Set use case specific environment
export USE_CASE=system_design
export API_PORT=${API_PORT:-8001}

# Start the API server for this specific use case
echo "Starting System Design API server on port ${API_PORT}..."
exec uvicorn api.use_case_server:app \
    --host 0.0.0.0 \
    --port ${API_PORT} \
    --reload \
    --log-level info \
    --access-log \
    --app-dir /app
