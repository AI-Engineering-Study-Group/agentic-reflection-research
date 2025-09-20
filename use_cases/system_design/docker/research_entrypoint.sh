#!/bin/bash
set -e

echo "Starting System Design Use Case - Research Mode"
echo "Jupyter Port: ${JUPYTER_PORT:-8891}"
echo "API Port: ${API_PORT:-8001}"

# Wait for database if needed
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
export JUPYTER_PORT=${JUPYTER_PORT:-8891}

# Start both API server and Jupyter in background for research
echo "Starting System Design API server on port ${API_PORT}..."
uvicorn api.use_case_server:app \
    --host 0.0.0.0 \
    --port ${API_PORT} \
    --reload \
    --log-level info &

# Wait a moment for API to start
sleep 3

echo "Starting Jupyter Lab for System Design research on port ${JUPYTER_PORT}..."
exec jupyter lab \
    --ip=0.0.0.0 \
    --port=${JUPYTER_PORT} \
    --no-browser \
    --allow-root \
    --NotebookApp.token=${JUPYTER_TOKEN:-system_design_research} \
    --notebook-dir=/app/use_cases/system_design
