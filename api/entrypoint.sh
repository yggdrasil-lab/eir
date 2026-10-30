#!/bin/sh
set -e

# Swarm delivers secrets as files, but the app reads them from the environment.
if [ -n "${EIR_MCP_TOKEN_FILE:-}" ]; then
    EIR_MCP_TOKEN=$(cat "$EIR_MCP_TOKEN_FILE")
    export EIR_MCP_TOKEN
fi
if [ -n "${EIR_WEBHOOK_SECRET_FILE:-}" ]; then
    EIR_WEBHOOK_SECRET=$(cat "$EIR_WEBHOOK_SECRET_FILE")
    export EIR_WEBHOOK_SECRET
fi

# Retry database migration until Postgres accepts connections.
attempt=1
until alembic upgrade head; do
    if [ "$attempt" -ge 30 ]; then
        echo "ERROR: database not reachable after ${attempt} attempts" >&2
        exit 1
    fi
    echo "waiting for database (attempt ${attempt})"
    attempt=$((attempt + 1))
    sleep 2
done

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
