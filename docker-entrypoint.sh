#!/bin/sh
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting recognition service..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${API_PORT:-8000}" --workers 1
