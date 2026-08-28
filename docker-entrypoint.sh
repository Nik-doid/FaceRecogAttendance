#!/bin/sh
set -e

# No migrations to run: the service keeps no local database. Enrolment embeddings
# are cached under STORAGE_PATH and rebuilt from the photo sources when missing.
echo "Starting recognition service..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${API_PORT:-8000}" --workers 1
