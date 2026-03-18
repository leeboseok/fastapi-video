#!/bin/sh
set -e

HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-11113}
WORKERS=${WORKERS:-1}

exec uvicorn nfs.main:app --host $HOST --port $PORT --workers $WORKERS --reload --reload-dir /app/nfs
