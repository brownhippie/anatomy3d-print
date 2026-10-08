#!/bin/sh
set -e
exec uvicorn app:app --app-dir webapp --host 0.0.0.0 --port "${PORT:-8000}"
