#!/bin/sh
# Downloads the licensed SMPL-X model file at container start, from a URL
# only the deployer controls — the weights themselves are never committed
# to this repo or baked into the image, since they're license-gated.
set -e

MODEL_DIR="${SMPLX_MODEL_DIR:-models/smplx}"
MODEL_FILE="$MODEL_DIR/SMPLX_${SMPLX_GENDER:-NEUTRAL}.npz"

if [ ! -f "$MODEL_FILE" ]; then
  if [ -z "$SMPLX_MODEL_URL" ]; then
    echo "ERROR: $MODEL_FILE is missing and SMPLX_MODEL_URL is not set."
    echo "Set SMPLX_MODEL_URL to a private URL you control that serves your"
    echo "own licensed SMPL-X download. See README.md."
    exit 1
  fi
  mkdir -p "$MODEL_DIR"
  echo "Downloading SMPL-X model from SMPLX_MODEL_URL..."
  curl -fsSL "$SMPLX_MODEL_URL" -o "$MODEL_FILE"
fi

exec uvicorn app:app --app-dir webapp --host 0.0.0.0 --port "${PORT:-8000}"
