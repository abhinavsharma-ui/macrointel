#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$ROOT_DIR/project"
VENV_DIR="$ROOT_DIR/venv"
FETCH_LOG="$ROOT_DIR/fetch_symbols.log"
BUILD_LOG="$ROOT_DIR/build_features.log"
FINNHUB_LOG="$ROOT_DIR/finnhub_features.log"

timestamp() {
  date -u '+%Y-%m-%d %H:%M:%S UTC'
}

activate_venv() {
  if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  fi
}

cd "$ROOT_DIR"

: > "$FETCH_LOG"
: > "$BUILD_LOG"
: > "$FINNHUB_LOG"

{
  echo "[$(timestamp)] Starting symbol expansion run"
  echo "[$(timestamp)] Repo root: $ROOT_DIR"
  activate_venv
  python "$PROJECT_DIR/fetch_10yr_data.py" --all --workers 8
  echo "[$(timestamp)] Fetch completed successfully"
} >> "$FETCH_LOG" 2>&1

{
  echo "[$(timestamp)] Starting feature build"
  activate_venv
  python "$PROJECT_DIR/build_10yr_features.py"
  echo "[$(timestamp)] Feature build completed successfully"
} >> "$BUILD_LOG" 2>&1

{
  echo "[$(timestamp)] Starting Finnhub feature enrichment"
  activate_venv
  python "$PROJECT_DIR/build_finnhub_features.py"
  echo "[$(timestamp)] Finnhub feature enrichment completed successfully"
} >> "$FINNHUB_LOG" 2>&1

echo "[$(timestamp)] Symbol expansion pipeline finished"
