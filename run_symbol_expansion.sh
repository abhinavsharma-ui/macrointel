#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$ROOT_DIR/project"
VENV_DIR="$ROOT_DIR/venv"
FETCH_LOG="$ROOT_DIR/fetch_symbols.log"
BUILD_LOG="$ROOT_DIR/build_features.log"
OPTIONS_LOG="$ROOT_DIR/options_features.log"
EARNINGS_LOG="$ROOT_DIR/earnings_insider_features.log"

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
: > "$OPTIONS_LOG"
: > "$EARNINGS_LOG"

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
  echo "[$(timestamp)] Starting options feature enrichment"
  activate_venv
  python "$PROJECT_DIR/build_options_features.py" --log-file "$OPTIONS_LOG"
  echo "[$(timestamp)] Options feature enrichment completed successfully"
} >> "$OPTIONS_LOG" 2>&1

{
  echo "[$(timestamp)] Starting earnings + insider feature enrichment"
  activate_venv
  python "$PROJECT_DIR/build_earnings_insider_features.py" --log-file "$EARNINGS_LOG"
  echo "[$(timestamp)] Earnings + insider enrichment completed successfully"
} >> "$EARNINGS_LOG" 2>&1

echo "[$(timestamp)] Symbol expansion pipeline finished"
