#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$ROOT_DIR/run_symbol_expansion.sh"
SCREEN_NAME="databuilder"

if [ ! -f "$RUNNER" ]; then
  echo "[ERROR] Missing runner: $RUNNER"
  exit 1
fi

screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
screen -dmS "$SCREEN_NAME" bash "$RUNNER"

echo "Data builder started in screen '$SCREEN_NAME'"
echo "Monitor fetch log: tail -f $ROOT_DIR/fetch_symbols.log"
echo "Monitor build log: tail -f $ROOT_DIR/build_features.log"
echo "Monitor Finnhub log: tail -f $ROOT_DIR/finnhub_features.log"
echo "Attach session: screen -r $SCREEN_NAME"
