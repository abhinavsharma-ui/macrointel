#!/usr/bin/env bash
# Start the current MacroIntel dashboard in the background.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$ROOT_DIR/project"
VENV_DIR="$ROOT_DIR/venv"
PORT="${DASHBOARD_ULTRA_PORT:-${PORT:-5055}}"
LOG_FILE="${DASHBOARD_LOG:-/tmp/dash.log}"

echo "MacroIntel dashboard start"
echo "root: $ROOT_DIR"
echo "port: $PORT"
echo "log:  $LOG_FILE"

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "[setup] Installing requirements"
  pip install -q -r "$PROJECT_DIR/requirements.txt"
fi

if [ ! -f "$PROJECT_DIR/.env" ]; then
  echo "[warn] Missing $PROJECT_DIR/.env. Public defaults may work, but live services need private keys."
fi

if pgrep -f "dashboard_ultra.py" >/dev/null 2>&1; then
  echo "[start] Existing dashboard_ultra.py process found. Stop it first with ./stop.sh."
  pgrep -af "dashboard_ultra.py" || true
  exit 0
fi

cd "$PROJECT_DIR"
export DASHBOARD_ULTRA_PORT="$PORT"
export PORT="$PORT"
nohup python dashboard_ultra.py > "$LOG_FILE" 2>&1 &
PID="$!"

sleep 2

if ps -p "$PID" >/dev/null 2>&1; then
  echo "[ok] Dashboard running as pid $PID"
  echo "[ok] URL: http://127.0.0.1:$PORT"
  echo "[ok] Tail logs: tail -f $LOG_FILE"
else
  echo "[error] Dashboard exited. Last log lines:"
  tail -80 "$LOG_FILE" 2>/dev/null || true
  exit 1
fi
