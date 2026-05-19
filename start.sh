#!/bin/bash
# MacroIntelligence startup script for Google Cloud VM
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/project"
VENV_DIR="$SCRIPT_DIR/venv"
PORT="${PORT:-8888}"
export DASHBOARD_PORT="$PORT"
SCREEN_NAME="macro"
SCREENER_SCRIPT="$PROJECT_DIR/scripts/universe_screener.py"
MASTER_UNIVERSE_FILE="$PROJECT_DIR/data/full_universe.json"
LIVE_UNIVERSE_FILE="$PROJECT_DIR/data/live_universe.json"
LIVE_UNIVERSE_RELATIVE="data/live_universe.json"

echo "================================================"
echo "  MacroIntelligence Dashboard Launcher"
echo "================================================"

# Activate virtualenv
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Creating virtualenv..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install/update deps
echo "[setup] Installing requirements..."
pip install -q -r "$PROJECT_DIR/requirements.txt"

# Check .env exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
  echo "[ERROR] Missing $PROJECT_DIR/.env - copy your .env file first!"
  exit 1
fi

if [ ! -f "$SCREENER_SCRIPT" ]; then
  echo "[ERROR] Missing universe screener at $SCREENER_SCRIPT"
  exit 1
fi

echo "[setup] Refreshing live universe..."
python "$SCREENER_SCRIPT" \
  --input "$MASTER_UNIVERSE_FILE" \
  --output "$LIVE_UNIVERSE_FILE"

if [ ! -s "$LIVE_UNIVERSE_FILE" ]; then
  echo "[ERROR] Missing $LIVE_UNIVERSE_FILE after screener run"
  exit 1
fi

# Kill existing screen session if running
screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true

# Launch in screen
echo "[start] Launching on port $PORT inside screen '$SCREEN_NAME'..."
screen -dmS "$SCREEN_NAME" bash -c "
  cd '$PROJECT_DIR'
  source '$VENV_DIR/bin/activate'
  export UNIVERSE_FILE_PATH='$LIVE_UNIVERSE_RELATIVE'
  python run.py --port \"$PORT\" 2>&1 | tee '$SCRIPT_DIR/dashboard.log'
"

sleep 2

if screen -list | grep -q "$SCREEN_NAME"; then
  echo ""
  echo "  Dashboard running at http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_VM_IP'):$PORT"
  echo ""

  # Auto-launch watchdog if not already running
  WATCHDOG_SCRIPT="$PROJECT_DIR/scripts/watchdog.sh"
  if [ -f "$WATCHDOG_SCRIPT" ]; then
    if ! screen -list 2>/dev/null | grep -q "\.watchdog"; then
      echo "[watchdog] Starting watchdog in screen 'watchdog'..."
      screen -dmS watchdog bash "$WATCHDOG_SCRIPT"
      echo "[watchdog] Watchdog running"
    else
      echo "[watchdog] Already running"
    fi
  fi

  echo ""
  echo "  Useful commands:"
  echo "    screen -r $SCREEN_NAME     # attach to live session"
  echo "    screen -r watchdog         # attach to watchdog"
  echo "    tail -f $SCRIPT_DIR/dashboard.log  # view logs"
  echo "    tail -f $SCRIPT_DIR/logs/watchdog.log  # watchdog logs"
  echo "    $SCRIPT_DIR/stop.sh        # stop the server"
  echo "================================================"
else
  echo "[ERROR] Screen session failed to start. Check dashboard.log"
  cat "$SCRIPT_DIR/dashboard.log" 2>/dev/null | tail -30
  exit 1
fi
