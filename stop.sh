#!/usr/bin/env bash
# Stop the current MacroIntel dashboard.
set -euo pipefail

if pgrep -f "dashboard_ultra.py" >/dev/null 2>&1; then
  echo "Stopping dashboard_ultra.py"
  pkill -f "dashboard_ultra.py"
  sleep 1
else
  echo "dashboard_ultra.py is not running"
fi
