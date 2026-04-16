#!/bin/bash
# Watchdog — restarts the macro screen session if it dies.
# Launch: screen -dmS watchdog bash scripts/watchdog.sh
#
# Checks every 60 seconds. Logs restarts to logs/watchdog.log.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_SCRIPT="$SCRIPT_DIR/../start.sh"
LOG_DIR="$SCRIPT_DIR/../logs"
LOG_FILE="$LOG_DIR/watchdog.log"
SCREEN_NAME="macro"
CHECK_INTERVAL=60

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$LOG_FILE"
}

log "Watchdog started (pid=$$, checking '$SCREEN_NAME' every ${CHECK_INTERVAL}s)"

while true; do
    sleep "$CHECK_INTERVAL"

    if screen -list 2>/dev/null | grep -q "\.$SCREEN_NAME[[:space:]]"; then
        # Session alive — nothing to do
        continue
    fi

    log "WARNING: screen '$SCREEN_NAME' not found — restarting via start.sh"
    if [ -x "$START_SCRIPT" ]; then
        bash "$START_SCRIPT" >> "$LOG_FILE" 2>&1
        log "start.sh exited with code $?"
    else
        log "ERROR: $START_SCRIPT not found or not executable"
    fi
done
