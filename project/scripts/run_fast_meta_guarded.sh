#!/usr/bin/env bash
set -euo pipefail

LOG_OUT="${1:-logs/fast_meta_guarded_stdout.log}"
LOG_ERR="${2:-logs/fast_meta_guarded_stderr.log}"
MIN_POS="${META_MIN_POSITIVE_LABELS:-5000}"

rm -f "$LOG_OUT" "$LOG_ERR"
nohup python -u scripts/fast_meta_retrain_24.py > "$LOG_OUT" 2> "$LOG_ERR" &
PID=$!
echo "guarded_retrain_pid $PID"

while kill -0 "$PID" 2>/dev/null; do
  line=$(grep -hE "positive labels|positives=" "$LOG_OUT" "$LOG_ERR" 2>/dev/null | tail -1 || true)
  if [ -n "$line" ]; then
    echo "LABEL_LINE $line"
    pos=$(echo "$line" | sed -nE 's/.*\| ([0-9]+) positive labels.*/\1/p; s/.*positives=([0-9]+).*/\1/p' | tail -1)
    if [ -n "$pos" ]; then
      if [ "$pos" -lt "$MIN_POS" ]; then
        echo "ABORTING: positive labels $pos < required $MIN_POS"
        kill "$PID" 2>/dev/null || true
        sleep 3
        pkill -P "$PID" 2>/dev/null || true
        exit 2
      fi
      echo "LABEL_SANITY_OK positive labels $pos >= $MIN_POS"
      wait "$PID"
      exit $?
    fi
  fi
  sleep 20
done

wait "$PID"
