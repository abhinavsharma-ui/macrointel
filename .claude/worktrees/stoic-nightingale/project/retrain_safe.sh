#!/usr/bin/env bash
set -euo pipefail

# Safe retrain runner:
# - sets precision-first env defaults in .env (idempotent upsert)
# - backs up checkpoints
# - runs retraining
# - validates new report against hard floors + previous report
# - auto-rolls back on failure
# - restarts run.py only on accepted model

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

VENV_ACTIVATE="${ROOT_DIR}/.venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: venv not found at $VENV_ACTIVATE"
  exit 1
fi
source "$VENV_ACTIVATE"

ENV_FILE="${ROOT_DIR}/.env"
mkdir -p "${ROOT_DIR}/models/checkpoints"

upsert_env() {
  local key="$1"
  local value="$2"
  if [[ -f "$ENV_FILE" ]] && rg -q "^${key}=" "$ENV_FILE"; then
    python3 - "$ENV_FILE" "$key" "$value" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
key = sys.argv[2]
val = sys.argv[3]
lines = path.read_text(encoding="utf-8").splitlines()
out = []
replaced = False
for line in lines:
    if line.startswith(key + "="):
        out.append(f"{key}={val}")
        replaced = True
    else:
        out.append(line)
if not replaced:
    out.append(f"{key}={val}")
path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
  else
    printf "%s=%s\n" "$key" "$value" >> "$ENV_FILE"
  fi
}

echo "[1/6] Applying Claude-pack runtime and training config..."
upsert_env "META_MODEL_RULE_OBJECTIVE" "precision"
upsert_env "META_MODEL_TAKE_THRESHOLD" "0.52"
upsert_env "META_MODEL_RUNTIME_THRESHOLD" "0.52"
upsert_env "META_MODEL_RUNTIME_MIN_EDGE_PCT" "0.18"
upsert_env "META_MODEL_RUNTIME_MIN_EDGE_RATIO" "0.08"
upsert_env "META_MODEL_RUNTIME_MIN_CONVICTION" "3.0"
upsert_env "META_MODEL_RUNTIME_LEADER_MIN_CONVICTION" "2.5"
upsert_env "META_MODEL_MIN_COVERAGE_PCT" "0.8"
upsert_env "META_MODEL_MAX_COVERAGE_PCT" "60"
upsert_env "META_MODEL_MIN_RULE_SAMPLES" "25"
upsert_env "META_MODEL_LABEL_MIN_EDGE_PCT" "0.18"
upsert_env "META_MODEL_LABEL_MIN_EDGE_RATIO" "0.08"
upsert_env "META_MODEL_HORIZON_DAYS" "8"
upsert_env "META_MODEL_WALKFORWARD_FOLDS" "6"
upsert_env "META_MODEL_MIN_TRAIN_DAYS" "200"
upsert_env "AUTO_RETRAIN_ON_START" "0"
upsert_env "BYBIT_WS_LIBRARY_PING" "0"

echo "[2/6] Backing up checkpoints..."
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${ROOT_DIR}/models/checkpoints/backup_before_retrain_${STAMP}"
mkdir -p "$BACKUP_DIR"
cp -r "${ROOT_DIR}/models/checkpoints/"* "$BACKUP_DIR/" 2>/dev/null || true

echo "[3/6] Running retrain_institutional_models.py (this can take a while)..."
python3 retrain_institutional_models.py

echo "[4/6] Validating new report..."
python3 - <<'PY'
import json
import pathlib
import sys

p = pathlib.Path("models/checkpoints/meta_walkforward_report.json")
q = pathlib.Path("models/checkpoints/meta_walkforward_report.previous.json")
if not p.exists():
    print("REJECT: new report missing")
    sys.exit(2)

new = json.loads(p.read_text(encoding="utf-8"))
new_s = ((new.get("walk_forward") or {}).get("summary") or {})
old_s = {}
if q.exists():
    old = json.loads(q.read_text(encoding="utf-8"))
    old_s = ((old.get("walk_forward") or {}).get("summary") or {})

def f(d, k):
    return float(d.get(k, 0) or 0)

np = f(new_s, "mean_precision")
nh = f(new_s, "mean_taken_hit_rate_pct")
ne = f(new_s, "mean_taken_edge_pct")
nc = f(new_s, "mean_coverage_pct")

op = f(old_s, "mean_precision")
oh = f(old_s, "mean_taken_hit_rate_pct")
oe = f(old_s, "mean_taken_edge_pct")
oc = f(old_s, "mean_coverage_pct")

print("NEW:", new_s)
print("OLD:", old_s)
print(f"NEW metrics -> precision={np:.4f}, hit={nh:.2f}, edge={ne:.3f}, coverage={nc:.2f}")

# hard floor
if np < 0.30 or nh < 40.0 or ne < -0.10:
    print("REJECT: below hard floor")
    sys.exit(3)

# degrade guard (if previous exists)
if old_s:
    if np < (op - 0.02) or nh < (oh - 3.0) or ne < (oe - 0.15):
        print("REJECT: materially worse than previous")
        sys.exit(4)

print("ACCEPT")
PY
VALID_EXIT=$?

if [[ $VALID_EXIT -ne 0 ]]; then
  echo "[5/6] Validation failed. Rolling back to previous directional checkpoint..."
  if [[ -f "${ROOT_DIR}/models/checkpoints/meta_walkforward_report.previous.json" && -f "${ROOT_DIR}/models/checkpoints/meta_directional_models.previous.joblib" ]]; then
    cp "${ROOT_DIR}/models/checkpoints/meta_walkforward_report.previous.json" "${ROOT_DIR}/models/checkpoints/meta_walkforward_report.json"
    cp "${ROOT_DIR}/models/checkpoints/meta_directional_models.previous.joblib" "${ROOT_DIR}/models/checkpoints/meta_directional_models.joblib"
  else
    echo "WARN: Previous checkpoint artifacts missing; cannot roll back directional bundle."
  fi
  echo "[6/6] Restarting run.py with rollback artifacts..."
  pkill -f "python run.py" || true
  sleep 2
  nohup python run.py > logs/system.out 2>&1 &
  echo "DONE: rollback path completed."
  exit 1
fi

echo "[5/6] Validation accepted."
echo "[6/6] Restarting run.py with new artifacts..."
pkill -f "python run.py" || true
sleep 2
nohup python run.py > logs/system.out 2>&1 &
sleep 8
HTTP_CODE="$(curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5050/ || true)"
echo "Health HTTP code: ${HTTP_CODE}"
echo "DONE: safe retrain completed."
