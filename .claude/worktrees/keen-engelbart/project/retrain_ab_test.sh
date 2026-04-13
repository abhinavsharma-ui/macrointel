#!/usr/bin/env bash
set -u -o pipefail

# A/B/C retrain runner for meta model.
# - Runs three predefined configs
# - Captures walk-forward summary for each
# - Picks winner by objective score
# - Promotes winner only if it passes safety floors
# - Otherwise restores baseline artifacts

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

VENV_ACTIVATE="${ROOT_DIR}/.venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: missing venv at $VENV_ACTIVATE"
  exit 1
fi
source "$VENV_ACTIVATE"

ENV_FILE="${ROOT_DIR}/.env"
CHK_DIR="${ROOT_DIR}/models/checkpoints"
mkdir -p "$CHK_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${CHK_DIR}/abtest_${STAMP}"
BASELINE_DIR="${WORK_DIR}/baseline"
CANDIDATE_DIR="${WORK_DIR}/candidates"
mkdir -p "$BASELINE_DIR" "$CANDIDATE_DIR"

echo "[0/6] Backing up baseline checkpoints..."
cp "${CHK_DIR}/meta_walkforward_report.json" "${BASELINE_DIR}/" 2>/dev/null || true
cp "${CHK_DIR}/meta_directional_models.joblib" "${BASELINE_DIR}/" 2>/dev/null || true
cp "${CHK_DIR}/meta_walkforward_report.previous.json" "${BASELINE_DIR}/" 2>/dev/null || true
cp "${CHK_DIR}/meta_directional_models.previous.joblib" "${BASELINE_DIR}/" 2>/dev/null || true

upsert_env() {
  local key="$1"
  local value="$2"
  if [[ -f "$ENV_FILE" ]] && rg -q "^${key}=" "$ENV_FILE"; then
    python3 - "$ENV_FILE" "$key" "$value" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]); k = sys.argv[2]; v = sys.argv[3]
lines = p.read_text(encoding="utf-8").splitlines()
out = []
replaced = False
for line in lines:
    if line.startswith(k + "="):
        out.append(f"{k}={v}")
        replaced = True
    else:
        out.append(line)
if not replaced:
    out.append(f"{k}={v}")
p.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
  else
    printf "%s=%s\n" "$key" "$value" >> "$ENV_FILE"
  fi
}

apply_common() {
  upsert_env "META_MODEL_RULE_OBJECTIVE" "precision"
  upsert_env "AUTO_RETRAIN_ON_START" "0"
  upsert_env "BYBIT_WS_LIBRARY_PING" "0"
}

apply_config() {
  local cfg="$1"
  case "$cfg" in
    A)
      upsert_env "META_MODEL_TAKE_THRESHOLD" "0.62"
      upsert_env "META_MODEL_RUNTIME_THRESHOLD" "0.62"
      upsert_env "META_MODEL_MIN_COVERAGE_PCT" "0.8"
      upsert_env "META_MODEL_MAX_COVERAGE_PCT" "10"
      upsert_env "META_MODEL_MIN_RULE_SAMPLES" "35"
      ;;
    B)
      upsert_env "META_MODEL_TAKE_THRESHOLD" "0.68"
      upsert_env "META_MODEL_RUNTIME_THRESHOLD" "0.68"
      upsert_env "META_MODEL_MIN_COVERAGE_PCT" "0.5"
      upsert_env "META_MODEL_MAX_COVERAGE_PCT" "7"
      upsert_env "META_MODEL_MIN_RULE_SAMPLES" "45"
      ;;
    C)
      upsert_env "META_MODEL_TAKE_THRESHOLD" "0.58"
      upsert_env "META_MODEL_RUNTIME_THRESHOLD" "0.58"
      upsert_env "META_MODEL_MIN_COVERAGE_PCT" "1.0"
      upsert_env "META_MODEL_MAX_COVERAGE_PCT" "12"
      upsert_env "META_MODEL_MIN_RULE_SAMPLES" "30"
      ;;
    *)
      echo "Unknown config: $cfg"
      return 1
      ;;
  esac
}

score_report() {
  local report_path="$1"
  python3 - "$report_path" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print("nan nan nan nan -999 FAIL")
    sys.exit(0)
d = json.loads(p.read_text(encoding="utf-8"))
s = ((d.get("walk_forward") or {}).get("summary") or {})
prec = float(s.get("mean_precision", 0) or 0)
hit = float(s.get("mean_taken_hit_rate_pct", 0) or 0)
edge = float(s.get("mean_taken_edge_pct", 0) or 0)
cov = float(s.get("mean_coverage_pct", 0) or 0)
# safety floor
ok = (prec >= 0.30 and hit >= 40.0 and edge > 0.0 and cov >= 0.5 and cov <= 12.0)
# objective for ranking among valid candidates
obj = (120.0 * prec) + (0.35 * hit) + (12.0 * edge) - (0.7 * abs(cov - 3.0))
print(f"{prec} {hit} {edge} {cov} {obj} {'OK' if ok else 'FAIL'}")
PY
}

BEST_CFG=""
BEST_OBJ="-999999"
BEST_STATUS="FAIL"

echo "[1/6] Running A/B/C retrain experiments..."
for CFG in A B C; do
  echo "---- Config ${CFG} ----"
  apply_common
  apply_config "$CFG" || continue

  python3 retrain_institutional_models.py
  RET=$?
  if [[ $RET -ne 0 ]]; then
    echo "Config ${CFG}: retrain failed (exit=${RET})"
    continue
  fi

  OUT="$(score_report "${CHK_DIR}/meta_walkforward_report.json")"
  PREC="$(echo "$OUT" | awk '{print $1}')"
  HIT="$(echo "$OUT" | awk '{print $2}')"
  EDGE="$(echo "$OUT" | awk '{print $3}')"
  COV="$(echo "$OUT" | awk '{print $4}')"
  OBJ="$(echo "$OUT" | awk '{print $5}')"
  STATUS="$(echo "$OUT" | awk '{print $6}')"
  echo "Config ${CFG} summary: precision=${PREC} hit=${HIT} edge=${EDGE} coverage=${COV} objective=${OBJ} status=${STATUS}"

  mkdir -p "${CANDIDATE_DIR}/${CFG}"
  cp "${CHK_DIR}/meta_walkforward_report.json" "${CANDIDATE_DIR}/${CFG}/" 2>/dev/null || true
  cp "${CHK_DIR}/meta_directional_models.joblib" "${CANDIDATE_DIR}/${CFG}/" 2>/dev/null || true
  cp "${CHK_DIR}/meta_walkforward_report.previous.json" "${CANDIDATE_DIR}/${CFG}/" 2>/dev/null || true
  cp "${CHK_DIR}/meta_directional_models.previous.joblib" "${CANDIDATE_DIR}/${CFG}/" 2>/dev/null || true
  printf "precision=%s\nhit=%s\nedge=%s\ncoverage=%s\nobjective=%s\nstatus=%s\n" \
    "$PREC" "$HIT" "$EDGE" "$COV" "$OBJ" "$STATUS" > "${CANDIDATE_DIR}/${CFG}/summary.txt"

  if [[ "$STATUS" == "OK" ]]; then
    if python3 - <<PY
best=float("${BEST_OBJ}")
curr=float("${OBJ}")
import sys
sys.exit(0 if curr > best else 1)
PY
    then
      BEST_OBJ="$OBJ"
      BEST_CFG="$CFG"
      BEST_STATUS="$STATUS"
    fi
  fi
done

echo "[2/6] Selecting winner..."
if [[ -n "$BEST_CFG" && "$BEST_STATUS" == "OK" ]]; then
  echo "Winner: ${BEST_CFG} (objective=${BEST_OBJ})"
  cp "${CANDIDATE_DIR}/${BEST_CFG}/meta_walkforward_report.json" "${CHK_DIR}/meta_walkforward_report.json"
  cp "${CANDIDATE_DIR}/${BEST_CFG}/meta_directional_models.joblib" "${CHK_DIR}/meta_directional_models.joblib"
else
  echo "No candidate passed safety floors. Restoring baseline."
  cp "${BASELINE_DIR}/meta_walkforward_report.json" "${CHK_DIR}/meta_walkforward_report.json" 2>/dev/null || true
  cp "${BASELINE_DIR}/meta_directional_models.joblib" "${CHK_DIR}/meta_directional_models.joblib" 2>/dev/null || true
fi

echo "[3/6] Restarting run.py..."
pkill -f "python run.py" || true
sleep 2
nohup python run.py > logs/system.out 2>&1 &

echo "[4/6] Checking health..."
sleep 10
HTTP_CODE="$(curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5050/ || true)"
echo "Health HTTP code: ${HTTP_CODE}"

echo "[5/6] Active summary:"
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("models/checkpoints/meta_walkforward_report.json")
if p.exists():
    d = json.loads(p.read_text(encoding="utf-8"))
    print((d.get("walk_forward") or {}).get("summary") or {})
else:
    print("meta_walkforward_report.json missing")
PY

echo "[6/6] Done."
echo "Artifacts: ${WORK_DIR}"
