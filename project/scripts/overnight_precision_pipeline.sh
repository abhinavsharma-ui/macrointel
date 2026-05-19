#!/usr/bin/env bash
set -u

cd /home/abhinavsharma1359/macro_intelligence_complete/project
source ../venv/bin/activate
mkdir -p logs reports

echo "OVERNIGHT PIPELINE START $(date -u)" | tee -a logs/overnight_precision_pipeline.log

if pgrep -af "meta_precision_scout.py" >/dev/null; then
  echo "Scout already running. Waiting..." | tee -a logs/overnight_precision_pipeline.log
  while pgrep -af "meta_precision_scout.py" >/dev/null; do
    date -u | tee -a logs/overnight_precision_pipeline.log
    grep -hE "BASE|BEST|WROTE|Traceback|Error|Exception|Meta dataset build|Meta dataset ready" logs/meta_precision_scout.log 2>/dev/null | tail -40 | tee -a logs/overnight_precision_pipeline.log
    sleep 60
  done
else
  if [ ! -s reports/meta_precision_scout_recommended.env ]; then
    echo "Scout not running and env missing. Starting scout..." | tee -a logs/overnight_precision_pipeline.log
    python -u scripts/meta_precision_scout.py > logs/meta_precision_scout.log 2>&1
  else
    echo "Scout env already exists. Reusing reports/meta_precision_scout_recommended.env" | tee -a logs/overnight_precision_pipeline.log
  fi
fi

if [ ! -s reports/meta_precision_scout_recommended.env ]; then
  echo "FAILED: scout did not create reports/meta_precision_scout_recommended.env" | tee reports/overnight_meta_decision.txt
  exit 1
fi

echo "Scout finished. Recommended env:" | tee -a logs/overnight_precision_pipeline.log
cat reports/meta_precision_scout_recommended.env | tee -a logs/overnight_precision_pipeline.log

pkill -f "python -u scripts/fast_meta_retrain_24.py" || true
sleep 3

set -a
source reports/meta_precision_scout_recommended.env
set +a

export RETRAIN_META_LOG_EVERY=25
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting targeted retrain $(date -u)" | tee -a logs/overnight_precision_pipeline.log
nohup python -u scripts/fast_meta_retrain_24.py > logs/fast_meta_26yr_targeted_stdout.log 2> logs/fast_meta_26yr_targeted_stderr.log &
pid=$!
echo "targeted_pid=$pid" | tee -a logs/overnight_precision_pipeline.log

while kill -0 "$pid" 2>/dev/null; do
  date -u | tee -a logs/overnight_precision_pipeline.log
  grep -hE "FAST META|price-only ranking|Meta dataset build|Meta dataset ready|fold|Regressor|FAST_META_DONE|Traceback|Error|Exception" logs/fast_meta_26yr_targeted_stderr.log logs/fast_meta_26yr_targeted_stdout.log 2>/dev/null | tail -50 | tee -a logs/overnight_precision_pipeline.log
  sleep 120
done

echo "Targeted retrain process ended $(date -u)" | tee -a logs/overnight_precision_pipeline.log

python - <<'PY' | tee reports/overnight_meta_decision.txt
import json
from pathlib import Path

latest_path = Path("models/checkpoints/fast_meta_latest.txt")
print("latest_pointer_exists", latest_path.exists())
if not latest_path.exists():
    print("DEPLOYABLE False")
    print("reason missing_fast_meta_latest")
    raise SystemExit(0)

latest = latest_path.read_text().strip()
rp = Path(latest) / "meta_walkforward_report.json"
print("candidate_dir", latest)
print("report_exists", rp.exists())
if not rp.exists():
    print("DEPLOYABLE False")
    print("reason missing_report")
    raise SystemExit(0)

r = json.loads(rp.read_text())
s = r.get("summary") or (r.get("walk_forward") or {}).get("summary") or {}
for k in [
    "mean_accuracy",
    "mean_precision",
    "mean_recall",
    "mean_coverage_pct",
    "mean_taken_edge_pct",
    "mean_taken_drawdown_pct",
    "mean_taken_edge_draw_ratio",
    "mean_taken_hit_rate_pct",
]:
    print(k, s.get(k))

precision = float(s.get("mean_precision",0) or 0)
edge = float(s.get("mean_taken_edge_pct",0) or 0)
drawdown = float(s.get("mean_taken_drawdown_pct",99) or 99)
hit = float(s.get("mean_taken_hit_rate_pct",0) or 0)
coverage = float(s.get("mean_coverage_pct",0) or 0)
edr = float(s.get("mean_taken_edge_draw_ratio",0) or 0)

deployable = (
    precision >= 0.45 and
    edge >= 0.25 and
    drawdown <= 3.25 and
    hit >= 44.5 and
    coverage >= 0.20 and
    edr >= 0.35
)

print("DEPLOYABLE", deployable)
if not deployable:
    print("reason failed_gate")
else:
    print("reason passed_gate_manual_review_before_trading")
PY

echo "OVERNIGHT PIPELINE DONE $(date -u)" | tee -a logs/overnight_precision_pipeline.log
