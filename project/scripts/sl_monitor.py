"""Compatibility wrapper for fixed-return SL/PT monitoring.

The canonical exit and entry logic lives in fixed_return_paper_execute.py.
This wrapper keeps the existing sl_monitor cron alive without maintaining a
second price source, PnL formula, and ledger writer.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXECUTOR = ROOT / "scripts" / "fixed_return_paper_execute.py"


def run() -> int:
    print(f"SL monitor delegating to {EXECUTOR}")
    return subprocess.call([sys.executable, str(EXECUTOR)], cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(run())
