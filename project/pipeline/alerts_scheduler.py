"""
Alerts Scheduler & Production Logging
=====================================
Production logging plus a compatibility no-op alert dispatcher.
"""

import json
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class JSONFormatter(logging.Formatter):
    """Emit logs as structured JSON for easy parsing / log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


def setup_production_logging(log_dir: str = "logs", level: int = logging.INFO):
    """
    Configure root logger for production:
      - Console: human-readable
      - File: JSON, rotated daily, kept 30 days
      - Error: separate error-only file
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console_fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
        datefmt="%H:%M:%S",
    )
    console_h = logging.StreamHandler()
    console_h.setFormatter(console_fmt)
    console_h.setLevel(level)
    root.addHandler(console_h)

    json_h = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "system.jsonl"),
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
    json_h.setFormatter(JSONFormatter())
    json_h.setLevel(level)
    root.addHandler(json_h)

    err_h = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "errors.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    err_h.setFormatter(console_fmt)
    err_h.setLevel(logging.ERROR)
    root.addHandler(err_h)

    for noisy in ["urllib3", "websocket", "engineio", "socketio", "werkzeug"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info(f"Production logging configured (log_dir={log_dir!r})")


def dispatch_signal_alerts(signal_store: Dict, min_conviction: float = 7.0):
    """
    Compatibility no-op.
    """
    _ = signal_store
    _ = min_conviction
    return None
