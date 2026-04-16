from __future__ import annotations

import logging
import time
from typing import MutableMapping

import structlog

_CONFIGURED = False


def _add_timestamp(_: object, __: str, event_dict: MutableMapping[str, object]) -> MutableMapping[str, object]:
    event_dict.setdefault("ts", round(time.time(), 6))
    return event_dict


def _event_to_msg(_: object, __: str, event_dict: MutableMapping[str, object]) -> MutableMapping[str, object]:
    event = event_dict.pop("event", None)
    if event is not None and "msg" not in event_dict:
        event_dict["msg"] = str(event)
    return event_dict


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            _add_timestamp,
            structlog.stdlib.add_log_level,
            _event_to_msg,
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(module_name: str) -> structlog.typing.FilteringBoundLogger:
    configure_logging()
    return structlog.get_logger().bind(module=module_name)
