"""
Lightweight performance knobs for running on laptops.

Goals:
- Reduce lag by capping native threadpools (BLAS/OpenMP/NumExpr/joblib).
- Optionally lower process priority on Windows to keep the UI responsive.

This module intentionally avoids importing heavy third‑party libraries so it can be
called very early (before NumPy/Pandas/XGBoost are imported).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    # joblib/loky: cap when code uses n_jobs=-1
    "LOKY_MAX_CPU_COUNT",
)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, *, min_value: int = 1, max_value: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        value = default
    else:
        try:
            value = int(float(str(raw).strip()))
        except Exception:
            value = default
    if max_value is not None:
        value = min(max_value, value)
    return max(min_value, value)


def _compute_default_threads(profile: str) -> int:
    cpu_count = os.cpu_count() or 2
    profile = (profile or "").strip().lower()
    if profile in {"full", "max", "server"}:
        return max(1, cpu_count)
    if profile in {"balanced", "default"}:
        return max(1, min(4, cpu_count))
    # laptop / lite (default)
    return max(1, min(2, cpu_count))


def _set_windows_priority(priority: str) -> bool:
    priority = (priority or "").strip().lower()
    if not priority:
        return False
    if not sys.platform.startswith("win"):
        return False

    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        set_priority_class = kernel32.SetPriorityClass
        get_current_process.restype = ctypes.c_void_p
        set_priority_class.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        set_priority_class.restype = ctypes.c_int

        classes = {
            "low": 0x00000040,  # IDLE_PRIORITY_CLASS
            "idle": 0x00000040,
            "below_normal": 0x00004000,  # BELOW_NORMAL_PRIORITY_CLASS
            "normal": 0x00000020,  # NORMAL_PRIORITY_CLASS
            "above_normal": 0x00008000,  # ABOVE_NORMAL_PRIORITY_CLASS
            "high": 0x00000080,  # HIGH_PRIORITY_CLASS
        }
        priority_class = classes.get(priority)
        if priority_class is None:
            return False

        handle = get_current_process()
        if not handle:
            return False

        ok = set_priority_class(handle, priority_class)
        return bool(ok)
    except Exception:
        return False


def _set_posix_nice(increment: int) -> bool:
    try:
        nice_fn = getattr(os, "nice", None)
        if nice_fn is None:
            return False
        nice_fn(int(increment))
        return True
    except Exception:
        return False


def apply_performance_profile() -> Dict[str, object]:
    """
    Apply thread caps and optional process-priority tweaks.

    Controlled via env vars:
    - PERF_PROFILE: laptop|balanced|full  (default: laptop)
    - MAX_CPU_THREADS: integer cap for native threadpools (default depends on PERF_PROFILE)
    - PROCESS_PRIORITY: windows priority class (below_normal recommended)
    - PERF_SET_PROCESS_PRIORITY: 1 to allow setting priority (default: 1 on Windows in laptop profile)
    """

    profile = (os.getenv("PERF_PROFILE") or os.getenv("PERFORMANCE_PROFILE") or "laptop").strip().lower() or "laptop"
    default_threads = _compute_default_threads(profile)
    threads = _env_int("MAX_CPU_THREADS", default_threads, min_value=1, max_value=(os.cpu_count() or None))

    applied_env: Dict[str, str] = {}
    if os.getenv("MAX_CPU_THREADS") is None:
        os.environ["MAX_CPU_THREADS"] = str(threads)
        applied_env["MAX_CPU_THREADS"] = str(threads)
    for name in _THREAD_ENV_VARS:
        if os.getenv(name) is None or not str(os.getenv(name)).strip():
            os.environ[name] = str(threads)
            applied_env[name] = str(threads)

    # HuggingFace tokenizers can spin up threads; disable unless explicitly enabled.
    if os.getenv("TOKENIZERS_PARALLELISM") is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        applied_env["TOKENIZERS_PARALLELISM"] = "false"

    priority_requested = (os.getenv("PROCESS_PRIORITY") or "").strip()
    allow_priority = _env_truthy(
        "PERF_SET_PROCESS_PRIORITY",
        default=(sys.platform.startswith("win") and profile in {"laptop", "lite"}),
    )
    priority_applied = False
    if allow_priority:
        if priority_requested:
            priority_applied = _set_windows_priority(priority_requested) or _set_posix_nice(5)
        elif profile in {"laptop", "lite"}:
            # Keep UI responsive on laptops by default.
            priority_applied = _set_windows_priority("below_normal") or _set_posix_nice(5)

    return {
        "profile": profile,
        "threads": threads,
        "env": applied_env,
        "process_priority_applied": priority_applied,
        "process_priority_requested": priority_requested,
    }
