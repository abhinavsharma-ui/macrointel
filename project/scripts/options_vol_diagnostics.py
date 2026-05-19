from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CACHE_PATH = Path(os.getenv("OPTIONS_DIAG_CACHE", "data/options_vol_diagnostics_cache.json"))
REPORT_PATH = Path(os.getenv("OPTIONS_DIAG_REPORT", "reports/options_vol_diagnostics.json"))


def _num(value, default=None):
    try:
        if value is None or value == "":
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _load_cache() -> dict:
    try:
        if CACHE_PATH.exists():
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _row_value(row, *names):
    if row is None:
        return None
    for name in names:
        try:
            if isinstance(row, dict) and name in row:
                return row.get(name)
            if hasattr(row, "get"):
                return row.get(name)
        except Exception:
            continue
    return None


def cached_or_fetch(symbol: str, row=None, max_age_hours: float | None = None) -> dict:
    symbol = str(symbol or "").upper().strip()
    max_age_hours = _num(max_age_hours, _num(os.getenv("OPTIONS_DIAG_CACHE_HOURS", "12"), 12.0))
    cache = _load_cache()
    cached = cache.get(symbol)
    now = time.time()
    if isinstance(cached, dict):
        age = now - _num(cached.get("_cached_epoch"), 0.0)
        if max_age_hours and age <= max_age_hours * 3600:
            out = dict(cached)
            out["cache_hit"] = True
            return out
    out = analyze_symbol(symbol, row=row)
    out["_cached_epoch"] = now
    cache[symbol] = out
    _save_cache(cache)
    return out


def analyze_symbol(symbol: str, row=None) -> dict:
    """Return options diagnostics using yfinance when available.

    This is a diagnostic layer only. It never throws and never blocks trading by
    itself; hard decisions remain in the ML gates and LLM risk filter.
    """
    symbol = str(symbol or "").upper().strip()
    realized = _num(_row_value(row, "realized_vol_21d", "hist_vol_30", "atr_pct"))
    if realized is not None and realized < 1:
        realized *= 100.0
    price = _num(_row_value(row, "price", "entry_price", "close"))

    base = {
        "symbol": symbol,
        "status": "unavailable",
        "iv_pct": None,
        "iv_rank_proxy": None,
        "iv_vs_realized": None,
        "put_call_ratio": None,
        "skew_proxy": None,
        "options_volume": None,
        "flags": [],
        "realized_vol_pct": round(realized, 3) if realized is not None else None,
        "price": round(price, 4) if price is not None else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import yfinance as yf

        tk = yf.Ticker(symbol)
        expirations = list(getattr(tk, "options", []) or [])
        if not expirations:
            base["status"] = "no_options_chain"
            base["flags"].append("no_options_chain")
            return base
        chain = tk.option_chain(expirations[0])
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        if calls.empty and puts.empty:
            base["status"] = "empty_options_chain"
            return base
        if price is None:
            try:
                price = _num((tk.fast_info or {}).get("last_price"))
            except Exception:
                price = None
        if price is None:
            price = _num(getattr(tk, "info", {}).get("regularMarketPrice"))
        all_opts = pd.concat([calls.assign(_side="call"), puts.assign(_side="put")], ignore_index=True)
        all_opts["strike_distance"] = (pd.to_numeric(all_opts.get("strike"), errors="coerce") - float(price or 0)).abs()
        all_opts["impliedVolatility"] = pd.to_numeric(all_opts.get("impliedVolatility"), errors="coerce")
        atm = all_opts.dropna(subset=["strike_distance", "impliedVolatility"]).sort_values("strike_distance").head(6)
        if atm.empty:
            base["status"] = "iv_missing"
            return base
        iv_pct = float(atm["impliedVolatility"].median() * 100.0)
        call_vol = float(pd.to_numeric(calls.get("volume"), errors="coerce").fillna(0).sum()) if "volume" in calls.columns else 0.0
        put_vol = float(pd.to_numeric(puts.get("volume"), errors="coerce").fillna(0).sum()) if "volume" in puts.columns else 0.0
        pcr = put_vol / call_vol if call_vol > 0 else None
        skew = None
        try:
            otm_put = puts[pd.to_numeric(puts["strike"], errors="coerce") < float(price)].copy()
            otm_call = calls[pd.to_numeric(calls["strike"], errors="coerce") > float(price)].copy()
            put_iv = pd.to_numeric(otm_put.get("impliedVolatility"), errors="coerce").dropna().median()
            call_iv = pd.to_numeric(otm_call.get("impliedVolatility"), errors="coerce").dropna().median()
            if pd.notna(put_iv) and pd.notna(call_iv):
                skew = float((put_iv - call_iv) * 100.0)
        except Exception:
            skew = None
        iv_vs_realized = iv_pct / realized if realized and realized > 0 else None
        iv_rank_proxy = max(0.0, min(100.0, iv_pct))
        flags = []
        if iv_pct >= 100:
            flags.append("catastrophic_iv")
        elif iv_pct >= 75:
            flags.append("high_iv")
        if iv_vs_realized is not None and iv_vs_realized >= 1.8:
            flags.append("iv_premium_elevated")
        if pcr is not None and pcr >= 2.0:
            flags.append("put_volume_skew")
        if skew is not None and skew >= 20:
            flags.append("downside_skew")
        return {
            **base,
            "status": "ok",
            "iv_pct": round(iv_pct, 3),
            "iv_rank_proxy": round(iv_rank_proxy, 3),
            "iv_vs_realized": round(iv_vs_realized, 3) if iv_vs_realized is not None else None,
            "put_call_ratio": round(pcr, 3) if pcr is not None else None,
            "skew_proxy": round(skew, 3) if skew is not None else None,
            "options_volume": round(call_vol + put_vol, 2),
            "flags": flags,
        }
    except Exception as exc:
        base["status"] = "error"
        base["error"] = str(exc)[:240]
        return base


def write_report(rows: list[dict]) -> dict:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "diagnostics": rows,
        "high_risk": [r for r in rows if any(flag in r.get("flags", []) for flag in ("catastrophic_iv", "high_iv", "put_volume_skew", "downside_skew"))],
    }
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        payload["write_error"] = str(exc)[:180]
    return payload
