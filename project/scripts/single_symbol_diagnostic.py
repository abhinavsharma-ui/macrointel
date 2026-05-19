from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
NY = ZoneInfo("America/New_York")

sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / ".env.example", override=False)
LIVE_ROOT = Path(os.getenv("SIG_LIVE_ROOT", "data/features"))
REPORT_PATH = Path(os.getenv("SYMBOL_DIAG_REPORT", "reports/symbol_diagnostics.json"))
MIN_FEATURE_ROWS = int(os.getenv("SYMBOL_DIAG_MIN_FEATURE_ROWS", "260"))
os.chdir(ROOT)

import fixed_return_daily_signals as sig  # noqa: E402
from intraday_universe_refresh import fetch_live_prices, recompute_core_features, update_feature_file  # noqa: E402

_SEARCH_LLM_KEY_INDEX = 0
_SEARCH_LLM_DEAD_INDICES: set[int] = set()
_SEARCH_LLM_DEAD_AT: dict[int, float] = {}
_SEARCH_LLM_KEY_COUNT = 0


def _llm_filter_worker(queue, search_keys, key_index, dead_indices, candidate, spy_pct, qqq_pct, env_overrides):
    try:
        sig.LLM_API_KEYS = list(search_keys)
        sig._llm_key_index = int(key_index)
        sig._llm_dead_key_indices = set(dead_indices)
        for key, value in env_overrides.items():
            os.environ[key] = str(value)
        decisions = sig.call_llm_filter([candidate], spy_pct=spy_pct, qqq_pct=qqq_pct, num_signals=1)
        try:
            sig._save_news_cache()
            sig._save_sector_cache()
        except Exception:
            pass
        queue.put(
            {
                "decisions": decisions,
                "key_index": getattr(sig, "_llm_key_index", 0),
                "dead_indices": sorted(getattr(sig, "_llm_dead_key_indices", set())),
                "error": None,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "decisions": {},
                "key_index": getattr(sig, "_llm_key_index", 0),
                "dead_indices": sorted(getattr(sig, "_llm_dead_key_indices", set())),
                "error": str(exc),
            }
        )


def norm_symbol(value: str) -> str:
    return Path(str(value).strip()).stem.replace("_US", "").replace(".US", "").upper()


def num(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def compact_float(value, digits: int = 6):
    try:
        value = float(value)
        if not math.isfinite(value):
            return None
        return round(value, digits)
    except Exception:
        return None


def gate(name: str, passed: bool, detail: str = "", value=None) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail, "value": value}


def find_feature_file(symbol: str, root: Path | None = None) -> Path | None:
    root = root or LIVE_ROOT
    for name in (f"{symbol}.parquet", f"{symbol}_US.parquet", f"{symbol}.US.parquet"):
        path = root / name
        if path.exists():
            return path
    return None


def lookup_alpaca_asset(symbol: str) -> dict:
    try:
        from alpaca.trading.client import TradingClient

        key = os.getenv("ALPACA_API_KEY", "")
        secret = os.getenv("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            return {"ok": False, "status": "alpaca_keys_missing"}
        client = TradingClient(key, secret, paper=os.getenv("ALPACA_PAPER", "true").lower() != "false")
        asset = client.get_asset(symbol)
        return {
            "ok": True,
            "symbol": getattr(asset, "symbol", symbol),
            "name": getattr(asset, "name", ""),
            "exchange": str(getattr(asset, "exchange", "")),
            "asset_class": str(getattr(asset, "asset_class", "")),
            "status": str(getattr(asset, "status", "")),
            "tradable": bool(getattr(asset, "tradable", False)),
            "marginable": bool(getattr(asset, "marginable", False)),
            "shortable": bool(getattr(asset, "shortable", False)),
        }
    except Exception as exc:
        return {"ok": False, "status": "asset_lookup_failed", "error": str(exc)[:240]}


def fetch_alpaca_daily_bars(symbol: str, days: int = 950) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    try:
        from alpaca.data.enums import DataFeed
        feed = DataFeed.IEX
    except Exception:
        feed = None

    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY/ALPACA_SECRET_KEY are required")
    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(365, days))

    kwargs = {"symbol_or_symbols": [symbol], "timeframe": TimeFrame.Day, "start": start, "end": end}
    if feed is not None:
        kwargs["feed"] = feed
    try:
        request = StockBarsRequest(**kwargs)
        bars = client.get_stock_bars(request)
    except TypeError:
        kwargs.pop("feed", None)
        request = StockBarsRequest(**kwargs)
        bars = client.get_stock_bars(request)
    raw = getattr(bars, "df", None)
    if raw is None or raw.empty:
        raise RuntimeError("alpaca returned no daily bars")

    frame = raw.copy()
    if isinstance(frame.index, pd.MultiIndex):
        names = list(frame.index.names)
        if "symbol" in names:
            frame = frame.xs(symbol, level="symbol")
        else:
            frame = frame.loc[symbol]
    frame = frame.reset_index()
    date_col = next((c for c in ("timestamp", "date", "time") if c in frame.columns), None)
    if date_col is None:
        raise RuntimeError("alpaca bars missing timestamp column")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce")
    if "close" not in out.columns:
        raise RuntimeError("alpaca bars missing close column")
    out = out.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
    out = out[out["close"] > 0].reset_index(drop=True)
    if len(out) < 260:
        raise RuntimeError(f"only {len(out)} daily bars; need at least 260")
    out["symbol"] = symbol
    return recompute_core_features(out)


def ensure_feature_file(symbol: str, refresh: bool = True, history_days: int = 950) -> dict:
    root = LIVE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    path = find_feature_file(symbol, root)
    existed = path is not None
    fetch_status = "already_in_system" if existed else "fetched_from_alpaca"
    asset = lookup_alpaca_asset(symbol)

    if path is None:
        path = root / f"{symbol}.parquet"
        df = fetch_alpaca_daily_bars(symbol, history_days)
        df.to_parquet(path, index=False)
        os.utime(path, None)

    mark = {}
    refresh_result = {}
    if refresh:
        marks = fetch_live_prices([symbol], batch_size=1)
        mark = marks.get(symbol, {})
        if mark.get("ok"):
            refresh_result = update_feature_file(path, mark, datetime.now(NY), dry_run=False)

    return {
        "path": str(path),
        "existed": existed,
        "source": fetch_status,
        "asset": asset,
        "live_mark": mark,
        "refresh": refresh_result,
    }


def latest_symbol_row(symbol: str, path: Path, features: list[str]) -> tuple[dict, dict]:
    df = sig.load_file(path)
    if df is None or df.empty:
        raise RuntimeError("feature file is empty or unreadable")
    sym = str(df["symbol"].iloc[-1] if "symbol" in df.columns and len(df) else symbol)
    last_row = df.iloc[-1]
    price = num(last_row.get("close"))
    adv = num(last_row.get("adv20_dollar_vol"))
    pct_today = None
    if len(df) >= 2:
        prev_close = num(df["close"].iloc[-2])
        pct_today = (price - prev_close) / prev_close * 100.0 if prev_close else None
    if "rsi_14" in df.columns:
        valid_feat = df[df["rsi_14"].notna()]
        feature_row = valid_feat.iloc[-1].copy() if len(valid_feat) else last_row.copy()
    else:
        feature_row = last_row.copy()
    row = {col: num(feature_row.get(col)) for col in features}
    row.update(
        {
            "symbol": norm_symbol(sym),
            "price": price,
            "entry_price": price,
            "close": price,
            "adv20_dollar_vol": adv,
            "pct_today": pct_today,
            "feature_date": str(last_row.get("date", ""))[:10],
        }
    )
    meta = {
        "rows": int(len(df)),
        "required_rows": MIN_FEATURE_ROWS,
        "price": price,
        "adv20_dollar_vol": adv,
        "pct_today": pct_today,
        "feature_date": row["feature_date"],
        "feature_row_date": str(feature_row.get("date", row["feature_date"]))[:10],
        "file_mtime_et": str(sig.file_mtime_et_date(path)),
        "fresh_today": sig.file_mtime_et_date(path) == datetime.now(NY).date(),
        "missing_model_features": [col for col in features if col not in df.columns],
    }
    return row, meta


def predict_probability(model, features: list[str], row: dict) -> float:
    x = pd.DataFrame([row])
    x = x.reindex(columns=features).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    proba = model.predict_proba(x)
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return float(proba[0, pos_idx])


def position_pct(symbol: str, probability: float, vol_multiplier: float) -> float:
    ref_half_kelly = 0.2725
    target_iv = 30.0
    pt_pct = sig.PROFIT_TARGET_PCT
    sl_pct = sig.STOP_LOSS_PCT
    edge = probability * (pt_pct / 100.0) - (1 - probability) * (sl_pct / 100.0)
    kelly_f = edge / (pt_pct / 100.0) if pt_pct > 0 else 0.5
    half_kelly = kelly_f * 0.5
    kelly_mult = max(0.25, min(2.0, half_kelly / ref_half_kelly))
    iv_sym = sig._per_symbol_iv.get(symbol.upper(), None)
    vol_scalar = max(0.5, min(2.0, target_iv / iv_sym)) if (iv_sym and iv_sym > 5) else 1.0
    raw = sig.BASE_POSITION_PCT * kelly_mult * vol_scalar * vol_multiplier
    return round(max(sig.BASE_POSITION_PCT * 0.25, min(sig.BASE_POSITION_PCT * 2.0, raw)), 6)


def factor_overlay_for_row(symbol: str, probability: float, row: dict) -> dict:
    try:
        import pandas as pd
        from factor_overlay import apply_factor_overlay

        frame = pd.DataFrame([{**row, "symbol": symbol, "probability": probability}])
        scored, report = apply_factor_overlay(frame)
        out = scored.iloc[0].to_dict()
        return {
            "status": report.get("status", "ok"),
            "factor_composite": compact_float(out.get("factor_composite"), 6),
            "factor_rank_score": compact_float(out.get("factor_rank_score"), 6),
            "momentum": compact_float(out.get("factor_momentum"), 6),
            "low_vol": compact_float(out.get("factor_low_vol"), 6),
            "quality": compact_float(out.get("factor_quality"), 6),
            "value": compact_float(out.get("factor_value"), 6),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:180]}


def options_diagnostics_for_row(symbol: str, row: dict) -> dict:
    try:
        from options_vol_diagnostics import cached_or_fetch

        return cached_or_fetch(symbol, row=row)
    except Exception as exc:
        return {"symbol": symbol, "status": "error", "error": str(exc)[:180]}


def quick_allowed_universe(symbol: str, row: dict) -> tuple[bool, str]:
    try:
        if sig.RUNTIME_STATE.exists():
            data = json.loads(sig.RUNTIME_STATE.read_text(encoding="utf-8"))
            base = {sig.norm_sym(k) for k in (data.get("signal_store") or {}).keys()}
            if symbol in base:
                return True, "runtime_state"
    except Exception:
        pass
    if symbol.endswith((".NS", ".BO", ".NSE", ".BSE")):
        return False, "non_us_suffix"
    eff_min_adv = sig.MIN_ADV * (1.0 - sig.ADV_CUT)
    if num(row.get("price")) <= 0:
        return False, "missing_price"
    if num(row.get("adv20_dollar_vol")) < eff_min_adv:
        return False, f"adv_below_expanded_universe_floor_{eff_min_adv:.0f}"
    return True, "feature_adv_expanded_universe"


def universe_rank(symbol: str, probability: float) -> dict:
    """Fast dashboard context from the latest actual cron score cache."""
    try:
        scores_path = getattr(sig, "SCORES_JSON", Path("reports/fixed_return_daily_scores.json"))
        if scores_path.exists():
            data = json.loads(scores_path.read_text(encoding="utf-8"))
            scores = data.get("scores", []) if isinstance(data, dict) else []
            probs = [num(s.get("probability")) for s in scores if isinstance(s, dict)]
            symbols = {sig.norm_sym(s.get("symbol", "")) for s in scores if isinstance(s, dict)}
            sorted_probs = sorted(probs + ([] if symbol in symbols else [probability]), reverse=True)
            if symbol in symbols:
                match = next((s for s in scores if sig.norm_sym(s.get("symbol", "")) == symbol), {})
                all_rank = int(match.get("ml_rank") or 0) or None
            else:
                all_rank = sum(1 for p in probs if p > probability) + 1
            final_cutoff = None
            final_symbols = set()
            if sig.OUT_JSON.exists():
                final = json.loads(sig.OUT_JSON.read_text(encoding="utf-8"))
                final_signals = final.get("signals", []) if isinstance(final, dict) else []
                final_probs = [num(s.get("probability")) for s in final_signals if isinstance(s, dict)]
                final_symbols = {sig.norm_sym(s.get("symbol", "")) for s in final_signals if isinstance(s, dict)}
                final_cutoff = min(final_probs) if final_probs else None
            would_clear_cutoff = probability >= final_cutoff if final_cutoff is not None else probability >= sig.SIG_THRESHOLD
            return {
                "ok": True,
                "method": "latest_actual_cron_score_cache",
                "exact_full_rescore": True,
                "scored_universe_count": data.get("scored_universe_count") or len(scores),
                "rank": all_rank,
                "qualified_count": data.get("qualified_count"),
                "top_n": data.get("top_n") or sig.SIG_TOP_N,
                "selected_by_top_n": symbol in final_symbols or would_clear_cutoff,
                "top_n_cutoff_probability": final_cutoff,
                "latest_report_symbols": sorted(final_symbols),
            }
        if not sig.OUT_JSON.exists():
            return {
                "ok": True,
                "method": "no_saved_daily_report",
                "rank": None,
                "scored_universe_count": None,
                "qualified_count": None,
                "top_n": sig.SIG_TOP_N,
                "selected_by_top_n": probability >= sig.SIG_THRESHOLD,
                "top_n_cutoff_probability": None,
            }
        data = json.loads(sig.OUT_JSON.read_text(encoding="utf-8"))
        signals = data.get("signals", []) if isinstance(data, dict) else []
        probs = [num(s.get("probability")) for s in signals if isinstance(s, dict)]
        selected_symbols = {sig.norm_sym(s.get("symbol", "")) for s in signals if isinstance(s, dict)}
        all_rank = next((i + 1 for i, s in enumerate(signals) if sig.norm_sym(s.get("symbol", "")) == symbol), None)
        cutoff = None
        if probs:
            cutoff = min(probs)
        would_clear_cutoff = probability >= cutoff if cutoff is not None else probability >= sig.SIG_THRESHOLD
        return {
            "ok": True,
            "method": "latest_saved_daily_report",
            "exact_full_rescore": False,
            "scored_universe_count": data.get("scored_universe_count"),
            "rank": all_rank,
            "qualified_count": data.get("qualified_count"),
            "top_n": sig.SIG_TOP_N,
            "selected_by_top_n": symbol in selected_symbols or would_clear_cutoff,
            "top_n_cutoff_probability": cutoff,
            "latest_report_symbols": sorted(selected_symbols),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:240]}


def earnings_feature_gate(row: dict) -> dict:
    blocked = []
    for col, threshold in {"official_event_hit": 1, "filing_event_hit": 1, "event_day_extreme": 1}.items():
        if num(row.get(col)) >= threshold:
            blocked.append(col)
    shock_cap = float(os.getenv("SIG_PEER_SHOCK_CAP", "2.0"))
    if abs(num(row.get("peer_earnings_shock_3d"))) >= shock_cap:
        blocked.append("peer_earnings_shock_3d")
    return {"passed": not blocked, "blocked_by": blocked}


def earnings_calendar_gate(symbol: str) -> dict:
    try:
        import yfinance as yf

        hold_end = datetime.now(NY).date() + timedelta(days=sig.HOLD_DAYS + 2)
        cal = yf.Ticker(symbol).calendar
        if not cal:
            return {"passed": True, "status": "no_calendar"}
        earn_dates = cal.get("Earnings Date", [])
        if not isinstance(earn_dates, list):
            earn_dates = [earn_dates]
        hit_dates = []
        for item in earn_dates:
            if item is None:
                continue
            dt = item.date() if hasattr(item, "date") else item
            if datetime.now(NY).date() <= dt <= hold_end:
                hit_dates.append(str(dt))
        return {"passed": not hit_dates, "status": "checked", "blocked_dates": hit_dates}
    except Exception as exc:
        return {"passed": True, "status": "check_failed_open", "error": str(exc)[:180]}


def search_key_order(search_keys: list[str]) -> list[int]:
    n = len(search_keys)
    return [
        idx
        for offset in range(n)
        for idx in [(_SEARCH_LLM_KEY_INDEX + offset) % n]
        if idx not in _SEARCH_LLM_DEAD_INDICES
    ]


def retire_search_key(idx: int, status_code, reason: str = "") -> None:
    global _SEARCH_LLM_KEY_INDEX, _SEARCH_LLM_DEAD_INDICES, _SEARCH_LLM_DEAD_AT
    _SEARCH_LLM_DEAD_INDICES.add(idx)
    _SEARCH_LLM_DEAD_AT[idx] = time.time()
    if not _SEARCH_LLM_KEY_COUNT:
        return
    for offset in range(1, _SEARCH_LLM_KEY_COUNT + 1):
        next_idx = (idx + offset) % _SEARCH_LLM_KEY_COUNT
        if next_idx not in _SEARCH_LLM_DEAD_INDICES:
            _SEARCH_LLM_KEY_INDEX = next_idx
            break
    print(f"SYMBOL DIAG LLM key[{idx}] retired status={status_code} {reason}", flush=True)


def extract_json_dict(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip().strip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    data = json.loads(text)
    return data if isinstance(data, dict) else {}


def collect_llm_tool_evidence(symbol: str, signal: dict, row: dict, spy_pct, qqq_pct, sector: str) -> dict:
    calls = [
        ("check_short_interest", {"symbol": symbol}),
        ("check_options_iv", {"symbol": symbol}),
        ("check_price_momentum", {"symbol": symbol}),
        ("check_sector_performance", {"symbol": symbol, "sector": sector}),
        ("analyze_news_risk", {"symbol": symbol}),
        (
            "assess_trade_setup",
            {
                "symbol": symbol,
                "probability": signal.get("probability"),
                "pct_today": row.get("pct_today"),
                "spy_pct": spy_pct,
                "qqq_pct": qqq_pct,
                "sector": sector,
            },
        ),
        ("check_catalyst_events", {"symbol": symbol, "days": 30}),
    ]
    evidence = {}
    for name, args in calls:
        try:
            out = sig._handle_tool_call(name, dict(args))
            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except Exception:
                    out = {"raw": out}
            evidence[name] = out
        except Exception as exc:
            evidence[name] = {"status": "tool_error", "error": str(exc)[:240]}
    return evidence


def run_llm_with_precomputed_evidence(symbol: str, signal: dict, row: dict, signal_date: str, search_keys: list[str]) -> dict:
    global _SEARCH_LLM_KEY_INDEX
    started = time.time()
    try:
        import requests
    except ImportError:
        return {"ran": False, "status": "requests_missing"}
    spy_pct, qqq_pct = sig.market_context_pct()
    sector = sig.get_sector(symbol) or ""
    headlines = sig.fetch_news(symbol, signal_date)
    candidate = {
        "symbol": symbol,
        "probability": signal["probability"],
        "pct_today": row.get("pct_today"),
        "sector": sector,
        "headlines": headlines,
    }
    evidence = collect_llm_tool_evidence(symbol, signal, row, spy_pct, qqq_pct, sector)
    spy_str = f"{spy_pct:+.2f}%" if spy_pct is not None else "unknown"
    qqq_str = f"{qqq_pct:+.2f}%" if qqq_pct is not None else "unknown"
    user_message = "\n".join(
        [
            f"Market today: SPY {spy_str}, QQQ {qqq_str}.",
            "Evaluate exactly one candidate using the same MacroIntelligence LLM filter rules.",
            "The production tool outputs are precomputed below from the same tool functions used by the nightly pipeline.",
            "",
            "Candidate:",
            json.dumps(candidate, sort_keys=True),
            "",
            "Tool evidence:",
            json.dumps(evidence, sort_keys=True),
            "",
            f"Return ONLY raw JSON in this exact shape: {{\"{symbol}\":{{\"decision\":\"proceed|reduce_half|skip\",\"reason\":\"brief but specific\",\"confidence\":\"high|medium|low\",\"gates\":{{}}}}}}",
        ]
    )
    messages = [{"role": "system", "content": sig.LLM_SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://macro-intelligence.local",
        "X-Title": "MacroIntelligence",
    }
    timeout = num(os.getenv("SYMBOL_DIAG_LLM_REQUEST_TIMEOUT_SECONDS", "30"), 30.0)
    models = [sig.LLM_MODEL] + [m for m in getattr(sig, "LLM_FALLBACK_MODELS", []) if m != sig.LLM_MODEL]
    last_error = ""
    for model in models:
        for idx in search_key_order(search_keys):
            try:
                headers["Authorization"] = f"Bearer {search_keys[idx]}"
                body = {"model": model, "messages": messages, "temperature": 0.1}
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=timeout)
                if resp.status_code in (401, 402, 403, 429):
                    retire_search_key(idx, resp.status_code, f"model={model}")
                    continue
                resp.raise_for_status()
                _SEARCH_LLM_KEY_INDEX = idx
                msg = resp.json()["choices"][0]["message"]
                text = (msg.get("content") or msg.get("reasoning") or "").strip()
                if not text:
                    for block in msg.get("reasoning_details") or []:
                        if isinstance(block, dict) and block.get("text"):
                            text = block["text"]
                            break
                decisions = {k.upper(): v for k, v in extract_json_dict(text).items()}
                decision = decisions.get(symbol) or decisions.get(symbol.upper()) or {}
                if not decision:
                    last_error = "no decision in JSON"
                    continue
                return {
                    "ran": True,
                    "status": "ok",
                    "mode": "precomputed_production_tools",
                    "duration_seconds": round(time.time() - started, 2),
                    "decision": str(decision.get("decision", "reduce_half")).lower().strip(),
                    "reason": str(decision.get("reason", "")).strip(),
                    "confidence": str(decision.get("confidence", "medium")).lower().strip(),
                    "raw": decision,
                    "candidate": candidate,
                    "tool_evidence": evidence,
                }
            except Exception as exc:
                last_error = str(exc)[:240]
                print(f"SYMBOL DIAG LLM {model} key[{idx}] failed: {last_error}", flush=True)
                continue
    status = "search_llm_keys_exhausted_or_rate_limited" if len(_SEARCH_LLM_DEAD_INDICES) >= len(search_keys) else "no_structured_decision"
    return {
        "ran": True,
        "status": status,
        "mode": "precomputed_production_tools",
        "duration_seconds": round(time.time() - started, 2),
        "error": last_error,
        "candidate": candidate,
        "tool_evidence": evidence,
    }


def _llm_evidence_worker(queue, symbol, signal, row, signal_date, search_keys, key_index, dead_indices, key_count):
    global _SEARCH_LLM_KEY_INDEX, _SEARCH_LLM_DEAD_INDICES, _SEARCH_LLM_KEY_COUNT
    try:
        _SEARCH_LLM_KEY_INDEX = int(key_index)
        _SEARCH_LLM_DEAD_INDICES = set(dead_indices)
        _SEARCH_LLM_KEY_COUNT = int(key_count)
        llm = run_llm_with_precomputed_evidence(symbol, signal, row, signal_date, list(search_keys))
        queue.put(
            {
                "llm": llm,
                "key_index": _SEARCH_LLM_KEY_INDEX,
                "dead_indices": sorted(_SEARCH_LLM_DEAD_INDICES),
                "error": None,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "llm": {"ran": True, "status": "search_llm_worker_error", "mode": "precomputed_production_tools", "error": str(exc)[:240]},
                "key_index": _SEARCH_LLM_KEY_INDEX,
                "dead_indices": sorted(_SEARCH_LLM_DEAD_INDICES),
                "error": str(exc)[:240],
            }
        )


def run_llm_evidence_with_timeout(symbol: str, signal: dict, row: dict, signal_date: str, search_keys: list[str]) -> dict:
    global _SEARCH_LLM_KEY_INDEX, _SEARCH_LLM_DEAD_INDICES, _SEARCH_LLM_DEAD_AT
    started = time.time()
    hard_timeout = num(os.getenv("SYMBOL_DIAG_LLM_HARD_TIMEOUT_SECONDS", "125"), 125.0)
    hard_timeout = max(30.0, hard_timeout)
    queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_llm_evidence_worker,
        args=(
            queue,
            symbol,
            signal,
            row,
            signal_date,
            search_keys,
            _SEARCH_LLM_KEY_INDEX,
            sorted(_SEARCH_LLM_DEAD_INDICES),
            _SEARCH_LLM_KEY_COUNT,
        ),
        daemon=True,
    )
    proc.start()
    proc.join(hard_timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "ran": True,
            "status": "search_llm_timeout",
            "mode": "precomputed_production_tools",
            "duration_seconds": round(time.time() - started, 2),
            "timeout_seconds": hard_timeout,
        }
    try:
        result = queue.get_nowait()
    except Exception:
        result = {}
    dead_after = set(result.get("dead_indices", [])) if isinstance(result, dict) else set(_SEARCH_LLM_DEAD_INDICES)
    _SEARCH_LLM_KEY_INDEX = int(result.get("key_index", _SEARCH_LLM_KEY_INDEX)) if isinstance(result, dict) else _SEARCH_LLM_KEY_INDEX
    now = time.time()
    for idx in dead_after:
        _SEARCH_LLM_DEAD_AT.setdefault(idx, now)
    _SEARCH_LLM_DEAD_AT = {idx: ts for idx, ts in _SEARCH_LLM_DEAD_AT.items() if idx in dead_after}
    _SEARCH_LLM_DEAD_INDICES = set(dead_after)
    llm = result.get("llm") if isinstance(result, dict) else None
    if isinstance(llm, dict):
        llm.setdefault("duration_seconds", round(time.time() - started, 2))
        return llm
    return {
        "ran": True,
        "status": "search_llm_worker_no_result",
        "mode": "precomputed_production_tools",
        "duration_seconds": round(time.time() - started, 2),
    }


def run_llm(symbol: str, signal: dict, row: dict, signal_date: str) -> dict:
    global _SEARCH_LLM_KEY_INDEX, _SEARCH_LLM_DEAD_INDICES, _SEARCH_LLM_DEAD_AT, _SEARCH_LLM_KEY_COUNT
    if not sig.LLM_FILTER_ENABLED:
        return {"ran": False, "status": "disabled"}
    search_keys = list(getattr(sig, "LLM_SEARCH_API_KEYS", []) or [])
    if not search_keys:
        raw = [k.strip() for k in os.getenv("GROQ_API_KEY", "").split(",") if k.strip()]
        search_keys = raw[-1:] if raw else []
    if not search_keys:
        return {"ran": False, "status": "keys_missing"}
    if _SEARCH_LLM_KEY_COUNT != len(search_keys):
        _SEARCH_LLM_KEY_INDEX = 0
        _SEARCH_LLM_DEAD_INDICES = set()
        _SEARCH_LLM_DEAD_AT = {}
        _SEARCH_LLM_KEY_COUNT = len(search_keys)
    else:
        ttl = num(os.getenv("SYMBOL_DIAG_DEAD_KEY_TTL_SECONDS", "1800"), 1800.0)
        if ttl > 0:
            now = time.time()
            expired = {idx for idx, marked_at in _SEARCH_LLM_DEAD_AT.items() if now - marked_at >= ttl}
            if expired:
                _SEARCH_LLM_DEAD_INDICES = _SEARCH_LLM_DEAD_INDICES - expired
                _SEARCH_LLM_DEAD_AT = {idx: ts for idx, ts in _SEARCH_LLM_DEAD_AT.items() if idx not in expired}
    llm_mode = os.getenv("SYMBOL_DIAG_LLM_MODE", "precomputed_production_tools").strip().lower()
    if llm_mode not in {"production_tool_loop", "tool_loop"}:
        return run_llm_evidence_with_timeout(symbol, signal, row, signal_date, search_keys)
    spy_pct, qqq_pct = sig.market_context_pct()
    sector = sig.get_sector(symbol) or ""
    headlines = sig.fetch_news(symbol, signal_date)
    candidate = {
        "symbol": symbol,
        "probability": signal["probability"],
        "pct_today": row.get("pct_today"),
        "sector": sector,
        "headlines": headlines,
    }
    started = time.time()
    env_overrides = {
        "LLM_TOOL_SLEEP_SECONDS": os.getenv("SYMBOL_DIAG_LLM_TOOL_SLEEP_SECONDS", "0.35"),
        "LLM_REQUEST_TIMEOUT_SECONDS": os.getenv("SYMBOL_DIAG_LLM_REQUEST_TIMEOUT_SECONDS", "20"),
        "LLM_TOTAL_TIMEOUT_SECONDS": os.getenv("SYMBOL_DIAG_LLM_TOTAL_TIMEOUT_SECONDS", "115"),
        "LLM_MAX_ROUNDS": os.getenv("SYMBOL_DIAG_LLM_MAX_ROUNDS", "10"),
    }
    hard_timeout = num(
        os.getenv("SYMBOL_DIAG_LLM_HARD_TIMEOUT_SECONDS", env_overrides["LLM_TOTAL_TIMEOUT_SECONDS"]),
        115.0,
    )
    hard_timeout = max(30.0, hard_timeout)
    queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_llm_filter_worker,
        args=(
            queue,
            search_keys,
            _SEARCH_LLM_KEY_INDEX,
            sorted(_SEARCH_LLM_DEAD_INDICES),
            candidate,
            spy_pct,
            qqq_pct,
            env_overrides,
        ),
        daemon=True,
    )
    proc.start()
    proc.join(hard_timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "ran": True,
            "status": "search_llm_timeout",
            "duration_seconds": round(time.time() - started, 2),
            "timeout_seconds": hard_timeout,
            "candidate": candidate,
        }
    try:
        result = queue.get_nowait()
    except Exception:
        result = {}
    decisions = result.get("decisions", {}) if isinstance(result, dict) else {}
    dead_after = set(result.get("dead_indices", [])) if isinstance(result, dict) else set(_SEARCH_LLM_DEAD_INDICES)
    _SEARCH_LLM_KEY_INDEX = int(result.get("key_index", _SEARCH_LLM_KEY_INDEX)) if isinstance(result, dict) else _SEARCH_LLM_KEY_INDEX
    now = time.time()
    for idx in dead_after:
        _SEARCH_LLM_DEAD_AT.setdefault(idx, now)
    _SEARCH_LLM_DEAD_AT = {idx: ts for idx, ts in _SEARCH_LLM_DEAD_AT.items() if idx in dead_after}
    _SEARCH_LLM_DEAD_INDICES = set(dead_after)
    if isinstance(result, dict) and result.get("error"):
        return {
            "ran": True,
            "status": "search_llm_worker_error",
            "duration_seconds": round(time.time() - started, 2),
            "error": str(result.get("error"))[:240],
            "candidate": candidate,
        }
    decision = decisions.get(symbol) or decisions.get(symbol.upper()) or {}
    if not decision:
        status = "search_llm_keys_exhausted_or_rate_limited" if len(dead_after) >= len(search_keys) else "no_structured_decision"
        return {
            "ran": True,
            "status": status,
            "duration_seconds": round(time.time() - started, 2),
            "candidate": candidate,
        }
    return {
        "ran": True,
        "status": "ok",
        "duration_seconds": round(time.time() - started, 2),
        "decision": str(decision.get("decision", "reduce_half")).lower().strip(),
        "reason": str(decision.get("reason", "")).strip(),
        "confidence": str(decision.get("confidence", "medium")).lower().strip(),
        "raw": decision,
        "candidate": candidate,
    }


def portable_context(symbol: str, signal: dict, row: dict, rank: dict, gates: list[dict], sector: str) -> dict:
    gate_map = {str(g.get("name")): bool(g.get("passed")) for g in gates}
    failed = [str(g.get("name")) for g in gates if not g.get("passed")]
    threshold = sig.SIG_THRESHOLD
    probability = num(signal.get("probability"))
    cutoff = rank.get("top_n_cutoff_probability")
    top_n_gap = None if cutoff is None else round(probability - num(cutoff), 6)
    target_pct = sig.PROFIT_TARGET_PCT
    stop_pct = sig.STOP_LOSS_PCT

    tech_keys = [
        "pct_today",
        "rsi_14",
        "trend_10d",
        "pct_from_52w_high",
        "return_5d",
        "return_10d",
        "return_20d",
        "adv20_dollar_vol",
        "iv_pct",
        "put_call_ratio",
        "short_pct_of_float",
    ]
    technical = {}
    for key in tech_keys:
        if key in row and row.get(key) is not None:
            technical[key] = compact_float(row.get(key), 4)
    for key in sorted(row):
        if key in technical or key in {"symbol"}:
            continue
        val = compact_float(row.get(key), 4)
        if val is not None:
            technical[key] = val

    if not failed:
        pipeline_stage = "passes_all_pre_trade_gates"
    elif "ml_threshold" in failed:
        pipeline_stage = "research_candidate_only"
    else:
        pipeline_stage = "hard_blocked_before_llm"

    notes = [
        f"ML probability {probability:.3f} vs threshold {threshold:.3f}",
        f"Target {target_pct:.2f}% / stop {stop_pct:.2f}% / hold {sig.HOLD_DAYS} business days",
    ]
    if sector:
        notes.append(f"Sector: {sector}")
    if failed:
        notes.append("Failed gates: " + ", ".join(failed))

    return {
        "symbol": symbol,
        "pipeline_stage": pipeline_stage,
        "top_n_used_as_gate": False,
        "passes_ml_threshold": gate_map.get("ml_threshold", False),
        "failed_gates": failed,
        "ml_threshold": threshold,
        "ml_margin": round(probability - threshold, 6),
        "top_n_cutoff_probability": cutoff,
        "top_n_margin": top_n_gap,
        "rank_method": rank.get("method"),
        "rank_exact_full_rescore": bool(rank.get("exact_full_rescore")),
        "position_size_pct": round(num(signal.get("position_pct")) * 100.0, 4),
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "reward_to_risk": round(target_pct / stop_pct, 3) if stop_pct else None,
        "hold_days": sig.HOLD_DAYS,
        "sector": sector,
        "technical_snapshot": technical,
        "notes": notes,
    }


def final_verdict(gates: list[dict], llm: dict) -> tuple[str, bool]:
    failed = [g for g in gates if not g.get("passed")]
    if failed:
        first = failed[0]["name"]
        if first == "blocklist":
            return "blocked_symbol", False
        if first == "ml_threshold":
            return "below_ml_threshold", False
        return f"blocked_by_{first}", False
    if llm.get("ran") and llm.get("decision") == "skip":
        return "llm_skip", False
    if llm.get("ran") and llm.get("decision") == "proceed":
        return "allowed_proceed", True
    if llm.get("ran") and llm.get("decision") == "reduce_half":
        return "allowed_reduce_half", True
    return "allowed_before_llm", True


def build_diagnostics(
    symbol: str,
    signal: dict,
    row: dict,
    feature_meta: dict,
    feature_info: dict,
    rank: dict,
    gates: list[dict],
    llm: dict,
    sector: str,
    regime: str,
    vol: float,
    vol_mult: float,
    model_path: Path,
    features: list[str],
) -> dict:
    failed = [g for g in gates if not g.get("passed")]
    failed_names = [g["name"] for g in failed]
    feature_snapshot = {}
    for key in sorted(row):
        if key == "symbol":
            continue
        val = compact_float(row.get(key), 6)
        if val is not None:
            feature_snapshot[key] = val
    return {
        "decision": {
            "would_trade_stock_only": not failed and llm.get("decision") != "skip",
            "failed_gates": failed_names,
            "top_n_capacity_gate_ignored": True,
            "top_n_note": "Top-N is daily portfolio-capacity context only; this single-stock diagnostic does not block on it.",
            "llm_ran": bool(llm.get("ran")),
            "llm_decision": llm.get("decision"),
            "llm_status": llm.get("status"),
            "llm_reason": llm.get("reason"),
            "llm_forced": bool(llm.get("forced")),
            "llm_research_only": bool(llm.get("research_only")),
            "llm_ignored_failed_gates": llm.get("ignored_failed_gates", []),
            "llm_blocking_failed_gates": llm.get("blocking_failed_gates", []),
        },
        "gates": gates,
        "data": {
            "passed": "data" not in failed_names,
            "rows": feature_meta.get("rows"),
            "required_rows": feature_meta.get("required_rows", MIN_FEATURE_ROWS),
            "meaning": f"Data fails when usable feature history is below {feature_meta.get('required_rows', MIN_FEATURE_ROWS)} rows. The model can still score, but production treats the history as thin.",
            "source": feature_info.get("source"),
            "feature_file": feature_info.get("path"),
            "feature_date": feature_meta.get("feature_date"),
            "feature_row_date": feature_meta.get("feature_row_date"),
            "file_mtime_et": feature_meta.get("file_mtime_et"),
            "fresh_today": feature_meta.get("fresh_today"),
            "missing_model_features": feature_meta.get("missing_model_features", []),
        },
        "model": {
            "path": str(model_path),
            "probability": signal.get("probability"),
            "threshold": sig.SIG_THRESHOLD,
            "ml_margin": round(num(signal.get("probability")) - sig.SIG_THRESHOLD, 6),
            "model_features": len(features),
            "missing_model_feature_count": len(feature_meta.get("missing_model_features", [])),
        },
        "daily_context": {
            "used_as_gate": False,
            "rank_method": rank.get("method"),
            "rank": rank.get("rank"),
            "scored_universe_count": rank.get("scored_universe_count"),
            "top_n": rank.get("top_n"),
            "top_n_cutoff_probability": rank.get("top_n_cutoff_probability"),
            "top_n_margin": None if rank.get("top_n_cutoff_probability") is None else round(num(signal.get("probability")) - num(rank.get("top_n_cutoff_probability")), 6),
            "latest_report_symbols": rank.get("latest_report_symbols", []),
        },
        "risk_plan": {
            "entry_price": signal.get("entry_price"),
            "profit_target_price": signal.get("profit_target_price"),
            "stop_loss_price": signal.get("stop_loss_price"),
            "profit_target_pct": sig.PROFIT_TARGET_PCT,
            "stop_loss_pct": sig.STOP_LOSS_PCT,
            "hold_days": sig.HOLD_DAYS,
            "expected_exit_date": signal.get("expected_exit_date"),
            "position_pct": signal.get("position_pct"),
            "position_pct_display": round(num(signal.get("position_pct")) * 100.0, 4),
            "vol_multiplier": vol_mult,
        },
        "factor_overlay": signal.get("factor_composite") if signal.get("factor_composite") is not None else None,
        "options_diagnostics": signal.get("options_diagnostics"),
        "market": {
            "sector": sector,
            "regime": regime,
            "spy_realized_vol": round(vol, 4),
            "price": row.get("price"),
            "adv20_dollar_vol": row.get("adv20_dollar_vol"),
            "pct_today": row.get("pct_today"),
            "alpaca_asset": feature_info.get("asset", {}),
            "live_mark": feature_info.get("live_mark", {}),
            "refresh": feature_info.get("refresh", {}),
        },
        "features": {
            "snapshot": feature_snapshot,
        },
    }


def diagnose_symbol(
    symbol: str,
    refresh: bool = True,
    run_llm_filter: bool = True,
    history_days: int = 950,
    force_llm: bool = False,
) -> dict:
    symbol = norm_symbol(symbol)
    if not symbol or not symbol.replace(".", "").isalnum():
        return {"ok": False, "error": "invalid symbol"}

    model_path, model, features = sig.load_model()
    vol, regime, vol_mult = sig.spy_vol()
    feature_info = ensure_feature_file(symbol, refresh=refresh, history_days=history_days)
    path = Path(feature_info["path"])
    row, feature_meta = latest_symbol_row(symbol, path, features)
    probability = predict_probability(model, features, row)
    row["probability"] = probability
    factor_context = factor_overlay_for_row(symbol, probability, row)
    option_context = options_diagnostics_for_row(symbol, row)
    if option_context.get("iv_pct"):
        sig._per_symbol_iv[symbol.upper()] = float(option_context["iv_pct"])

    allowed_universe_pass, allowed_reason = quick_allowed_universe(symbol, row)
    sector = sig.get_sector(symbol) or ""
    sector_pass = not (sector and sector.lower() in sig.EXCLUDE_SECTORS_LOWER)
    blocklist_pass = sig.is_allowed_symbol(symbol)
    price_pass = row["price"] >= sig.MIN_PRICE
    adv_pass = row["adv20_dollar_vol"] >= sig.MIN_ADV
    ml_pass = probability >= sig.SIG_THRESHOLD
    rank = universe_rank(symbol, probability)
    earnings_features = earnings_feature_gate(row)
    earnings_calendar = earnings_calendar_gate(symbol)

    regime_pass = True
    regime_detail = f"{regime} vol {vol:.2f}%"
    if sig.SIG_REGIME_FILTER_ENABLED:
        stressed, spy_ret = sig._check_spy_regime()
        regime_pass = (not stressed) or sig.SIG_REGIME_SOFT_HALF
        regime_detail = f"SPY {spy_ret:+.2f}%, soft_half={sig.SIG_REGIME_SOFT_HALF}"

    gates = [
        gate("blocklist", blocklist_pass, "symbol is not in ETF/blocklist files"),
        gate("data", feature_meta["rows"] >= MIN_FEATURE_ROWS, f"{feature_meta['rows']} / {MIN_FEATURE_ROWS} required feature rows from {feature_info['source']}"),
        gate("freshness", feature_meta["fresh_today"], f"feature file mtime ET {feature_meta['file_mtime_et']}"),
        gate("price", price_pass, f"minimum ${sig.MIN_PRICE:.2f}", round(row["price"], 4)),
        gate("liquidity", adv_pass, f"minimum ADV ${sig.MIN_ADV:,.0f}", round(row["adv20_dollar_vol"], 2)),
        gate("allowed_universe", allowed_universe_pass, allowed_reason),
        gate("sector", sector_pass, sector or "unknown"),
        gate("ml_threshold", ml_pass, f"threshold {sig.SIG_THRESHOLD:.3f}", round(probability, 6)),
        gate("earnings_features", earnings_features["passed"], ",".join(earnings_features["blocked_by"]) or "clear"),
        gate("earnings_calendar", earnings_calendar["passed"], earnings_calendar.get("status", "")),
        gate("regime", regime_pass, regime_detail),
    ]

    signal_date = datetime.now(NY).date().isoformat()
    expected_exit_date = (pd.Timestamp(signal_date) + pd.tseries.offsets.BDay(sig.HOLD_DAYS)).date().isoformat()
    signal = {
        "rank": rank.get("rank"),
        "symbol": symbol,
        "probability": round(probability, 6),
        "entry_price": round(row["price"], 4),
        "position_pct": position_pct(symbol, probability, vol_mult),
        "base_position_pct": sig.BASE_POSITION_PCT,
        "vol_multiplier": vol_mult,
        "profit_target_price": round(row["price"] * (1 + sig.PROFIT_TARGET_PCT / 100.0), 4),
        "stop_loss_price": round(row["price"] * (1 - sig.STOP_LOSS_PCT / 100.0), 4),
        "expected_exit_date": expected_exit_date,
        "hold_days": sig.HOLD_DAYS,
        "feature_date": feature_meta["feature_date"],
        "factor_composite": factor_context.get("factor_composite"),
        "factor_rank_score": factor_context.get("factor_rank_score"),
        "options_diagnostics": {k: option_context.get(k) for k in ("status", "iv_pct", "iv_rank_proxy", "iv_vs_realized", "put_call_ratio", "skew_proxy", "flags")},
    }

    pre_llm_gate_names = {
        "blocklist", "data", "freshness", "price", "liquidity", "allowed_universe",
        "sector", "ml_threshold", "earnings_features", "earnings_calendar",
    }
    pre_llm_failed = [g["name"] for g in gates if g["name"] in pre_llm_gate_names and not g.get("passed")]
    hard_force_blockers = {"blocklist", "allowed_universe"}
    force_ignored = [name for name in pre_llm_failed if name not in hard_force_blockers]
    force_blockers = [name for name in pre_llm_failed if name in hard_force_blockers]
    should_run_llm = run_llm_filter and (not pre_llm_failed or (force_llm and not force_blockers))
    if should_run_llm:
        llm = run_llm(symbol, signal, row, signal_date)
        llm["forced"] = bool(force_llm and force_ignored)
        llm["research_only"] = bool(force_llm and force_ignored)
        llm["ignored_failed_gates"] = force_ignored
        llm["blocking_failed_gates"] = force_blockers
    else:
        llm = {
            "ran": False,
            "status": "not_run_actual_pipeline_failed_" + (pre_llm_failed[0] if pre_llm_failed else "llm_disabled"),
            "failed_pre_llm_gates": pre_llm_failed,
            "force_llm_requested": bool(force_llm),
            "blocking_failed_gates": force_blockers,
        }
    verdict, would_trade = final_verdict(gates, llm)
    friend_context = portable_context(symbol, signal, row, rank, gates, sector)
    diagnostics = build_diagnostics(
        symbol=symbol,
        signal=signal,
        row=row,
        feature_meta=feature_meta,
        feature_info=feature_info,
        rank=rank,
        gates=gates,
        llm=llm,
        sector=sector,
        regime=regime,
        vol=vol,
        vol_mult=vol_mult,
        model_path=model_path,
        features=features,
    )

    payload = {
        "ok": True,
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "would_trade": would_trade,
        "signal": signal,
        "model": {
            "path": str(model_path),
            "threshold": sig.SIG_THRESHOLD,
            "top_n": sig.SIG_TOP_N,
            "profit_target_pct": sig.PROFIT_TARGET_PCT,
            "stop_loss_pct": sig.STOP_LOSS_PCT,
            "hold_days": sig.HOLD_DAYS,
        },
        "feature": feature_meta,
        "source": feature_info,
        "sector": sector,
        "rank": rank,
        "gates": gates,
        "pre_llm_failed_gates": pre_llm_failed,
        "friend_context": friend_context,
        "diagnostics": diagnostics,
        "factor_overlay": factor_context,
        "options_diagnostics": option_context,
        "earnings_features": earnings_features,
        "earnings_calendar": earnings_calendar,
        "llm": llm,
    }
    append_report(payload)
    return payload


def append_report(payload: dict) -> None:
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if REPORT_PATH.exists():
            old = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            existing = old if isinstance(old, list) else old.get("diagnostics", [])
        existing = [payload] + existing[:49]
        REPORT_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception:
        pass


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run one-symbol MacroIntel fixed-return diagnostic.")
    parser.add_argument("symbol")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--force-llm", action="store_true", help="Run LLM for research even when ML/top-N gates fail.")
    parser.add_argument("--history-days", type=int, default=950)
    args = parser.parse_args()
    result = diagnose_symbol(
        args.symbol,
        refresh=not args.no_refresh,
        run_llm_filter=not args.no_llm,
        history_days=args.history_days,
        force_llm=args.force_llm,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
