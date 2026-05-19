from __future__ import annotations

import json
import math
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
os.chdir(ROOT)

import fixed_return_daily_signals as sig  # noqa: E402
from intraday_universe_refresh import fetch_live_prices, recompute_core_features, update_feature_file  # noqa: E402


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
        "price": price,
        "adv20_dollar_vol": adv,
        "pct_today": pct_today,
        "feature_date": row["feature_date"],
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


def run_llm(symbol: str, signal: dict, row: dict, signal_date: str) -> dict:
    if not sig.LLM_FILTER_ENABLED:
        return {"ran": False, "status": "disabled"}
    search_keys = list(getattr(sig, "LLM_SEARCH_API_KEYS", []) or [])
    if not search_keys:
        raw = [k.strip() for k in os.getenv("GROQ_API_KEY", "").split(",") if k.strip()]
        search_keys = raw[-1:] if raw else []
    if not search_keys:
        return {"ran": False, "status": "keys_missing"}
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
    old_keys = sig.LLM_API_KEYS
    old_index = getattr(sig, "_llm_key_index", 0)
    old_dead = set(getattr(sig, "_llm_dead_key_indices", set()))
    old_sleep = os.environ.get("LLM_TOOL_SLEEP_SECONDS")
    try:
        sig.LLM_API_KEYS = search_keys
        sig._llm_key_index = 0
        sig._llm_dead_key_indices = set()
        os.environ["LLM_TOOL_SLEEP_SECONDS"] = os.getenv("SYMBOL_DIAG_LLM_TOOL_SLEEP_SECONDS", "0.35")
        decisions = sig.call_llm_filter([candidate], spy_pct, qqq_pct, 1)
        sig._save_news_cache()
        sig._save_sector_cache()
    finally:
        sig.LLM_API_KEYS = old_keys
        sig._llm_key_index = old_index
        sig._llm_dead_key_indices = old_dead
        if old_sleep is None:
            os.environ.pop("LLM_TOOL_SLEEP_SECONDS", None)
        else:
            os.environ["LLM_TOOL_SLEEP_SECONDS"] = old_sleep
    decision = decisions.get(symbol) or decisions.get(symbol.upper()) or {}
    if not decision:
        return {
            "ran": True,
            "status": "no_structured_decision",
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


def final_verdict(gates: list[dict], llm: dict) -> tuple[str, bool]:
    failed = [g for g in gates if not g.get("passed")]
    if failed:
        first = failed[0]["name"]
        if first == "blocklist":
            return "blocked_symbol", False
        if first == "ml_threshold":
            return "below_ml_threshold", False
        if first == "top_n":
            return "not_selected_by_daily_top_n", False
        return f"blocked_by_{first}", False
    if llm.get("ran") and llm.get("decision") == "skip":
        return "llm_skip", False
    if llm.get("ran") and llm.get("decision") == "proceed":
        return "allowed_proceed", True
    if llm.get("ran") and llm.get("decision") == "reduce_half":
        return "allowed_reduce_half", True
    return "allowed_before_llm", True


def diagnose_symbol(symbol: str, refresh: bool = True, run_llm_filter: bool = True, history_days: int = 950) -> dict:
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

    allowed_universe_pass, allowed_reason = quick_allowed_universe(symbol, row)
    sector = sig.get_sector(symbol) or ""
    sector_pass = not (sector and sector.lower() in sig.EXCLUDE_SECTORS_LOWER)
    blocklist_pass = sig.is_allowed_symbol(symbol)
    price_pass = row["price"] >= sig.MIN_PRICE
    adv_pass = row["adv20_dollar_vol"] >= sig.MIN_ADV
    ml_pass = probability >= sig.SIG_THRESHOLD
    rank = universe_rank(symbol, probability)
    top_n_pass = bool(rank.get("selected_by_top_n")) and allowed_universe_pass and ml_pass
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
        gate("data", feature_meta["rows"] >= 260, f"{feature_meta['rows']} feature rows from {feature_info['source']}"),
        gate("freshness", feature_meta["fresh_today"], f"feature file mtime ET {feature_meta['file_mtime_et']}"),
        gate("price", price_pass, f"minimum ${sig.MIN_PRICE:.2f}", round(row["price"], 4)),
        gate("liquidity", adv_pass, f"minimum ADV ${sig.MIN_ADV:,.0f}", round(row["adv20_dollar_vol"], 2)),
        gate("allowed_universe", allowed_universe_pass, allowed_reason),
        gate("sector", sector_pass, sector or "unknown"),
        gate("ml_threshold", ml_pass, f"threshold {sig.SIG_THRESHOLD:.3f}", round(probability, 6)),
        gate("top_n", top_n_pass, f"top {sig.SIG_TOP_N} context via {rank.get('method', 'unknown')}", rank.get("rank")),
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
    }

    pre_llm_gate_names = {
        "blocklist", "data", "freshness", "price", "liquidity", "allowed_universe",
        "sector", "ml_threshold", "top_n", "earnings_features", "earnings_calendar",
    }
    pre_llm_failed = [g["name"] for g in gates if g["name"] in pre_llm_gate_names and not g.get("passed")]
    should_run_llm = run_llm_filter and not pre_llm_failed
    llm = (
        run_llm(symbol, signal, row, signal_date)
        if should_run_llm
        else {
            "ran": False,
            "status": "not_run_actual_pipeline_failed_" + (pre_llm_failed[0] if pre_llm_failed else "llm_disabled"),
            "failed_pre_llm_gates": pre_llm_failed,
        }
    )
    verdict, would_trade = final_verdict(gates, llm)

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
    parser.add_argument("--history-days", type=int, default=950)
    args = parser.parse_args()
    result = diagnose_symbol(
        args.symbol,
        refresh=not args.no_refresh,
        run_llm_filter=not args.no_llm,
        history_days=args.history_days,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
