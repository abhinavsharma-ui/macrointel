from __future__ import annotations

"""Refresh the fixed-return US feature store with recent daily OHLCV bars.

This is the cron-facing script used before fixed_return_daily_signals.py.  It
bulk-downloads recent daily bars, merges them into each per-symbol parquet, and
recomputes the core technical features used by the signal model.
"""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

FEATURE_ROOT = Path(os.getenv("SIG_LIVE_ROOT", "data/features"))
REPORT_PATH = Path("reports/daily_data_refresh_report.json")
DEFAULT_PERIOD = os.getenv("DAILY_REFRESH_PERIOD", "5d")
DEFAULT_BATCH_SIZE = int(os.getenv("DAILY_REFRESH_BATCH_SIZE", "50"))


def norm_symbol(value: str) -> str:
    return Path(str(value)).stem.replace("_US", "").replace(".US", "").upper()


def yf_symbol(symbol: str) -> str:
    return norm_symbol(symbol).replace(".", "-")


def is_us_symbol(symbol: str) -> bool:
    sym = norm_symbol(symbol)
    return bool(sym) and not sym.endswith((".NS", ".BO", ".NSE", ".BSE"))


def chunked(items: list[str], size: int):
    for start in range(0, len(items), max(1, size)):
        yield items[start : start + max(1, size)]


def load_symbols(root: Path, limit: int) -> list[str]:
    symbols = [norm_symbol(path.name) for path in sorted(root.glob("*.parquet")) if is_us_symbol(path.name)]
    symbols = sorted(dict.fromkeys(symbols))
    return symbols[:limit] if limit and limit > 0 else symbols


def clean_existing(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df is None:
        df = pd.DataFrame()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        df = df.drop(columns=["date"])
    else:
        dates = pd.to_datetime(df.index, errors="coerce")
    df = df.reset_index(drop=True)
    df["date"] = dates.to_numpy()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def download_batch(symbols: list[str], period: str) -> pd.DataFrame:
    yf_symbols = [yf_symbol(sym) for sym in symbols]
    return yf.download(
        yf_symbols,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
        timeout=20,
    )


def extract_symbol_frame(raw: pd.DataFrame, symbol: str, batch_size: int) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    yf_sym = yf_symbol(symbol)
    sub = raw
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(raw.columns.get_level_values(0))
        level1 = list(raw.columns.get_level_values(1))
        if yf_sym in level0:
            sub = raw[yf_sym]
        elif symbol in level0:
            sub = raw[symbol]
        elif yf_sym in level1:
            sub = raw.xs(yf_sym, axis=1, level=1)
        elif symbol in level1:
            sub = raw.xs(symbol, axis=1, level=1)
        else:
            return pd.DataFrame()
    elif batch_size > 1:
        return pd.DataFrame()

    out = pd.DataFrame(index=sub.index)
    col_map = {
        "open": ["Open", "open"],
        "high": ["High", "high"],
        "low": ["Low", "low"],
        "close": ["Close", "Adj Close", "close", "adj_close"],
        "volume": ["Volume", "volume"],
    }
    for target, candidates in col_map.items():
        source = next((col for col in candidates if col in sub.columns), None)
        if source is not None:
            out[target] = pd.to_numeric(sub[source], errors="coerce")
    if "close" not in out.columns:
        return pd.DataFrame()
    out = out.dropna(subset=["close"]).copy()
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out.index, errors="coerce").tz_localize(None).normalize()
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out[["date"] + [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]]


def merge_bars(path: Path, bars: pd.DataFrame, dry_run: bool) -> tuple[bool, str]:
    if bars.empty:
        return False, "no_price_data"
    existing = clean_existing(path)
    if existing.empty:
        merged = bars.copy()
    else:
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.normalize()
        bars = bars.copy()
        bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
        merged = existing.set_index("date", drop=False)
        for _, bar in bars.iterrows():
            dt = bar["date"]
            if pd.isna(dt):
                continue
            if dt in merged.index:
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in bar and pd.notna(bar[col]):
                        merged.loc[dt, col] = bar[col]
            else:
                row = existing.iloc[-1].copy()
                row["date"] = dt
                for col in existing.columns:
                    if col not in {"date", "open", "high", "low", "close", "volume"}:
                        row[col] = math.nan
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in bar and pd.notna(bar[col]):
                        row[col] = bar[col]
                merged.loc[dt] = row
        merged = merged.reset_index(drop=True)
    merged = merged.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if not dry_run:
        merged.to_parquet(path, index=False)
        os.utime(path, None)
    return True, str(pd.to_datetime(merged["date"], errors="coerce").dropna().iloc[-1].date())


def write_report(payload: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(FEATURE_ROOT))
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = time.time()
    root = Path(args.root)
    symbols = load_symbols(root, args.limit)
    total_updated = 0
    total_failed = 0
    failures: list[dict] = []
    latest_dates: dict[str, int] = {}

    print(f"DAILY REFRESH START symbols={len(symbols)} period={args.period} batch_size={args.batch_size}", flush=True)
    for batch_no, batch in enumerate(chunked(symbols, args.batch_size), 1):
        try:
            raw = download_batch(batch, args.period)
        except Exception as exc:
            total_failed += len(batch)
            failures.append({"batch": batch_no, "symbols": batch, "error": str(exc)})
            print(f"REFRESH batch={batch_no} updated=0 failed={len(batch)} error={exc}", flush=True)
            continue

        updated = 0
        failed = 0
        for sym in batch:
            path = root / f"{sym}.parquet"
            if not path.exists():
                path = root / f"{sym}_US.parquet"
            if not path.exists():
                failed += 1
                failures.append({"symbol": sym, "error": "feature_file_missing"})
                continue
            bars = extract_symbol_frame(raw, sym, len(batch))
            ok, reason = merge_bars(path, bars, args.dry_run)
            if ok:
                updated += 1
                latest_dates[reason] = latest_dates.get(reason, 0) + 1
            else:
                failed += 1
                failures.append({"symbol": sym, "error": reason})
        total_updated += updated
        total_failed += failed
        print(f"REFRESH batch={batch_no} updated={updated} failed={failed}", flush=True)

    payload = {
        "ok": True,
        "dry_run": bool(args.dry_run),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "period": args.period,
        "symbols": len(symbols),
        "updated": total_updated,
        "failed": total_failed,
        "latest_dates": latest_dates,
        "duration_seconds": round(time.time() - started, 2),
        "failures": failures[:200],
    }
    write_report(payload)
    print(f"REFRESH DONE updated={total_updated} failed={total_failed} duration={payload['duration_seconds']}s", flush=True)
    return 0 if total_updated > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
