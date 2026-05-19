import argparse, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Load .env if python-dotenv is installed (graceful no-op otherwise)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

MODEL_CANDIDATES = [
    Path("models/checkpoints/fixed_return_h10_model.joblib"),
    Path("models/fixed_return_h10_model.joblib"),
]
LIVE_ROOT = Path(os.getenv("SIG_LIVE_ROOT", "data/features"))
HIST_ROOT = Path(os.getenv("SIG_HIST_ROOT", "data/features_26yr_liquid"))
RUNTIME_STATE = Path(os.getenv("SIG_RUNTIME_STATE", "data/runtime_state.json"))
OUT_JSON = Path(os.getenv("SIG_OUT_JSON", "reports/fixed_return_daily_signals.json"))
OUT_CSV = Path(os.getenv("SIG_OUT_CSV", "reports/fixed_return_daily_signals.csv"))

SIG_THRESHOLD = float(os.getenv("SIG_THRESHOLD", "0.55"))
SIG_TOP_N = int(os.getenv("SIG_TOP_N", "5"))
BASE_POSITION_PCT = float(os.getenv("SIG_POSITION_PCT", "0.0075"))
MIN_PRICE = float(os.getenv("SIG_MIN_PRICE", "5"))
MIN_ADV = float(os.getenv("SIG_MIN_ADV20_DOLLAR_VOL", "5000000"))
ADV_CUT = float(os.getenv("SIG_RUNTIME_UNIVERSE_ADV_CUT", "0.30"))
EXPANDED_CAP = int(os.getenv("SIG_EXPANDED_UNIVERSE_CAP", "400"))
PROFIT_TARGET_PCT = float(os.getenv("SIG_PROFIT_TARGET_PCT", "5.0"))
STOP_LOSS_PCT = float(os.getenv("SIG_STOP_LOSS_PCT", "3.0"))
HOLD_DAYS = int(os.getenv("SIG_HOLD_DAYS", "8"))

# ── Priority 2: Market cap filter — keep only liquid large/mid-caps ───────────
# Default 500M. Set SIG_MIN_MARKET_CAP=0 to disable.
SIG_MIN_MARKET_CAP = float(os.getenv("SIG_MIN_MARKET_CAP", "500000000"))

# ── Priority 1: SPY Market Regime Filter ──────────────────────────────────────
# Set SIG_REGIME_FILTER=1 to enable. When market is stressed, signals = [].
SIG_REGIME_FILTER_ENABLED = os.getenv("SIG_REGIME_FILTER", "0") != "0"
SIG_REGIME_SPY_MA_DAYS    = int(os.getenv("SIG_REGIME_SPY_MA_DAYS", "50"))
SIG_REGIME_RETURN_FLOOR   = float(os.getenv("SIG_REGIME_RETURN_FLOOR", "-5.0"))
SIG_REGIME_RETURN_DAYS    = int(os.getenv("SIG_REGIME_RETURN_DAYS", "20"))
# Soft mode: instead of skipping entirely, halve effective position size
SIG_REGIME_SOFT_HALF      = os.getenv("SIG_REGIME_SOFT_HALF", "0") != "0"
# ─────────────────────────────────────────────────────────────────────────────

# Adaptive threshold: if fewer than SIG_MIN_SIGNALS qualify, step down by 0.01
# until SIG_THRESHOLD_FLOOR is reached. Set SIG_MIN_SIGNALS=0 to disable.
SIG_MIN_SIGNALS = int(os.getenv("SIG_MIN_SIGNALS", "0"))
SIG_THRESHOLD_FLOOR = float(os.getenv("SIG_THRESHOLD_FLOOR", "0.50"))

# Earnings blackout: symbols with earnings within this many calendar days are dropped.
# Default = HOLD_DAYS + 2 (evaluated at runtime). Override with SIG_EARNINGS_BLACKOUT_DAYS.
_EARNINGS_BLACKOUT_OVERRIDE = os.getenv("SIG_EARNINGS_BLACKOUT_DAYS", "")

EXCLUDE_SECTORS = [s.strip() for s in os.getenv("SIG_EXCLUDE_SECTORS", "").split(",") if s.strip()]
EXCLUDE_SECTORS_LOWER = {s.lower() for s in EXCLUDE_SECTORS}
MAX_BIOTECH_PER_DAY = int(os.getenv("SIG_MAX_BIOTECH_PER_DAY", "999"))
SECTOR_METADATA_CANDIDATES = [
    Path("data/symbol_metadata.csv"),
    Path("data/symbol_sectors.csv"),
]
SECTOR_CACHE_PATH = Path("data/sector_cache.json")

# LLM judgment layer (Groq — llama-3.3-70b-versatile)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_FILTER_ENABLED = os.getenv("LLM_FILTER_ENABLED", "1") != "0"
LLM_MODEL = os.getenv("LLM_FILTER_MODEL", "llama-3.3-70b-versatile")
LLM_NEWS_CACHE_PATH = Path("data/llm_news_cache.json")

# Market-cap cache
_MCAP_CACHE_PATH = Path("data/market_cap_cache.json")
_mcap_cache = None
_mcap_cache_dirty = False

_metadata_sectors = None
_sector_cache = None
_sector_cache_dirty = False
_news_cache = None
_news_cache_dirty = False


# ── Market cap helpers ────────────────────────────────────────────────────────

def _load_mcap_cache():
    global _mcap_cache
    if _mcap_cache is not None:
        return _mcap_cache
    if _MCAP_CACHE_PATH.exists():
        try:
            _mcap_cache = json.loads(_MCAP_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _mcap_cache = {}
    else:
        _mcap_cache = {}
    return _mcap_cache


def _save_mcap_cache():
    global _mcap_cache_dirty
    if _mcap_cache is None or not _mcap_cache_dirty:
        return
    try:
        _MCAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MCAP_CACHE_PATH.write_text(
            json.dumps(_mcap_cache, indent=2, sort_keys=True), encoding="utf-8"
        )
        _mcap_cache_dirty = False
    except Exception as e:
        print(f"WARN mcap cache write failed: {e}", flush=True)


def get_market_cap(symbol: str) -> float:
    """Return market cap in USD, or 0 on failure. Cached per symbol."""
    global _mcap_cache_dirty
    sym = str(symbol).upper()
    cache = _load_mcap_cache()
    if sym in cache:
        return float(cache[sym] or 0)
    mcap = 0.0
    try:
        import yfinance as yf
        info = yf.Ticker(sym).info or {}
        mcap = float(info.get("marketCap") or info.get("market_cap") or 0)
    except Exception:
        mcap = 0.0
    cache[sym] = mcap
    _mcap_cache_dirty = True
    return mcap


# ── Sector helpers (unchanged) ────────────────────────────────────────────────

def _load_metadata_sectors():
    global _metadata_sectors
    if _metadata_sectors is not None:
        return _metadata_sectors
    table = {}
    for path in SECTOR_METADATA_CANDIDATES:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"WARN sector metadata read failed {path}: {e}", flush=True)
            continue
        sym_col = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), None)
        sec_col = next((c for c in df.columns if c.lower() == "sector"), None)
        if sec_col is None:
            sec_col = next((c for c in df.columns if c.lower() == "industry"), None)
        if sym_col is None or sec_col is None:
            continue
        for _, r in df[[sym_col, sec_col]].dropna().iterrows():
            table[str(r[sym_col]).upper()] = str(r[sec_col])
        if table:
            print(f"SECTOR metadata loaded {len(table)} symbols from {path}", flush=True)
            break
    _metadata_sectors = table
    return _metadata_sectors


def _load_sector_cache():
    global _sector_cache
    if _sector_cache is not None:
        return _sector_cache
    if SECTOR_CACHE_PATH.exists():
        try:
            _sector_cache = json.loads(SECTOR_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _sector_cache = {}
    else:
        _sector_cache = {}
    return _sector_cache


def _save_sector_cache():
    global _sector_cache_dirty
    if _sector_cache is None or not _sector_cache_dirty:
        return
    try:
        SECTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SECTOR_CACHE_PATH.write_text(
            json.dumps(_sector_cache, indent=2, sort_keys=True), encoding="utf-8"
        )
        _sector_cache_dirty = False
    except Exception as e:
        print(f"WARN sector cache write failed: {e}", flush=True)


def get_sector(symbol):
    global _sector_cache_dirty
    sym = str(symbol).upper()
    meta = _load_metadata_sectors()
    if sym in meta:
        return meta[sym]
    cache = _load_sector_cache()
    if sym in cache:
        return cache[sym]
    sector = ""
    try:
        import yfinance as yf
        info = yf.Ticker(sym).info or {}
        sector = str(info.get("sector") or "")
    except Exception:
        sector = ""
    cache[sym] = sector
    _sector_cache_dirty = True
    return sector


def _is_biotech_like(sector):
    if not sector:
        return False
    s = sector.lower()
    return (
        "biotech" in s
        or "pharma" in s
        or "drug manuf" in s
        or "health tech" in s
        or s == "healthcare"
    )


def select_top_n_with_biotech_cap(sorted_df, n, max_biotech):
    if max_biotech >= n or sorted_df.empty:
        return sorted_df.head(n)
    selected, biotech_count = [], 0
    for _, row in sorted_df.iterrows():
        if len(selected) >= n:
            break
        sec = get_sector(row["symbol"])
        if _is_biotech_like(sec):
            if biotech_count >= max_biotech:
                continue
            biotech_count += 1
        selected.append(row)
    if not selected:
        return sorted_df.iloc[0:0]
    return pd.DataFrame(selected).reset_index(drop=True)


# ────────────────────────────────── LLM judgment layer ──────────────────────────────────

def _load_news_cache():
    global _news_cache
    if _news_cache is not None:
        return _news_cache
    if LLM_NEWS_CACHE_PATH.exists():
        try:
            _news_cache = json.loads(LLM_NEWS_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _news_cache = {}
    else:
        _news_cache = {}
    return _news_cache


def _save_news_cache():
    global _news_cache_dirty
    if _news_cache is None or not _news_cache_dirty:
        return
    try:
        LLM_NEWS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LLM_NEWS_CACHE_PATH.write_text(
            json.dumps(_news_cache, indent=2), encoding="utf-8"
        )
        _news_cache_dirty = False
    except Exception as e:
        print(f"WARN llm news cache write failed: {e}", flush=True)


def _extract_news_title(item):
    if not isinstance(item, dict):
        return ""
    title = item.get("title")
    if not title:
        content = item.get("content")
        if isinstance(content, dict):
            title = content.get("title") or content.get("description")
    return str(title or "").strip()


def fetch_news(symbol, date_iso):
    """Up to 3 recent news titles for a symbol; cached as {symbol}_{date}."""
    global _news_cache_dirty
    sym = str(symbol).upper()
    cache = _load_news_cache()
    key = f"{sym}_{date_iso}"
    if key in cache:
        return list(cache[key])
    headlines = []
    try:
        import yfinance as yf
        news = yf.Ticker(sym).news or []
        for item in news:
            title = _extract_news_title(item)
            if title:
                headlines.append(title)
            if len(headlines) >= 3:
                break
    except Exception:
        pass
    cache[key] = headlines
    _news_cache_dirty = True
    return headlines


def market_context_pct():
    """Return (spy_pct_today, qqq_pct_today) — today's % change vs prev close."""
    out = {"SPY": None, "QQQ": None}
    try:
        import yfinance as yf
        for sym in ("SPY", "QQQ"):
            try:
                hist = yf.Ticker(sym).history(period="5d")
                if hist is None or len(hist) < 2:
                    continue
                prev = float(hist["Close"].iloc[-2])
                last = float(hist["Close"].iloc[-1])
                if prev > 0:
                    out[sym] = round((last - prev) / prev * 100.0, 3)
            except Exception:
                pass
    except Exception:
        pass
    return out["SPY"], out["QQQ"]


LLM_SYSTEM_PROMPT = """You are a senior equity-trading judgment layer reviewing machine-generated long-trade candidates.

Trade parameters:
- 10-day long momentum/mean-reversion trades
- Entry at next morning's open
- Profit target 8%, no stop loss

For each candidate, decide PROCEED, REDUCE_HALF, or SKIP.

SKIP rules:
- Post-earnings collapse (stock dropped >10% on earnings)
- Clearly broken downtrend with no catalyst for reversal
- Overbought after a multi-day rally into resistance
- Fundamental company collapse (fraud, going-concern, mass-exec departures)
- Biotech/pharma with negative FDA/trial news

REDUCE_HALF rules:
- Moderately elevated risk but setup is still valid
- Broad market risk-off day (SPY < -1%)
- Notable sector headwinds

PROCEED rules:
- Pullback in established uptrend
- Oversold with a catalyst
- Broad market neutral or positive
- When in doubt, PROCEED. Lean toward PROCEED in ambiguity. Be extra cautious on biotech/pharma.

Output rules: return ONLY valid JSON. No prose, no markdown, no code fences. Schema:
{
  "SYMBOL": {"decision": "proceed|skip|reduce_half", "reason": "<10 words max>"},
  ...
}
One entry per input symbol. Include every symbol passed in."""


def _build_llm_user_message(candidates, spy_pct, qqq_pct, num_signals):
    spy_str = f"{spy_pct:+.2f}%" if spy_pct is not None else "unknown"
    qqq_str = f"{qqq_pct:+.2f}%" if qqq_pct is not None else "unknown"
    lines = [
        f"Market today: SPY {spy_str}, QQQ {qqq_str}.",
        f"{num_signals} candidates above threshold.",
        "",
        "Candidates:",
    ]
    for c in candidates:
        sec = c.get("sector") or ""
        sec_part = f"sector={sec}" if sec else "sector=unknown"
        pct = c.get("pct_today")
        pct_str = f"{pct:+.2f}%" if isinstance(pct, (int, float)) and pct is not None else "unknown"
        lines.append(
            f"- {c['symbol']} prob={c['probability']:.4f} today={pct_str} {sec_part}"
        )
        for h in c.get("headlines", []):
            lines.append(f"    NEWS: {h}")
    lines.append("")
    lines.append("Return ONLY JSON. One entry per symbol.")
    return "\n".join(lines)


def call_llm_filter(candidates, spy_pct, qqq_pct, num_signals):
    """Returns {SYMBOL: {decision, reason}} or {} on any failure (Groq via requests)."""
    if not GROQ_API_KEY:
        print("LLM FILTER: GROQ_API_KEY not set; skipping LLM step", flush=True)
        return {}
    try:
        import requests as _req
    except ImportError:
        print("LLM FILTER: requests not installed; skipping", flush=True)
        return {}
    try:
        user_msg = _build_llm_user_message(candidates, spy_pct, qqq_pct, num_signals)
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        resp = _req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
        # Strip markdown fences defensively.
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
        text = text.strip().strip("`").strip()
        decisions = json.loads(text)
        if not isinstance(decisions, dict):
            raise ValueError(f"LLM response not a dict: {type(decisions).__name__}")
        return decisions
    except Exception as e:
        print(f"LLM FILTER: call failed ({e}); proceeding with all signals", flush=True)
        return {}


def _apply_llm_decisions(signals, decisions):
    """Filter/halve signals based on LLM decisions. Returns a new list."""
    out = []
    for s in signals:
        sym = s["symbol"]
        d = decisions.get(sym) or decisions.get(sym.upper()) or {}
        decision = str(d.get("decision", "proceed")).lower().strip()
        reason = str(d.get("reason", "")).strip()[:120]
        if decision == "skip":
            print(f"LLM {sym}: skip — {reason}", flush=True)
            continue
        if decision == "reduce_half":
            s["position_pct"] = round(s["position_pct"] / 2.0, 6)
            s["llm_decision"] = "reduce_half"
            s["llm_reason"] = reason
            print(f"LLM {sym}: reduce_half — {reason}", flush=True)
            out.append(s)
            continue
        s["llm_decision"] = "proceed"
        s["llm_reason"] = reason
        print(f"LLM {sym}: proceed — {reason}", flush=True)
        out.append(s)
    for i, s in enumerate(out, 1):
        s["rank"] = i
    return out


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)

def norm_sym(x):
    s = Path(str(x)).stem if str(x).endswith(".parquet") else str(x)
    return s.replace("_US", "").replace(".US", "").upper()

def read_feature_file(path, symbol=True):
    df = pd.read_parquet(path)
    if df.empty or "close" not in df.columns:
        return None
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = pd.RangeIndex(len(df))
    if "date" in df.columns:
        dc = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = dc if dc.notna().sum() >= idx.notna().sum() else idx
    else:
        df["date"] = idx
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return None
    if "volume" in df.columns and "close" in df.columns:
        df["adv20_dollar_vol"] = (df["close"] * df["volume"]).rolling(20, min_periods=1).mean()
    if symbol:
        df["symbol"] = norm_sym(path.name)
    return df

def load_model():
    path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if path is None:
        die("fixed return model not found")
    obj = joblib.load(path)
    model = obj.get("model") if isinstance(obj, dict) else obj
    features = (obj.get("features") or obj.get("feature_cols")) if isinstance(obj, dict) else getattr(model, "feature_names_in_", None)
    if model is None or features is None:
        die(f"model file {path} missing model/features")
    return path, model, list(features)

def allowed_symbols():
    base = set()
    if RUNTIME_STATE.exists():
        try:
            j = json.load(open(RUNTIME_STATE))
            base = {str(k).upper() for k in (j.get("signal_store") or {}).keys()}
        except Exception:
            base = set()
    eff_min_adv = MIN_ADV * (1.0 - ADV_CUT)
    rows = []
    for p in sorted(HIST_ROOT.glob("*.parquet")):
        sym = norm_sym(p.name)
        if sym.endswith((".NS", ".BO", ".NSE", ".BSE")):
            continue
        try:
            df = read_feature_file(p)
            if df is None:
                continue
            r = df.iloc[-1]
            adv = float(r.get("adv20_dollar_vol", 0) or 0)
            px = float(r.get("close", 0) or 0)
            if adv >= eff_min_adv and px >= MIN_PRICE:
                rows.append((adv, sym))
        except Exception:
            continue
    expanded = {s for _, s in sorted(rows, reverse=True)[:EXPANDED_CAP]}
    return (base | expanded) if base else expanded

def spy_vol():
    for p in [LIVE_ROOT/"SPY.parquet", LIVE_ROOT/"SPY_US.parquet", LIVE_ROOT/"SPY.US.parquet"]:
        if not p.exists():
            continue
        df = read_feature_file(p, symbol=False)
        if df is None or len(df) < 21:
            continue
        vol = float(df["close"].pct_change().rolling(20, min_periods=10).std().iloc[-1] * np.sqrt(252) * 100.0)
        if math.isfinite(vol):
            if vol < 15: return vol, "low", 0.75
            if vol < 25: return vol, "medium", 1.0
            if vol < 40: return vol, "high", 1.25
            return vol, "extreme", 1.5
    return 20.0, "medium", 1.0


# ── SPY Market Regime Gate ────────────────────────────────────────────────────

def spy_regime():
    """
    Returns (is_stressed: bool, reason: str).

    stressed = True  → skip new entries (or halve in soft mode)
    stressed = False → market is healthy, proceed normally

    Logic:
      stressed_sma    = SPY close < SMA(SIG_REGIME_SPY_MA_DAYS)
      stressed_return = SPY N-day return < SIG_REGIME_RETURN_FLOOR%
      stressed        = stressed_sma OR stressed_return

    Falls back to False (not stressed) if SPY data is unavailable — fail open,
    not fail closed, so a data hiccup doesn't silently kill entries.
    """
    for p in [LIVE_ROOT / "SPY.parquet", LIVE_ROOT / "SPY_US.parquet", LIVE_ROOT / "SPY.US.parquet"]:
        if not p.exists():
            continue
        df = read_feature_file(p, symbol=False)
        if df is None or len(df) < max(SIG_REGIME_SPY_MA_DAYS, SIG_REGIME_RETURN_DAYS) + 5:
            continue

        prices = df["close"]
        spy_last = float(prices.iloc[-1])

        # SMA gate
        sma_val = float(
            prices.rolling(SIG_REGIME_SPY_MA_DAYS, min_periods=max(SIG_REGIME_SPY_MA_DAYS // 2, 10))
            .mean()
            .iloc[-1]
        )
        stressed_sma = math.isfinite(sma_val) and (spy_last < sma_val)

        # N-day return gate
        ret_nd = None
        if len(prices) >= SIG_REGIME_RETURN_DAYS + 1:
            prev_px = float(prices.iloc[-(SIG_REGIME_RETURN_DAYS + 1)])
            if prev_px > 0:
                ret_nd = (spy_last - prev_px) / prev_px * 100.0
        stressed_return = (ret_nd is not None) and (ret_nd < SIG_REGIME_RETURN_FLOOR)

        # Build reason string
        reasons = []
        if stressed_sma:
            reasons.append(
                f"SPY({spy_last:.2f}) < SMA{SIG_REGIME_SPY_MA_DAYS}({sma_val:.2f})"
            )
        if stressed_return:
            reasons.append(
                f"SPY {SIG_REGIME_RETURN_DAYS}d-return={ret_nd:.1f}% < floor={SIG_REGIME_RETURN_FLOOR}%"
            )

        if stressed_sma or stressed_return:
            return True, " | ".join(reasons)
        else:
            sma_gap = (spy_last - sma_val) / sma_val * 100.0 if math.isfinite(sma_val) else 0.0
            ret_str = f" | {SIG_REGIME_RETURN_DAYS}d-ret={ret_nd:.1f}%" if ret_nd is not None else ""
            return False, f"SPY({spy_last:.2f}) +{sma_gap:.1f}% above SMA{SIG_REGIME_SPY_MA_DAYS}{ret_str}"

    # SPY data not found — fail open
    return False, "SPY parquet not found — regime gate bypassed (fail-open)"

# ─────────────────────────────────────────────────────────────────────────────


def load_file(p):
    df = pd.read_parquet(p)
    if df is None or df.empty:
        return None
    if "date" in df.columns:
        date_values = pd.to_datetime(df["date"], errors="coerce")
    else:
        date_values = pd.to_datetime(df.index, errors="coerce")
    df = df.reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    df["date"] = date_values.to_numpy()
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    df["symbol"] = p.stem.replace("_US","").replace(".US","")
    if "adv20_dollar_vol" not in df.columns and {"close","volume"}.issubset(df.columns):
        df["adv20_dollar_vol"] = (df["close"] * df["volume"]).rolling(20, min_periods=5).mean()
    return df

def file_mtime_et_date(path):
    return datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo("America/New_York")).date()

def latest_rows(features, allowed):
    rows = []
    fresh_count = 0
    stale_count = 0
    sector_filtered = 0
    mcap_filtered = 0
    today_et = datetime.now(ZoneInfo("America/New_York")).date()

    for p in sorted(LIVE_ROOT.glob("*.parquet")):
        try:
            x = load_file(p)
        except Exception as e:
            print(f"SKIP {p.stem} load_error={e}", flush=True)
            continue
        if x is None or x.empty:
            continue
        sym = str(x["symbol"].iloc[-1] if "symbol" in x.columns and len(x) else p.stem.replace("_US","").replace(".US",""))
        file_date = file_mtime_et_date(p)
        if file_date != today_et:
            stale_count += 1
            print(f"SKIP {sym} stale data last={file_date}", flush=True)
            continue
        fresh_count += 1
        r = x.iloc[-1].copy()
        price = float(r.get("close", 0) or 0)
        adv = float(r.get("adv20_dollar_vol", 0) or 0)
        # Today's percent change vs previous close (for LLM context).
        pct_today = float("nan")
        if len(x) >= 2:
            try:
                prev_close = float(x["close"].iloc[-2])
                if prev_close > 0:
                    pct_today = (price - prev_close) / prev_close * 100.0
            except Exception:
                pct_today = float("nan")
        try:
            if not is_allowed_symbol(sym):
                continue
        except NameError:
            pass
        if price < MIN_PRICE or adv < MIN_ADV:
            continue
        if EXCLUDE_SECTORS_LOWER:
            sec = get_sector(sym)
            if sec and sec.lower() in EXCLUDE_SECTORS_LOWER:
                sector_filtered += 1
                continue
        row = {c: float(r.get(c, 0) or 0) for c in features}
        row.update({
            "symbol": sym,
            "price": price,
            "adv20_dollar_vol": adv,
            "pct_today": pct_today,
        })
        rows.append(row)

    print(f"FRESHNESS: {fresh_count} fresh, {stale_count} stale, scoring fresh only", flush=True)
    if EXCLUDE_SECTORS:
        print(f"SECTOR FILTER: excluded={sector_filtered} sectors={EXCLUDE_SECTORS}", flush=True)
    _save_sector_cache()
    if fresh_count < 50:
        print(f"WARN only {fresh_count} fresh symbols available", flush=True)
    if not rows:
        raise SystemExit("no fresh feature rows available; refusing to score stale data")
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    model_path, model, features = load_model()
    allowed = allowed_symbols()
    vol, regime, mult = spy_vol()

    # ── SPY Market Regime Gate ────────────────────────────────────────────────
    # Checked BEFORE loading live data (cheap fast-fail).
    regime_blocked = False
    regime_reason  = ""
    if SIG_REGIME_FILTER_ENABLED:
        regime_stressed, regime_reason = spy_regime()
        if regime_stressed:
            if SIG_REGIME_SOFT_HALF:
                # Soft mode: halve position, still enter
                mult = mult * 0.5
                print(
                    f"REGIME GATE (soft): market stressed — {regime_reason} — "
                    f"halving position multiplier to {mult:.2f}",
                    flush=True,
                )
            else:
                # Hard mode: skip all new entries today
                regime_blocked = True
                print(
                    f"REGIME GATE: market stressed — {regime_reason} — "
                    f"skipping all new entries today",
                    flush=True,
                )
        else:
            print(f"REGIME GATE: market OK — {regime_reason}", flush=True)
    # ─────────────────────────────────────────────────────────────────────────

    signal_date = datetime.now().date().isoformat()

    if regime_blocked:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "signal_date": signal_date,
            "model_path": str(model_path),
            "vol_regime": regime,
            "vol_multiplier": mult,
            "spy_realized_vol": round(vol, 4),
            "regime_blocked": True,
            "regime_reason": regime_reason,
            "signals": [],
        }
        print(f"SIGNAL DATE {signal_date}")
        print(f"REGIME BLOCKED: 0 signals emitted")
        if not args.dry_run:
            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            pd.DataFrame([]).to_csv(OUT_CSV, index=False)
            print(f"SIGNALS WRITTEN TO {OUT_JSON} (regime_blocked=True)")
        return

    live = latest_rows(features, allowed)

    X = live[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else proba.shape[1]-1
    live["probability"] = proba[:, pos_idx]

    # ── Adaptive threshold ────────────────────────────────────────────
    thresh = SIG_THRESHOLD
    qualified = live[live["probability"] >= thresh].sort_values("probability", ascending=False)
    if SIG_MIN_SIGNALS > 0 and len(qualified) < SIG_MIN_SIGNALS:
        while thresh > SIG_THRESHOLD_FLOOR and len(qualified) < SIG_MIN_SIGNALS:
            thresh = round(thresh - 0.01, 4)
            qualified = live[live["probability"] >= thresh].sort_values("probability", ascending=False)
        if thresh < SIG_THRESHOLD:
            print(
                f"ADAPTIVE THRESHOLD: stepped {SIG_THRESHOLD:.2f}→{thresh:.2f}"
                f" to find {len(qualified)} candidates (min={SIG_MIN_SIGNALS})",
                flush=True,
            )
    pre_cap = len(qualified)
    print(f"QUALIFIED: {pre_cap} symbols above threshold={thresh:.3f}", flush=True)
    # ─────────────────────────────────────────────────────────────────

    picks = select_top_n_with_biotech_cap(qualified, SIG_TOP_N, MAX_BIOTECH_PER_DAY)
    print(f"TOP_N CAP: selected={len(picks)} from qualified={pre_cap}", flush=True)
    if MAX_BIOTECH_PER_DAY < SIG_TOP_N:
        print(f"BIOTECH CAP: max_per_day={MAX_BIOTECH_PER_DAY}", flush=True)

    # ── Market cap filter (Priority 2) ──────────────────────────────────────
    # Drops micro/small-caps the LLM can't evaluate (no options chain, poor liquidity).
    if SIG_MIN_MARKET_CAP > 0 and len(picks) > 0:
        pre_mcap = len(picks)
        cap_eligible = []
        for sym in picks["symbol"].tolist():
            mcap = get_market_cap(sym)
            if mcap >= SIG_MIN_MARKET_CAP or mcap == 0:
                # 0 = lookup failed, fail-open (don't block on data unavailability)
                cap_eligible.append(sym)
            else:
                print(
                    f"MCAP FILTER: dropped {sym} mcap=${mcap/1e9:.2f}B < "
                    f"${SIG_MIN_MARKET_CAP/1e9:.1f}B floor",
                    flush=True,
                )
        _save_mcap_cache()
        picks = picks[picks["symbol"].isin(cap_eligible)]
        dropped = pre_mcap - len(picks)
        if dropped:
            print(f"MCAP FILTER: removed={dropped} remaining={len(picks)}", flush=True)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Earnings / event risk filter ─────────────────────────────────
    pre = len(picks)
    earnings_cols = {
        "official_event_hit":        1,
        "filing_event_hit":          1,
        "event_day_extreme":         1,
    }
    for col, threshold in earnings_cols.items():
        if col in picks.columns:
            picks = picks[picks[col].fillna(0) < threshold]
    if "peer_earnings_shock_3d" in picks.columns:
        shock_cap = float(os.getenv("SIG_PEER_SHOCK_CAP", "2.0"))
        picks = picks[picks["peer_earnings_shock_3d"].fillna(0).abs() < shock_cap]
    print(f"EARNINGS FILTER (feature cols) removed={pre - len(picks)} remaining={len(picks)}", flush=True)
    try:
        import yfinance as yf
        from datetime import date, timedelta
        blackout_days = int(_EARNINGS_BLACKOUT_OVERRIDE) if _EARNINGS_BLACKOUT_OVERRIDE else HOLD_DAYS + 2
        hold_end = date.today() + timedelta(days=blackout_days)
        safe_symbols = []
        for sym in picks["symbol"].tolist():
            try:
                cal = yf.Ticker(sym).calendar
                if not cal:
                    safe_symbols.append(sym)
                    continue
                earn_dates = cal.get("Earnings Date", [])
                if not isinstance(earn_dates, list):
                    earn_dates = [earn_dates]
                too_close = any(
                    date.today() <= (d.date() if hasattr(d, "date") else d) <= hold_end
                    for d in earn_dates if d is not None
                )
                if too_close:
                    print(f"EARNINGS UPCOMING filter={sym} date={earn_dates[0]}")
                else:
                    safe_symbols.append(sym)
            except Exception:
                safe_symbols.append(sym)
        picks = picks[picks["symbol"].isin(safe_symbols)]
    except Exception as e:
        print(f"EARNINGS FORWARD CHECK error={e}")
    picks = picks.head(SIG_TOP_N)
    # ─────────────────────────────────────────────────────────────────
    _save_sector_cache()
    expected_exit_date = (pd.Timestamp(signal_date) + BDay(HOLD_DAYS)).date().isoformat()
    eff_pos = BASE_POSITION_PCT * mult
    signals = []
    for rank, (_, r) in enumerate(picks.iterrows(), 1):
        entry = float(r.get("entry_price", r.get("price", r.get("close", 0))) or 0)
        signals.append({
            "rank": rank, "symbol": str(r.symbol), "probability": round(float(r.probability), 6),
            "entry_price": round(entry, 4), "position_pct": round(eff_pos, 6),
            "base_position_pct": BASE_POSITION_PCT, "vol_multiplier": mult,
            "profit_target_price": round(entry * (1 + PROFIT_TARGET_PCT / 100), 4),
            "stop_loss_price": round(entry * (1 - STOP_LOSS_PCT / 100), 4),
            "expected_exit_date": expected_exit_date, "hold_days": HOLD_DAYS,
            "feature_date": str(r.get("feature_date", r.get("date", "unknown"))),
        })

    # ── LLM judgment layer (Groq llama-3.3-70b-versatile) ───────────
    if LLM_FILTER_ENABLED and signals:
        try:
            spy_pct, qqq_pct = market_context_pct()
            candidates = []
            for s in signals:
                sym = s["symbol"]
                pct_today = None
                row_match = live[live["symbol"] == sym]
                if len(row_match) > 0 and "pct_today" in row_match.columns:
                    val = row_match["pct_today"].iloc[0]
                    if pd.notna(val):
                        pct_today = float(val)
                sector = get_sector(sym) or ""
                headlines = fetch_news(sym, signal_date)
                candidates.append({
                    "symbol": sym,
                    "probability": s["probability"],
                    "pct_today": pct_today,
                    "sector": sector,
                    "headlines": headlines,
                })
            _save_news_cache()
            _save_sector_cache()
            print(f"LLM FILTER: scoring {len(candidates)} candidates with {LLM_MODEL}", flush=True)
            decisions = call_llm_filter(candidates, spy_pct, qqq_pct, len(candidates))
            if decisions:
                signals = _apply_llm_decisions(signals, decisions)
                print(f"LLM FILTER: kept {len(signals)} signals after judgment", flush=True)
            else:
                print("LLM FILTER: no decisions returned; keeping all signals", flush=True)
        except Exception as e:
            print(f"LLM FILTER: unexpected error ({e}); keeping all signals", flush=True)
    elif not LLM_FILTER_ENABLED:
        print("LLM FILTER: disabled via LLM_FILTER_ENABLED=0", flush=True)
    # ─────────────────────────────────────────────────────────────────

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signal_date": signal_date,
        "model_path": str(model_path),
        "vol_regime": regime,
        "vol_multiplier": mult,
        "spy_realized_vol": round(vol, 4),
        "base_position_pct": BASE_POSITION_PCT,
        "effective_position_pct": round(eff_pos, 6),
        "threshold": SIG_THRESHOLD,
        "scored_universe_count": int(len(live)),
        "allowed_universe_count": int(len(allowed)),
        "profit_target_pct": PROFIT_TARGET_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "hold_days": HOLD_DAYS,
        "regime_blocked": False,
        "regime_reason": regime_reason,
        "signals": signals,
    }
    print(f"SIGNAL DATE {signal_date}")
    print(f"MODEL {model_path}")
    print(f"UNIVERSE scored={len(live)} allowed={len(allowed)}")
    print(f"VOL {regime} spy_realized_vol={vol:.2f}% multiplier={mult}")
    for s in signals:
        print(f"{s['rank']} {s['symbol']} prob={s['probability']:.4f} entry={s['entry_price']:.2f} PT={s['profit_target_price']:.2f} SL={s['stop_loss_price']:.2f} pos={s['position_pct']:.4f}")
    if args.dry_run:
        print("DRY RUN: no files written")
        return
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(signals).to_csv(OUT_CSV, index=False)
    print(f"SIGNALS WRITTEN TO {OUT_JSON}")

if __name__ == "__main__":
    main()
