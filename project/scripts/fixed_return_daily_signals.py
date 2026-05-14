import time
import argparse, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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
EXPANDED_CAP = int(os.getenv("SIG_EXPANDED_UNIVERSE_CAP", "4000"))
PROFIT_TARGET_PCT = float(os.getenv("SIG_PROFIT_TARGET_PCT", "10.0"))
STOP_LOSS_PCT = float(os.getenv("SIG_STOP_LOSS_PCT", "3.0"))
HOLD_DAYS = int(os.getenv("SIG_HOLD_DAYS", "8"))

EXCLUDE_SECTORS = [s.strip() for s in os.getenv("SIG_EXCLUDE_SECTORS", "").split(",") if s.strip()]
EXCLUDE_SECTORS_LOWER = {s.lower() for s in EXCLUDE_SECTORS}
MAX_BIOTECH_PER_DAY = int(os.getenv("SIG_MAX_BIOTECH_PER_DAY", "999"))
SECTOR_METADATA_CANDIDATES = [
    Path("data/symbol_metadata.csv"),
    Path("data/symbol_sectors.csv"),
]
SECTOR_CACHE_PATH = Path("data/sector_cache.json")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_FILTER_ENABLED = os.getenv("LLM_FILTER_ENABLED", "1") != "0"
LLM_MODEL = os.getenv("LLM_FILTER_MODEL", "llama-3.3-70b-versatile")
LLM_NEWS_CACHE_PATH = Path("data/llm_news_cache.json")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

_metadata_sectors = None
_sector_cache = None
_sector_cache_dirty = False
_news_cache = None
_news_cache_dirty = False


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
            if len(headlines) >= 10:
                break
    except Exception:
        pass
    cache[key] = headlines
    _news_cache_dirty = True
    return headlines


def market_context_pct():
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


LLM_SYSTEM_PROMPT = """
════════════════════════════════════════════════════════
NULL HYPOTHESIS: DEFAULT = REDUCE_HALF
Burden of proof is on PROCEED. Clear positive case = PROCEED.
Ambiguity = REDUCE_HALF. Only hard evidence of harm = SKIP.
════════════════════════════════════════════════════════

You are a skeptical senior equity-trading risk officer.
Trade: 10-day long momentum/mean-reversion, entry at next open,
profit target 8%, stop loss 6%.

PHASE 1 - PRE-MORTEM
Before calling any tool, assume this trade hits its stop loss on
Day 3. Write 2-3 bullets explaining WHY it failed using only
information already visible (symbol, today move, sector, headlines).
Label: PRE-MORTEM: <bullets>

PHASE 2 - TOOL INVESTIGATION (ReAct)
You MUST call all 6 tools exactly once, in any order.
After EACH tool returns, write one THOUGHT sentence before calling next.
Format: THOUGHT: <one sentence interpreting data just returned>
UNKNOWN PROTOCOL: empty/null tool result = UNKNOWN, never assume clean.

PHASE 3 - DETERMINISTIC CHECKLIST
After all 6 tools, answer each gate TRUE/FALSE:
  [ ] G1 Short interest < 25% of float
  [ ] G2 Options IV < 80% (no options chain = REDUCE_HALF not SKIP)
  [ ] G3 10-day price trend not in confirmed downtrend
  [ ] G4 Sector ETF 5-day return > -2%
  [ ] G5 No hard-stop news (FDA rejection, fraud, going-concern,
         earnings collapse >10%, mass exec departure)
  [ ] G6 RSI-14 not > 75
If ANY gate FALSE or UNKNOWN -> SKIP or REDUCE_HALF.

PHASE 4 - STEELMAN + REBUTTAL
Write strongest argument FOR skipping. Then rebut it.
If cannot rebut -> SKIP.
  STEELMAN_SKIP: <best skip argument>
  REBUTTAL: <counter, or "Cannot rebut.">

PHASE 5 - FINAL DECISION
SKIP       -> G5 hard-stop news OR G1>40% OR G2>100% OR cannot rebut steelman
REDUCE_HALF -> G1 25-40% OR G2 60-100% OR unknown options chain OR G6 RSI 70-80
REDUCE_HALF -> G1 borderline (25-40%) OR G2 borderline (60-80%) OR
               no options chain OR sector -1% to -2%
PROCEED    -> all 6 gates TRUE, rebuttal holds, pre-mortem priced in

INVALID reasons for PROCEED:
  - "News is neutral"
  - "Probability score is high"
  - "No red flags found"
  - "Market is broadly positive" (alone)

EMPTY DATA: empty tool result = UNKNOWN not CLEAN.
Three or more UNKNOWN -> auto REDUCE_HALF (not SKIP — unknown != bad).

OUTPUT: return ONLY valid JSON, no prose, no fences:
{
  "SYMBOL": {
    "decision": "proceed|skip|reduce_half",
    "reason": "<15 words citing decisive gate or factor>",
    "scratchpad": "<pre-mortem + key THOUGHTs + checklist + steelman>"
  }
}
One entry per input symbol. Include every symbol passed in.
"""


TOOLS = [
    {"type":"function","function":{"name":"check_short_interest","description":"Return short interest % of float and short ratio. Gate G1: <25% required for PROCEED.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
    {"type":"function","function":{"name":"check_options_iv","description":"Return 30-day ATM implied volatility (%) and put/call ratio. Gate G2: IV<80% required.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
    {"type":"function","function":{"name":"check_price_momentum","description":"Return 10-day trend (up/down/flat), RSI-14, pct from 52w high/low. Gates G3 and G6.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
    {"type":"function","function":{"name":"check_sector_performance","description":"Return 5-day return of sector ETF. Gate G4: >-2% required.","parameters":{"type":"object","properties":{"symbol":{"type":"string"},"sector":{"type":"string"}},"required":["symbol","sector"]}}},
    {"type":"function","function":{"name":"analyze_news_risk","description":"Classify headlines: HARD_STOP / ELEVATED / NEUTRAL / POSITIVE_CATALYST. Gate G5: no HARD_STOP.","parameters":{"type":"object","properties":{"symbol":{"type":"string"},"headlines":{"type":"array","items":{"type":"string"}}},"required":["symbol","headlines"]}}},
    {"type":"function","function":{"name":"assess_trade_setup","description":"Holistic setup score 0-100. <40 lean SKIP, 40-65 REDUCE_HALF, >65 PROCEED.","parameters":{"type":"object","properties":{"symbol":{"type":"string"},"probability":{"type":"number"},"pct_today":{"type":"number"},"spy_pct":{"type":"number"},"qqq_pct":{"type":"number"},"sector":{"type":"string"}},"required":["symbol","probability"]}}}
]


def _handle_tool_call(name, args):
    import json as _j
    sym = args.get("symbol","").upper()
    try:
        import yfinance as _yf
    except ImportError:
        _yf = None

    if name == "check_short_interest":
        r = {"symbol":sym,"short_pct_of_float":None,"short_ratio":None,"status":"unknown"}
        if _yf:
            try:
                info = _yf.Ticker(sym).info
                spf = info.get("shortPercentOfFloat")
                sr  = info.get("shortRatio")
                if spf: r["short_pct_of_float"] = round(float(spf)*100,2)
                if sr:  r["short_ratio"] = round(float(sr),2)
                r["status"] = "ok" if spf else "no_data"
            except Exception as e: r["status"]=f"error:{e}"
        return _j.dumps(r)

    elif name == "check_options_iv":
        r = {"symbol":sym,"iv_pct":None,"put_call_ratio":None,"status":"unknown"}
        if _yf:
            try:
                tk = _yf.Ticker(sym); exp = tk.options
                if exp:
                    ch = tk.option_chain(exp[0]); spot = tk.info.get("regularMarketPrice")
                    if spot and not ch.calls.empty:
                        atm = ch.calls.iloc[(ch.calls["strike"]-spot).abs().argsort()[:1]]
                        r["iv_pct"] = round(float(atm["impliedVolatility"].iloc[0])*100,1)
                    if not ch.calls.empty and not ch.puts.empty:
                        r["put_call_ratio"] = round(ch.puts["openInterest"].sum()/max(ch.calls["openInterest"].sum(),1),2)
                    r["status"] = "ok" if r["iv_pct"] else "no_data"
                else: r["status"]="no_chain"
            except Exception as e: r["status"]=f"error:{e}"
        return _j.dumps(r)

    elif name == "check_price_momentum":
        r = {"symbol":sym,"trend_10d":"unknown","rsi_14":None,"pct_from_52w_high":None,"pct_from_52w_low":None,"status":"unknown"}
        if _yf:
            try:
                hist = _yf.Ticker(sym).history(period="60d")
                c = hist["Close"] if "Close" in hist.columns else hist["close"]
                if len(c)>=10:
                    sl = (c.iloc[-1]-c.iloc[-10])/c.iloc[-10]*100
                    r["trend_10d"] = "up" if sl>1 else ("down" if sl<-1 else "flat")
                    d=c.diff(); g=d.clip(lower=0).ewm(com=13,adjust=False).mean(); l=(-d.clip(upper=0)).ewm(com=13,adjust=False).mean()
                    rs=g.iloc[-1]/max(float(l.iloc[-1]),1e-9); r["rsi_14"]=round(float(100-100/(1+rs)),1)
                    r["pct_from_52w_high"]=round((c.iloc[-1]-c.max())/c.max()*100,1)
                    r["pct_from_52w_low"]=round((c.iloc[-1]-c.min())/c.min()*100,1)
                    r["status"]="ok"
                else: r["status"]="insufficient_data"
            except Exception as e: r["status"]=f"error:{e}"
        return _j.dumps(r)

    elif name == "check_sector_performance":
        ETF={"technology":"XLK","information technology":"XLK","healthcare":"XLV","health care":"XLV","financials":"XLF","financial services":"XLF","consumer discretionary":"XLY","consumer staples":"XLP","energy":"XLE","industrials":"XLI","materials":"XLB","real estate":"XLRE","utilities":"XLU","communication services":"XLC","communication":"XLC"}
        sec=str(args.get("sector","")).lower(); etf=next((v for k,v in ETF.items() if k in sec),"SPY")
        r={"symbol":sym,"sector_etf":etf,"five_day_return_pct":None,"status":"unknown"}
        if _yf:
            try:
                h=_yf.Ticker(etf).history(period="10d"); c=h["Close"] if "Close" in h.columns else h["close"]
                if len(c)>=5: r["five_day_return_pct"]=round(float((c.iloc[-1]-c.iloc[-5])/c.iloc[-5]*100),2); r["status"]="ok"
                else: r["status"]="insufficient_data"
            except Exception as e: r["status"]=f"error:{e}"
        return _j.dumps(r)

    elif name == "analyze_news_risk":
        import re as _re
        hs=args.get("headlines",[]); hsc=0; cl=[]
        HARD=[r"fda.{0,20}(reject|refuse|decline|hold)",r"(fraud|going.concern|bankrupt|chapter.1[12]|delist)",r"(ceo|cfo|coo).{0,30}(resign|depart|fired|step.down)",r"(sec|doj).{0,20}(invest|charge|indict)",r"earnings.{0,30}(miss|drop|plunge).{0,20}[1-9]\d%"]
        ELEV=[r"(downgrade|cut).{0,20}(price.target|rating)",r"(lawsuit|class.action)",r"guidance.{0,20}(cut|lower|trim)"]
        for h in hs:
            hl=h.lower(); cat="NEUTRAL"
            for p in HARD:
                if _re.search(p,hl): cat="HARD_STOP"; hsc+=1; break
            if cat=="NEUTRAL":
                for p in ELEV:
                    if _re.search(p,hl): cat="ELEVATED"; break
            if cat=="NEUTRAL" and any(w in hl for w in ["beat","exceed","raise","upgrade","record","expand"]): cat="POSITIVE_CATALYST"
            cl.append({"headline":h[:100],"category":cat})
        return _j.dumps({"symbol":sym,"classifications":cl,"hard_stop_count":hsc,"gate_G5_pass":hsc==0,"status":"ok"})

    elif name == "assess_trade_setup":
        prob=float(args.get("probability",0.5)); pd_=float(args.get("pct_today",0) or 0)
        sp=float(args.get("spy_pct",0) or 0); qp=float(args.get("qqq_pct",0) or 0)
        score=50+(prob-0.5)*80
        if pd_>3: score-=10
        if pd_<-5: score+=5
        mkt=(sp+qp)/2
        if mkt>0.5: score+=8
        if mkt<-0.5: score-=8
        if mkt<-1: score-=8
        score=max(0,min(100,score))
        lean="SKIP" if score<40 else ("REDUCE_HALF" if score<65 else "PROCEED")
        return _j.dumps({"symbol":sym,"setup_score":round(score,1),"lean":lean,"status":"ok"})

    return '{"status":"unknown_tool"}'


def _build_llm_user_message(candidates, spy_pct, qqq_pct, num_signals):
    spy_str = f"{spy_pct:+.2f}%" if spy_pct is not None else "unknown"
    qqq_str = f"{qqq_pct:+.2f}%" if qqq_pct is not None else "unknown"
    lines = [f"Market today: SPY {spy_str}, QQQ {qqq_str}.",f"{num_signals} candidates above threshold.","","Candidates:"]
    for c in candidates:
        sec=c.get("sector") or ""; pct=c.get("pct_today")
        pct_str=f"{pct:+.2f}%" if isinstance(pct,(int,float)) and pct is not None else "unknown"
        lines.append(f"- {c['symbol']} prob={c['probability']:.4f} today={pct_str} sector={sec or 'unknown'}")
        for h in c.get("headlines",[]): lines.append(f"    NEWS: {h}")
    lines.append(""); lines.append("For each symbol: run all 6 tools, complete all 5 phases, then return JSON.")
    return "\n".join(lines)


def call_llm_filter(candidates, spy_pct, qqq_pct, num_signals):
    if not GROQ_API_KEY:
        print("LLM FILTER: key not set; skipping",flush=True); return {}
    try:
        import requests as _req, json as _j
    except ImportError:
        return {}
    try:
        messages=[{"role":"system","content":LLM_SYSTEM_PROMPT},{"role":"user","content":_build_llm_user_message(candidates,spy_pct,qqq_pct,num_signals)}]
        headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json","HTTP-Referer":"https://macro-intelligence.local","X-Title":"MacroIntelligence"}
        tool_calls_made=0
        for _round in range(14):
            body={"model":LLM_MODEL,"messages":messages,"tools":TOOLS,"tool_choice":"required" if tool_calls_made<6 else "auto","temperature":0.1}
            resp=_req.post("https://openrouter.ai/api/v1/chat/completions",headers=headers,json=body,timeout=60)
            resp.raise_for_status()
            choice=resp.json()["choices"][0]; msg=choice["message"]; finish=choice.get("finish_reason","")
            messages.append(msg)
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn=tc["function"]["name"]
                    try: fa=_j.loads(tc["function"]["arguments"] or "{}")
                    except: fa={}
                    if fn=="assess_trade_setup":
                        sym=fa.get("symbol","").upper(); cm=next((c for c in candidates if c["symbol"].upper()==sym),{})
                        fa.setdefault("pct_today",cm.get("pct_today")); fa.setdefault("spy_pct",spy_pct); fa.setdefault("qqq_pct",qqq_pct); fa.setdefault("sector",cm.get("sector",""))
                    if fn=="check_sector_performance":
                        sym=fa.get("symbol","").upper(); cm=next((c for c in candidates if c["symbol"].upper()==sym),{})
                        fa.setdefault("sector",cm.get("sector",""))
                    tr=_handle_tool_call(fn,fa)
                    print(f"  TOOL {fn}({fa.get('symbol','')}) -> {tr[:80]}",flush=True)
                    tool_calls_made+=1
                    messages.append({"role":"tool","tool_call_id":tc["id"],"content":tr})
                continue
            if finish in ("stop","end_turn",""):
                text=(msg.get("content") or "").strip()
                if text.startswith("```"):
                    parts=text.split("```"); text=parts[1] if len(parts)>1 else text
                    if text.lstrip().lower().startswith("json"): text=text.lstrip()[4:]
                text=text.strip().strip("`").strip()
                bs=text.find("{"); be=text.rfind("}")
                if bs!=-1 and be!=-1: text=text[bs:be+1]
                d=_j.loads(text)
                if not isinstance(d,dict): raise ValueError("not a dict")
                return {k.upper():v for k,v in d.items()}
            return {}
        return {}
    except Exception as e:
        print(f"LLM FILTER: failed ({e}); keeping all signals",flush=True); return {}



def _apply_llm_decisions(signals, decisions):
    out = []
    for s in signals:
        sym = s["symbol"]
        d = decisions.get(sym) or decisions.get(sym.upper()) or {}
        decision = str(d.get("decision", "proceed")).lower().strip()
        reason = str(d.get("reason", "")).strip()[:120]
        confidence = str(d.get("confidence", "medium")).lower().strip()
        if decision == "skip":
            print(f"LLM {sym}: skip [{confidence}] — {reason}", flush=True)
            continue
        if decision == "reduce_half":
            s["position_pct"] = round(s["position_pct"] / 2.0, 6)
            s["llm_decision"] = "reduce_half"
            s["llm_reason"] = reason
            s["llm_confidence"] = confidence
            print(f"LLM {sym}: reduce_half [{confidence}] — {reason}", flush=True)
            out.append(s)
            continue
        s["llm_decision"] = "proceed"
        s["llm_reason"] = reason
        s["llm_confidence"] = confidence
        print(f"LLM {sym}: proceed [{confidence}] — {reason}", flush=True)
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
    import builtins as _bt
    _all_pq = sorted(HIST_ROOT.glob("*.parquet"))
    _lim2 = getattr(_bt, "_SYMBOL_LIMIT", 0)
    if _lim2 > 0: _all_pq = _all_pq[:_lim2 * 10]
    for p in _all_pq:
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



def _recompute_rolling_features(df):
    """Recompute rolling indicators so newly-appended OHLCV rows have valid features."""
    if df is None or df.empty or "close" not in df.columns:
        return df
    try:
        c = df["close"].astype(float)
        h = df["high"].astype(float) if "high" in df.columns else c
        l = df["low"].astype(float) if "low" in df.columns else c
        # Returns
        df["return_1d"] = c.pct_change(1) * 100
        df["returns_1d"] = c.pct_change(1)
        df["return_5d"] = c.pct_change(5) * 100
        df["return_20d"] = c.pct_change(20) * 100
        # SMAs
        df["sma_20"] = c.rolling(20, min_periods=10).mean()
        df["sma_50"] = c.rolling(50, min_periods=25).mean()
        df["sma_200"] = c.rolling(200, min_periods=50).mean()
        df["sma_200_sign"] = (df["sma_200"].diff() > 0).astype(float)
        df["sma_50_sign"] = (df["sma_50"].diff() > 0).astype(float)
        df["close_vs_sma_50"] = (c - df["sma_50"]) / df["sma_50"].replace(0, np.nan) * 100
        df["close_vs_sma_200"] = (c - df["sma_200"]) / df["sma_200"].replace(0, np.nan) * 100
        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain_14 = gain.ewm(com=13, adjust=False).mean()
        avg_loss_14 = loss.ewm(com=13, adjust=False).mean().replace(0, np.nan)
        df["rsi_14"] = 100 - 100 / (1 + avg_gain_14 / avg_loss_14)
        avg_gain_21 = gain.ewm(com=20, adjust=False).mean()
        avg_loss_21 = loss.ewm(com=20, adjust=False).mean().replace(0, np.nan)
        df["rsi_21"] = 100 - 100 / (1 + avg_gain_21 / avg_loss_21)
        # ATR
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(com=13, adjust=False).mean()
        df["atr_pct"] = df["atr_14"] / c.replace(0, np.nan) * 100
        # Bollinger Bands
        bb_mid = c.rolling(20, min_periods=10).mean()
        bb_std = c.rolling(20, min_periods=10).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = (bb_upper - bb_lower).replace(0, np.nan)
        df["bb_width"] = bb_range / bb_mid.replace(0, np.nan)
        df["bb_position"] = (c - bb_lower) / bb_range
        df["zscore_vs_60d"] = (c - c.rolling(60, min_periods=20).mean()) / c.rolling(60, min_periods=20).std().replace(0, np.nan)
        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        # Momentum
        df["momentum_20d"] = c.pct_change(20) * 100
        df["momentum_60d"] = c.pct_change(60) * 100
        df["momentum_differencing"] = df["momentum_20d"] - df["momentum_60d"]
        df["momentum_squared"] = df["momentum_20d"] ** 2
        df["roc_20"] = c.pct_change(20) * 100
        # Volatility
        daily_ret = c.pct_change()
        df["realized_vol_21d"] = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        vol_short = daily_ret.rolling(21, min_periods=10).std()
        vol_long = daily_ret.rolling(63, min_periods=30).std().replace(0, np.nan)
        df["vol_regime_ratio"] = vol_short / vol_long
        df["vol_regime_stressed"] = (df["vol_regime_ratio"] > 1.5).astype(float)
        # Price acceleration
        df["price_acceleration"] = df["return_1d"] - df["return_1d"].shift(1)
    except Exception as e:
        print(f"WARN _recompute_rolling_features: {e}", flush=True)
    return df


def _recompute_rolling_features(df):
    """Recompute rolling indicators so newly-appended OHLCV rows have valid features."""
    if df is None or df.empty or "close" not in df.columns:
        return df
    try:
        c = df["close"].astype(float)
        h = df["high"].astype(float) if "high" in df.columns else c
        l = df["low"].astype(float) if "low" in df.columns else c
        # Returns
        df["return_1d"] = c.pct_change(1) * 100
        df["returns_1d"] = c.pct_change(1)
        df["return_5d"] = c.pct_change(5) * 100
        df["return_20d"] = c.pct_change(20) * 100
        # SMAs
        df["sma_20"] = c.rolling(20, min_periods=10).mean()
        df["sma_50"] = c.rolling(50, min_periods=25).mean()
        df["sma_200"] = c.rolling(200, min_periods=50).mean()
        df["sma_200_sign"] = (df["sma_200"].diff() > 0).astype(float)
        df["sma_50_sign"] = (df["sma_50"].diff() > 0).astype(float)
        df["close_vs_sma_50"] = (c - df["sma_50"]) / df["sma_50"].replace(0, np.nan) * 100
        df["close_vs_sma_200"] = (c - df["sma_200"]) / df["sma_200"].replace(0, np.nan) * 100
        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain_14 = gain.ewm(com=13, adjust=False).mean()
        avg_loss_14 = loss.ewm(com=13, adjust=False).mean().replace(0, np.nan)
        df["rsi_14"] = 100 - 100 / (1 + avg_gain_14 / avg_loss_14)
        avg_gain_21 = gain.ewm(com=20, adjust=False).mean()
        avg_loss_21 = loss.ewm(com=20, adjust=False).mean().replace(0, np.nan)
        df["rsi_21"] = 100 - 100 / (1 + avg_gain_21 / avg_loss_21)
        # ATR
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(com=13, adjust=False).mean()
        df["atr_pct"] = df["atr_14"] / c.replace(0, np.nan) * 100
        # Bollinger Bands
        bb_mid = c.rolling(20, min_periods=10).mean()
        bb_std = c.rolling(20, min_periods=10).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = (bb_upper - bb_lower).replace(0, np.nan)
        df["bb_width"] = bb_range / bb_mid.replace(0, np.nan)
        df["bb_position"] = (c - bb_lower) / bb_range
        df["zscore_vs_60d"] = (c - c.rolling(60, min_periods=20).mean()) / c.rolling(60, min_periods=20).std().replace(0, np.nan)
        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        # Momentum
        df["momentum_20d"] = c.pct_change(20) * 100
        df["momentum_60d"] = c.pct_change(60) * 100
        df["momentum_differencing"] = df["momentum_20d"] - df["momentum_60d"]
        df["momentum_squared"] = df["momentum_20d"] ** 2
        df["roc_20"] = c.pct_change(20) * 100
        # Volatility
        daily_ret = c.pct_change()
        df["realized_vol_21d"] = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        vol_short = daily_ret.rolling(21, min_periods=10).std()
        vol_long = daily_ret.rolling(63, min_periods=30).std().replace(0, np.nan)
        df["vol_regime_ratio"] = vol_short / vol_long
        df["vol_regime_stressed"] = (df["vol_regime_ratio"] > 1.5).astype(float)
        # Price acceleration
        df["price_acceleration"] = df["return_1d"] - df["return_1d"].shift(1)
    except Exception as e:
        print(f"WARN _recompute_rolling_features: {e}", flush=True)
    return df

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
    df = _recompute_rolling_features(df)
    df = _recompute_rolling_features(df)
    return df

def file_mtime_et_date(path):
    return datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo("America/New_York")).date()

def latest_rows(features, allowed):
    rows = []
    fresh_count = 0
    stale_count = 0
    sector_filtered = 0
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
        last_row = x.iloc[-1]
        price = float(last_row.get("close", 0) or 0)
        adv = float(last_row.get("adv20_dollar_vol", 0) or 0)
        pct_today = float("nan")
        if len(x) >= 2:
            try:
                prev_close = float(x["close"].iloc[-2])
                if prev_close > 0:
                    pct_today = (price - prev_close) / prev_close * 100.0
            except Exception:
                pct_today = float("nan")
        # Daily refresh appends raw OHLCV without recomputing rolling features.
        # Score on last row where rsi_14 is populated; use today close for price.
        if "rsi_14" in x.columns:
            valid_feat = x[x["rsi_14"].notna()]
            r = valid_feat.iloc[-1].copy() if len(valid_feat) > 0 else last_row.copy()
        else:
            r = last_row.copy()
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
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import builtins; builtins._SYMBOL_LIMIT = args.limit
    model_path, model, features = load_model()
    allowed = allowed_symbols()
    vol, regime, mult = spy_vol()
    live = latest_rows(features, allowed)

    X = live[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else proba.shape[1]-1
    live["probability"] = proba[:, pos_idx]
    qualified = live[live["probability"] >= SIG_THRESHOLD].sort_values("probability", ascending=False)
    pre_cap = len(qualified)
    picks = select_top_n_with_biotech_cap(qualified, SIG_TOP_N, MAX_BIOTECH_PER_DAY)
    if MAX_BIOTECH_PER_DAY < SIG_TOP_N:
        print(f"BIOTECH CAP: max_per_day={MAX_BIOTECH_PER_DAY} qualified={pre_cap} selected={len(picks)}", flush=True)

    pre = len(picks)
    earnings_cols = {"official_event_hit": 1, "filing_event_hit": 1, "event_day_extreme": 1}
    for col, threshold in earnings_cols.items():
        if col in picks.columns:
            picks = picks[picks[col].fillna(0) < threshold]
    if "peer_earnings_shock_3d" in picks.columns:
        shock_cap = float(os.getenv("SIG_PEER_SHOCK_CAP", "2.0"))
        picks = picks[picks["peer_earnings_shock_3d"].fillna(0).abs() < shock_cap]
    print(f"EARNINGS FILTER removed={pre - len(picks)} remaining={len(picks)}")
    try:
        import yfinance as yf
        from datetime import date, timedelta
        hold_end = date.today() + timedelta(days=HOLD_DAYS + 2)
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
    _save_sector_cache()
    signal_date = datetime.now().date().isoformat()
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
                candidates.append({"symbol": sym, "probability": s["probability"], "pct_today": pct_today, "sector": sector, "headlines": headlines})
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
