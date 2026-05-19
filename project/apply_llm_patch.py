#!/usr/bin/env python3
import pathlib, sys
TARGET = pathlib.Path("scripts/fixed_return_daily_signals.py")
if not TARGET.exists():
    sys.exit(f"ERROR: {TARGET} not found — run from project dir")
src = TARGET.read_text(encoding="utf-8")

NEW_PROMPT = '''LLM_SYSTEM_PROMPT = """
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
NULL HYPOTHESIS: DEFAULT = SKIP
Burden of proof is on PROCEED. Absence of a strong
positive case = SKIP.  Ambiguity = SKIP.
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

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
  [ ] G1 Short interest < 10% of float
  [ ] G2 Options IV < 50%
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
SKIP       -> any gate FALSE/UNKNOWN OR cannot rebut steelman
REDUCE_HALF -> all gates TRUE but G2 borderline (IV 40-50%) or
               G1 borderline (short 8-10%) or sector -1% to -2%
PROCEED    -> all 6 gates TRUE, rebuttal holds, pre-mortem priced in

INVALID reasons for PROCEED:
  - "News is neutral"
  - "Probability score is high"
  - "No red flags found"
  - "Market is broadly positive" (alone)

EMPTY DATA: empty tool result = UNKNOWN not CLEAN.
Three or more UNKNOWN -> auto SKIP.

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
'''

NEW_TOOLS = '''
TOOLS = [
    {"type":"function","function":{"name":"check_short_interest","description":"Return short interest % of float and short ratio. Gate G1: <10% required for PROCEED.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
    {"type":"function","function":{"name":"check_options_iv","description":"Return 30-day ATM implied volatility (%) and put/call ratio. Gate G2: IV<50% required.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
    {"type":"function","function":{"name":"check_price_momentum","description":"Return 10-day trend (up/down/flat), RSI-14, pct from 52w high/low. Gates G3 and G6.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
    {"type":"function","function":{"name":"check_sector_performance","description":"Return 5-day return of sector ETF. Gate G4: >-2% required.","parameters":{"type":"object","properties":{"symbol":{"type":"string"},"sector":{"type":"string"}},"required":["symbol","sector"]}}},
    {"type":"function","function":{"name":"analyze_news_risk","description":"Classify headlines: HARD_STOP / ELEVATED / NEUTRAL / POSITIVE_CATALYST. Gate G5: no HARD_STOP.","parameters":{"type":"object","properties":{"symbol":{"type":"string"},"headlines":{"type":"array","items":{"type":"string"}}},"required":["symbol","headlines"]}}},
    {"type":"function","function":{"name":"assess_trade_setup","description":"Holistic setup score 0-100. <40 lean SKIP, 40-65 REDUCE_HALF, >65 PROCEED.","parameters":{"type":"object","properties":{"symbol":{"type":"string"},"probability":{"type":"number"},"pct_today":{"type":"number"},"spy_pct":{"type":"number"},"qqq_pct":{"type":"number"},"sector":{"type":"string"}},"required":["symbol","probability"]}}}
]
'''

NEW_HANDLER = '''
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
                    d=c.diff(); g=d.clip(lower=0).rolling(14,min_periods=14).mean(); l=(-d.clip(upper=0)).rolling(14,min_periods=14).mean()
                    r["rsi_14"]=round(float(100-100/(1+g/l.replace(0,1e-9))).iloc[-1],1)
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
'''

NEW_FILTER = '''
def _build_llm_user_message(candidates, spy_pct, qqq_pct, num_signals):
    spy_str = f"{spy_pct:+.2f}%" if spy_pct is not None else "unknown"
    qqq_str = f"{qqq_pct:+.2f}%" if qqq_pct is not None else "unknown"
    lines = [f"Market today: SPY {spy_str}, QQQ {qqq_str}.",f"{num_signals} candidates above threshold.","","Candidates:"]
    for c in candidates:
        sec=c.get("sector") or ""; pct=c.get("pct_today")
        pct_str=f"{pct:+.2f}%" if isinstance(pct,(int,float)) and pct is not None else "unknown"
        lines.append(f"- {c[\'symbol\']} prob={c[\'probability\']:.4f} today={pct_str} sector={sec or \'unknown\'}")
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
                    print(f"  TOOL {fn}({fa.get(\'symbol\','')}) -> {tr[:80]}",flush=True)
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
'''

# ── Apply the patch ───────────────────────────────────────────────────────────
import re

# Replace LLM_SYSTEM_PROMPT block
src = re.sub(
    r'LLM_SYSTEM_PROMPT = """.*?"""',
    NEW_PROMPT.strip(),
    src, flags=re.DOTALL
)

# Insert TOOLS after LLM_SYSTEM_PROMPT block
if "TOOLS = [" not in src:
    src = src.replace(
        'def _build_llm_user_message(',
        NEW_TOOLS.strip() + '\n\n\ndef _build_llm_user_message('
    )

# Replace _handle_tool_call if exists, else insert before _build_llm_user_message
if "def _handle_tool_call(" in src:
    src = re.sub(r'def _handle_tool_call\(.*?(?=\ndef )', NEW_HANDLER.strip()+'\n\n\n', src, flags=re.DOTALL)
else:
    src = src.replace('def _build_llm_user_message(', NEW_HANDLER.strip()+'\n\n\ndef _build_llm_user_message(')

# Replace _build_llm_user_message + call_llm_filter
src = re.sub(r'def _build_llm_user_message\(.*?(?=\ndef _apply_llm_decisions)', NEW_FILTER.strip()+'\n\n\n', src, flags=re.DOTALL)

TARGET.write_text(src, encoding="utf-8")

# Verify
v = TARGET.read_text()
p = sum(f"PHASE {i}" in v for i in range(1,6))
print(f"\n{'='*50}")
print(f"File: {TARGET}")
print(f"PHASES    : {p}/5  {'OK' if p==5 else 'INCOMPLETE'}")
print(f"TOOLS     : {'OK' if 'TOOLS = [' in v else 'MISSING'}")
print(f"OpenRouter: {'OK' if 'openrouter.ai' in v else 'MISSING'}")
print(f"ToolLoop  : {'OK' if 'tool_calls_made' in v else 'MISSING'}")
print(f"{'='*50}")
