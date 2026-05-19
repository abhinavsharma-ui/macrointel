#!/usr/bin/env python3
"""
Patch fixed_return_daily_signals.py with:
  1. Full 5-phase LLM system prompt (null hypothesis / pre-mortem / ReAct /
     deterministic checklist / steelman+rebuttal)
  2. 6 structured tool definitions
  3. call_llm_filter rewritten for OpenRouter + tool-loop + scratchpad

Run on the VM:
    cd ~/macro_intelligence_complete/project
    python3 apply_llm_patch.py
"""
import pathlib, sys, textwrap

TARGET = pathlib.Path(
    "~/macro_intelligence_complete/project/scripts/fixed_return_daily_signals.py"
).expanduser()

if not TARGET.exists():
    sys.exit(f"ERROR: {TARGET} not found")

src = TARGET.read_text(encoding="utf-8")

# ── Verify we can find the anchor lines ──────────────────────────────────────
OLD_PROMPT_START = 'LLM_SYSTEM_PROMPT = """You are a senior equity-trading judgment layer'
OLD_FUNC_END     = '        return {}\n'   # last line of old call_llm_filter

if OLD_PROMPT_START not in src:
    # Maybe it was already patched — check for new prompt marker
    if "NULL HYPOTHESIS" in src and "PHASE 1" in src:
        print("✓ Prompt already patched. Verifying tool loop...")
        if "openrouter.ai" in src and "tool_calls_made" in src:
            print("✓ call_llm_filter already on OpenRouter + tool loop.")
            sys.exit(0)
        else:
            print("  call_llm_filter needs updating — continuing patch.")
            OLD_PROMPT_START = 'LLM_SYSTEM_PROMPT = """'   # fallback anchor
    else:
        sys.exit("ERROR: Cannot find anchor in file. Wrong file or already heavily modified.")

# ──────────────────────────────────────────────────────────────────────────────
# NEW CODE BLOCK  (replaces everything from LLM_SYSTEM_PROMPT through the end
# of the old call_llm_filter function)
# ──────────────────────────────────────────────────────────────────────────────
NEW_BLOCK = r'''LLM_SYSTEM_PROMPT = """
══════════════════════════════════════════════════════════
NULL HYPOTHESIS: DEFAULT = SKIP
Burden of proof is on PROCEED. Absence of a strong
positive case = SKIP.  Ambiguity = SKIP.
══════════════════════════════════════════════════════════

You are a skeptical senior equity-trading risk officer.
Trade: 10-day long momentum/mean-reversion, entry at next open,
profit target 8%, stop loss 6%.

════════════════════════════
PHASE 1 — PRE-MORTEM
════════════════════════════
Before calling any tool, assume this trade hits its stop loss on
Day 3. Write 2–3 bullet points explaining WHY it failed. Use only
information already visible in the user message (symbol, today's
move, sector, headlines). Label this block:
  PRE-MORTEM: <your bullets>

════════════════════════════
PHASE 2 — TOOL INVESTIGATION  (ReAct pattern)
════════════════════════════
You MUST call all 6 tools exactly once, in any order.
After EACH tool returns, write one THOUGHT sentence before
calling the next tool. Format:
  THOUGHT: <one sentence interpreting the data just returned>

UNKNOWN PROTOCOL: If a tool returns empty or null data, mark that
dimension as UNKNOWN — never assume it is clean or safe.

════════════════════════════
PHASE 3 — DETERMINISTIC CHECKLIST
════════════════════════════
After all 6 tools, answer each gate TRUE/FALSE:
  [ ] G1 Short interest < 10% of float
  [ ] G2 Options IV < 50% (no extreme vol premium)
  [ ] G3 10-day price trend not in confirmed downtrend
  [ ] G4 Sector ETF 5-day return > -2%
  [ ] G5 No hard-stop news (FDA rejection, fraud, going-concern,
         earnings collapse >10%, mass exec departure)
  [ ] G6 RSI 14-day not > 75 (not overbought)

If ANY gate is FALSE → decision is SKIP or REDUCE_HALF (see Phase 5).
If ANY gate is UNKNOWN → treat as FALSE for that gate.

════════════════════════════
PHASE 4 — STEELMAN + REBUTTAL
════════════════════════════
Write the single strongest argument FOR skipping this trade.
Then write your rebuttal. If you cannot rebut it, decision = SKIP.
  STEELMAN_SKIP: <best argument to skip>
  REBUTTAL: <your counter-argument, or "Cannot rebut.">

════════════════════════════
PHASE 5 — FINAL DECISION
════════════════════════════
SKIP   → any G1-G6 gate FALSE/UNKNOWN  OR  cannot rebut steelman
REDUCE_HALF → all gates TRUE but G2 borderline (IV 40-50%) or
              G1 borderline (short 8-10%) or sector -1% to -2%
PROCEED → all 6 gates TRUE, rebuttal holds, pre-mortem risks are
          already priced into the setup

✗ INVALID reasons for PROCEED:
  - "News is neutral" (neutral ≠ positive catalyst)
  - "Probability score is high" (model already selected it)
  - "No red flags found" (absence of evidence ≠ evidence of safety)
  - "Market is broadly positive" (alone is not sufficient)

════════════════════════════
EMPTY DATA PROTOCOL
════════════════════════════
Empty or null tool results = UNKNOWN (not CLEAN).
Three or more UNKNOWN dimensions → SKIP automatically.

════════════════════════════
OUTPUT FORMAT
════════════════════════════
After completing all phases, return ONLY valid JSON. No prose,
no markdown fences. Schema:
{
  "SYMBOL": {
    "decision": "proceed|skip|reduce_half",
    "reason": "<≤15 words citing the decisive gate or factor>",
    "scratchpad": "<pre-mortem + key THOUGHT lines + checklist + steelman summary>"
  }
}
One entry per input symbol. Include every symbol passed in.
"""


# ── 6 tools the LLM must call ─────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_short_interest",
            "description": (
                "Return short interest % of float and short ratio for the symbol. "
                "Gate G1: short interest < 10% is required for PROCEED."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_options_iv",
            "description": (
                "Return the 30-day ATM implied volatility (%) and put/call open interest "
                "ratio for the nearest expiry. Gate G2: IV < 50% required for PROCEED."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_price_momentum",
            "description": (
                "Return the 10-day price trend (up/down/flat), RSI-14, distance from "
                "52-week high (%), distance from 52-week low (%). "
                "Gate G3: trend not confirmed downtrend; Gate G6: RSI ≤ 75."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_sector_performance",
            "description": (
                "Return the 5-day return (%) of the sector ETF corresponding to the "
                "candidate's sector. Gate G4: sector ETF 5-day return > -2%."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol":  {"type": "string", "description": "Ticker symbol"},
                    "sector":  {"type": "string", "description": "Sector string, e.g. Technology"}
                },
                "required": ["symbol", "sector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_news_risk",
            "description": (
                "Classify each provided headline into a risk category: "
                "HARD_STOP (FDA rejection, fraud, going-concern, earnings collapse >10%, "
                "mass exec departure), ELEVATED (lawsuit, downgrade, guidance cut), "
                "NEUTRAL, or POSITIVE_CATALYST. "
                "Gate G5: no HARD_STOP headlines allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol":    {"type": "string"},
                    "headlines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of news headlines to classify"
                    }
                },
                "required": ["symbol", "headlines"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_trade_setup",
            "description": (
                "Holistic setup quality score (0-100) based on probability score, "
                "today's price move, market context (SPY/QQQ %), and sector. "
                "Score < 40 → lean SKIP; 40-65 → lean REDUCE_HALF; > 65 → lean PROCEED."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol":      {"type": "string"},
                    "probability": {"type": "number", "description": "Model probability 0-1"},
                    "pct_today":   {"type": "number", "description": "Today's % move"},
                    "spy_pct":     {"type": "number", "description": "SPY today % move"},
                    "qqq_pct":     {"type": "number", "description": "QQQ today % move"},
                    "sector":      {"type": "string"}
                },
                "required": ["symbol", "probability"],
            },
        },
    },
]


def _handle_tool_call(name: str, args: dict) -> str:
    """Execute a tool call and return a JSON string result."""
    import json as _json
    sym = args.get("symbol", "").upper()

    try:
        import yfinance as _yf
    except ImportError:
        _yf = None

    if name == "check_short_interest":
        result = {"symbol": sym, "short_pct_of_float": None, "short_ratio": None, "status": "unknown"}
        if _yf:
            try:
                info = _yf.Ticker(sym).info
                spf  = info.get("shortPercentOfFloat")
                sr   = info.get("shortRatio")
                if spf is not None:
                    result["short_pct_of_float"] = round(float(spf) * 100, 2)
                if sr is not None:
                    result["short_ratio"] = round(float(sr), 2)
                result["status"] = "ok" if spf is not None else "no_data"
            except Exception as e:
                result["status"] = f"error:{e}"
        return _json.dumps(result)

    elif name == "check_options_iv":
        result = {"symbol": sym, "iv_pct": None, "put_call_ratio": None, "status": "unknown"}
        if _yf:
            try:
                tk      = _yf.Ticker(sym)
                expiries = tk.options
                if expiries:
                    chain = tk.option_chain(expiries[0])
                    spot  = tk.info.get("regularMarketPrice") or tk.info.get("currentPrice")
                    calls = chain.calls
                    puts  = chain.puts
                    if spot and not calls.empty:
                        atm_call = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
                        iv_val   = atm_call["impliedVolatility"].iloc[0]
                        result["iv_pct"] = round(float(iv_val) * 100, 1)
                    if not calls.empty and not puts.empty:
                        pc = puts["openInterest"].sum() / max(calls["openInterest"].sum(), 1)
                        result["put_call_ratio"] = round(float(pc), 2)
                    result["status"] = "ok" if result["iv_pct"] is not None else "no_data"
                else:
                    result["status"] = "no_options_chain"
            except Exception as e:
                result["status"] = f"error:{e}"
        return _json.dumps(result)

    elif name == "check_price_momentum":
        result = {"symbol": sym, "trend_10d": "unknown", "rsi_14": None,
                  "pct_from_52w_high": None, "pct_from_52w_low": None, "status": "unknown"}
        if _yf:
            try:
                hist = _yf.Ticker(sym).history(period="60d")
                if len(hist) >= 10:
                    closes  = hist["close"] if "close" in hist.columns else hist["Close"]
                    last10  = closes.iloc[-10:]
                    slope   = (last10.iloc[-1] - last10.iloc[0]) / last10.iloc[0] * 100
                    result["trend_10d"] = "up" if slope > 1 else ("down" if slope < -1 else "flat")
                    # RSI-14
                    delta = closes.diff()
                    gain  = delta.clip(lower=0).rolling(14, min_periods=14).mean()
                    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
                    rs    = gain / loss.replace(0, 1e-9)
                    rsi   = 100 - 100 / (1 + rs)
                    result["rsi_14"] = round(float(rsi.iloc[-1]), 1)
                    high52 = closes.max()
                    low52  = closes.min()
                    price  = closes.iloc[-1]
                    result["pct_from_52w_high"] = round((price - high52) / high52 * 100, 1)
                    result["pct_from_52w_low"]  = round((price - low52)  / low52  * 100, 1)
                    result["status"] = "ok"
                else:
                    result["status"] = "insufficient_data"
            except Exception as e:
                result["status"] = f"error:{e}"
        return _json.dumps(result)

    elif name == "check_sector_performance":
        SECTOR_ETF = {
            "technology": "XLK", "information technology": "XLK",
            "healthcare": "XLV", "health care": "XLV",
            "financials": "XLF", "financial services": "XLF",
            "consumer discretionary": "XLY",
            "consumer staples": "XLP",
            "energy": "XLE",
            "industrials": "XLI",
            "materials": "XLB",
            "real estate": "XLRE",
            "utilities": "XLU",
            "communication services": "XLC", "communication": "XLC",
        }
        sector = str(args.get("sector", "")).lower()
        etf    = next((v for k, v in SECTOR_ETF.items() if k in sector), "SPY")
        result = {"symbol": sym, "sector_etf": etf, "five_day_return_pct": None, "status": "unknown"}
        if _yf:
            try:
                hist = _yf.Ticker(etf).history(period="10d")
                closes = hist["close"] if "close" in hist.columns else hist["Close"]
                if len(closes) >= 5:
                    ret = (closes.iloc[-1] - closes.iloc[-5]) / closes.iloc[-5] * 100
                    result["five_day_return_pct"] = round(float(ret), 2)
                    result["status"] = "ok"
                else:
                    result["status"] = "insufficient_data"
            except Exception as e:
                result["status"] = f"error:{e}"
        return _json.dumps(result)

    elif name == "analyze_news_risk":
        import re as _re
        headlines = args.get("headlines", [])
        HARD_STOP_PATTERNS = [
            r"fda.{0,20}(reject|refuse|decline|complet\w+ response|hold)",
            r"(fraud|going.concern|bankrupt|chapter.1[12]|delist)",
            r"(earnings|revenue|profit).{0,30}(miss|fall|drop|plunge|collapse).{0,20}[1-9]\d%",
            r"(ceo|cfo|coo|president).{0,30}(resign|depart|fired|step.down)",
            r"(sec|doj|ftc|cfpb).{0,20}(invest|charge|indict|probe|sue)",
            r"(acqui|merger|deal).{0,20}(terminat|collapse|fall.through|call.off)",
        ]
        ELEVATED_PATTERNS = [
            r"(downgrade|cut|lower).{0,20}(price.target|rating|outlook)",
            r"(lawsuit|litigation|class.action)",
            r"(guidance|forecast).{0,20}(cut|lower|reduce|trim)",
            r"(layoff|restructur|reorg).{0,20}\d+",
        ]
        classified = []
        hard_stop_count = 0
        for h in headlines:
            hl = h.lower()
            cat = "NEUTRAL"
            for p in HARD_STOP_PATTERNS:
                if _re.search(p, hl):
                    cat = "HARD_STOP"
                    hard_stop_count += 1
                    break
            if cat == "NEUTRAL":
                for p in ELEVATED_PATTERNS:
                    if _re.search(p, hl):
                        cat = "ELEVATED"
                        break
            if cat == "NEUTRAL":
                if any(w in hl for w in ["beat", "exceed", "raise", "upgrade", "accelerat", "record", "expand", "win"]):
                    cat = "POSITIVE_CATALYST"
            classified.append({"headline": h[:120], "category": cat})
        result = {
            "symbol": sym,
            "classifications": classified,
            "hard_stop_count": hard_stop_count,
            "gate_G5_pass": hard_stop_count == 0,
            "status": "ok",
        }
        return _json.dumps(result)

    elif name == "assess_trade_setup":
        prob     = float(args.get("probability", 0.5))
        pct_day  = float(args.get("pct_today",   0.0) or 0.0)
        spy_p    = float(args.get("spy_pct",      0.0) or 0.0)
        qqq_p    = float(args.get("qqq_pct",      0.0) or 0.0)
        sector   = str(args.get("sector", ""))
        # Heuristic scoring
        score = 50.0
        score += (prob - 0.5) * 80       # prob 0.7 → +16; prob 0.6 → +8
        if pct_day > 3:  score -= 10     # large gap-up is risky
        if pct_day < -3: score -= 5      # already sold off (mild positive for mean-rev)
        if pct_day < -5: score += 5      # oversold bounce candidate
        market_avg = (spy_p + qqq_p) / 2.0
        if market_avg > 0.5:  score += 8
        if market_avg < -0.5: score -= 8
        if market_avg < -1.0: score -= 8
        score = max(0, min(100, score))
        lean = "SKIP" if score < 40 else ("REDUCE_HALF" if score < 65 else "PROCEED")
        result = {
            "symbol": sym,
            "setup_score": round(score, 1),
            "lean": lean,
            "factors": {
                "probability_contribution": round((prob - 0.5) * 80, 1),
                "price_move_today_pct":     round(pct_day, 2),
                "market_context_avg_pct":   round(market_avg, 2),
            },
            "status": "ok",
        }
        return _json.dumps(result)

    return '{"status":"unknown_tool"}'


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
        sec     = c.get("sector") or ""
        sec_part = f"sector={sec}" if sec else "sector=unknown"
        pct     = c.get("pct_today")
        pct_str = f"{pct:+.2f}%" if isinstance(pct, (int, float)) and pct is not None else "unknown"
        lines.append(
            f"- {c['symbol']} prob={c['probability']:.4f} today={pct_str} {sec_part}"
        )
        for h in c.get("headlines", []):
            lines.append(f"    NEWS: {h}")
    lines.append("")
    lines.append(
        "For each symbol: run all 6 tools, complete all 5 phases, then return JSON."
    )
    return "\n".join(lines)


def call_llm_filter(candidates, spy_pct, qqq_pct, num_signals):
    """
    Returns {SYMBOL: {decision, reason, scratchpad}} or {} on any failure.
    Uses OpenRouter + 6 tool-calls (tool_choice='required' for first 6 iterations).
    """
    if not GROQ_API_KEY:
        print("LLM FILTER: GROQ_API_KEY (OpenRouter key) not set; skipping", flush=True)
        return {}
    try:
        import requests as _req
        import json as _json
    except ImportError:
        print("LLM FILTER: requests not installed; skipping", flush=True)
        return {}
    try:
        user_msg = _build_llm_user_message(candidates, spy_pct, qqq_pct, num_signals)
        headers = {
            "Authorization":  f"Bearer {GROQ_API_KEY}",
            "Content-Type":   "application/json",
            "HTTP-Referer":   "https://macro-intelligence.local",
            "X-Title":        "MacroIntelligence-LLMFilter",
        }
        messages = [
            {"role": "system",  "content": LLM_SYSTEM_PROMPT},
            {"role": "user",    "content": user_msg},
        ]

        tool_calls_made = 0
        MAX_TOOL_ROUNDS  = 12   # safety ceiling

        for _round in range(MAX_TOOL_ROUNDS):
            tc = "required" if tool_calls_made < 6 else "auto"
            body = {
                "model":       LLM_MODEL,
                "messages":    messages,
                "tools":       TOOLS,
                "tool_choice": tc,
                "temperature": 0.1,
            }
            resp = _req.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=60,
            )
            resp.raise_for_status()
            rj      = resp.json()
            choice  = rj["choices"][0]
            msg     = choice["message"]
            finish  = choice.get("finish_reason", "")

            # Append assistant turn
            messages.append(msg)

            # If the model wants to call tools, execute them
            if msg.get("tool_calls"):
                for tc_obj in msg["tool_calls"]:
                    fn_name = tc_obj["function"]["name"]
                    try:
                        fn_args = _json.loads(tc_obj["function"]["arguments"] or "{}")
                    except Exception:
                        fn_args = {}

                    # Inject spy_pct / qqq_pct into assess_trade_setup if missing
                    if fn_name == "assess_trade_setup":
                        sym = fn_args.get("symbol", "").upper()
                        cand_match = next(
                            (c for c in candidates if c["symbol"].upper() == sym), {}
                        )
                        fn_args.setdefault("pct_today", cand_match.get("pct_today"))
                        fn_args.setdefault("spy_pct",   spy_pct)
                        fn_args.setdefault("qqq_pct",   qqq_pct)
                        fn_args.setdefault("sector",    cand_match.get("sector", ""))
                    # Inject sector for check_sector_performance
                    if fn_name == "check_sector_performance":
                        sym = fn_args.get("symbol", "").upper()
                        cand_match = next(
                            (c for c in candidates if c["symbol"].upper() == sym), {}
                        )
                        fn_args.setdefault("sector", cand_match.get("sector", ""))

                    tool_result = _handle_tool_call(fn_name, fn_args)
                    print(
                        f"  TOOL {fn_name}({fn_args.get('symbol','')}) → "
                        f"{tool_result[:120]}",
                        flush=True,
                    )
                    tool_calls_made += 1
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc_obj["id"],
                        "content":      tool_result,
                    })
                continue   # next round to let model see tool results

            # No tool calls → model is done; extract JSON
            if finish in ("stop", "end_turn", ""):
                text = (msg.get("content") or "").strip()
                # Strip markdown fences
                if text.startswith("```"):
                    parts = text.split("```")
                    text  = parts[1] if len(parts) > 1 else text
                    if text.lstrip().lower().startswith("json"):
                        text = text.lstrip()[4:]
                text = text.strip().strip("`").strip()
                # Extract JSON block (model may add prose before/after)
                brace_start = text.find("{")
                brace_end   = text.rfind("}")
                if brace_start != -1 and brace_end != -1:
                    text = text[brace_start:brace_end + 1]
                decisions = _json.loads(text)
                if not isinstance(decisions, dict):
                    raise ValueError(f"LLM response not a dict: {type(decisions).__name__}")
                # Normalize keys to uppercase
                return {k.upper(): v for k, v in decisions.items()}

            # Unexpected finish reason
            print(f"LLM FILTER: unexpected finish_reason={finish!r}; aborting", flush=True)
            return {}

        print("LLM FILTER: exceeded max tool rounds; aborting", flush=True)
        return {}

    except Exception as e:
        print(f"LLM FILTER: call failed ({e}); proceeding with all signals", flush=True)
        return {}

'''  # end NEW_BLOCK


# ── Locate the region to replace ─────────────────────────────────────────────
# Find start: LLM_SYSTEM_PROMPT = """...
start_idx = src.find(OLD_PROMPT_START)
if start_idx == -1:
    sys.exit("ERROR: cannot locate LLM_SYSTEM_PROMPT in file")

# Find end: the closing line of call_llm_filter (last "        return {}")
# Search from start_idx forward for the function definition
func_def = "def call_llm_filter("
func_pos = src.find(func_def, start_idx)
if func_pos == -1:
    sys.exit("ERROR: cannot locate call_llm_filter function")

# Find the next top-level def after call_llm_filter
next_def = src.find("\ndef ", func_pos + len(func_def))
if next_def == -1:
    sys.exit("ERROR: cannot find end of call_llm_filter")

# The region to replace: from LLM_SYSTEM_PROMPT start → up to next top-level def
old_region = src[start_idx:next_def]
print(f"Replacing {len(old_region)} chars ({old_region.count(chr(10))} lines) "
      f"starting at char {start_idx}")

new_src = src[:start_idx] + NEW_BLOCK + src[next_def:]

# ── Write back ────────────────────────────────────────────────────────────────
TARGET.write_text(new_src, encoding="utf-8")

# ── Verify ────────────────────────────────────────────────────────────────────
verify = TARGET.read_text(encoding="utf-8")
phases_found = sum(f"PHASE {i}" in verify for i in range(1, 6))
tools_found  = "TOOLS = [" in verify
openrouter   = "openrouter.ai" in verify
loop_found   = "tool_calls_made" in verify

print(f"\n{'='*50}")
print(f"✓ File written: {TARGET}")
print(f"  PHASE count : {phases_found}/5  {'✓' if phases_found == 5 else '✗ INCOMPLETE'}")
print(f"  TOOLS list  : {'✓' if tools_found  else '✗ MISSING'}")
print(f"  OpenRouter  : {'✓' if openrouter   else '✗ MISSING'}")
print(f"  Tool loop   : {'✓' if loop_found   else '✗ MISSING'}")
print(f"{'='*50}")

if phases_found == 5 and tools_found and openrouter and loop_found:
    print("\nPatch applied successfully. Run smoke test:")
    print("  python3 scripts/fixed_return_daily_signals.py --dry-run")
else:
    print("\nWARNING: Some checks failed. Review the file manually.")
