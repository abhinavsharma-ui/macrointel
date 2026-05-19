#!/usr/bin/env python3
import pathlib, re, sys

TARGET = pathlib.Path("scripts/fixed_return_daily_signals.py")
src = TARGET.read_text(encoding="utf-8")

# ── 1. Fix the unterminated string at line 423 ───────────────────────────────
# The patch left a broken f-string like: return "
# Find and fix _handle_tool_call's last return line
src = re.sub(
    r"return '\{\"status\":\"unknown_tool\"\}'",
    'return \'{"status":"unknown_tool"}\'',
    src
)
# Also fix any bare: return "  (unterminated)
src = re.sub(r'return "\s*\n', 'return ""\n', src)

# ── 2. Inject TOOLS list if missing ─────────────────────────────────────────
TOOLS_BLOCK = '''
TOOLS = [
    {"type":"function","function":{"name":"check_short_interest","description":"Return short interest % of float and short ratio. Gate G1: <10% required for PROCEED.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
    {"type":"function","function":{"name":"check_options_iv","description":"Return 30-day ATM implied volatility (%) and put/call ratio. Gate G2: IV<50% required.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
    {"type":"function","function":{"name":"check_price_momentum","description":"Return 10-day trend (up/down/flat), RSI-14, pct from 52w high/low. Gates G3 and G6.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
    {"type":"function","function":{"name":"check_sector_performance","description":"Return 5-day return of sector ETF. Gate G4: >-2% required.","parameters":{"type":"object","properties":{"symbol":{"type":"string"},"sector":{"type":"string"}},"required":["symbol","sector"]}}},
    {"type":"function","function":{"name":"analyze_news_risk","description":"Classify headlines: HARD_STOP / ELEVATED / NEUTRAL / POSITIVE_CATALYST. Gate G5: no HARD_STOP.","parameters":{"type":"object","properties":{"symbol":{"type":"string"},"headlines":{"type":"array","items":{"type":"string"}}},"required":["symbol","headlines"]}}},
    {"type":"function","function":{"name":"assess_trade_setup","description":"Holistic setup score 0-100. <40 lean SKIP, 40-65 REDUCE_HALF, >65 PROCEED.","parameters":{"type":"object","properties":{"symbol":{"type":"string"},"probability":{"type":"number"},"pct_today":{"type":"number"},"spy_pct":{"type":"number"},"qqq_pct":{"type":"number"},"sector":{"type":"string"}},"required":["symbol","probability"]}}}
]
'''

if "TOOLS = [" not in src:
    # Insert right before _handle_tool_call or _build_llm_user_message
    for anchor in ["def _handle_tool_call(", "def _build_llm_user_message("]:
        if anchor in src:
            src = src.replace(anchor, TOOLS_BLOCK.strip() + "\n\n\n" + anchor, 1)
            break

# ── 3. Write back ────────────────────────────────────────────────────────────
TARGET.write_text(src, encoding="utf-8")

# ── 4. Verify ────────────────────────────────────────────────────────────────
v = TARGET.read_text()
p = sum(f"PHASE {i}" in v for i in range(1, 6))
print(f"\n{'='*50}")
print(f"File      : {TARGET}")
print(f"PHASES    : {p}/5  {'OK' if p==5 else 'INCOMPLETE'}")
print(f"TOOLS     : {'OK' if 'TOOLS = [' in v else 'MISSING'}")
print(f"OpenRouter: {'OK' if 'openrouter.ai' in v else 'MISSING'}")
print(f"ToolLoop  : {'OK' if 'tool_calls_made' in v else 'MISSING'}")
print(f"{'='*50}")

# syntax check
import py_compile, tempfile, shutil
tmp = pathlib.Path(tempfile.mktemp(suffix=".py"))
shutil.copy(TARGET, tmp)
try:
    py_compile.compile(str(tmp), doraise=True)
    print("SYNTAX    : OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX    : ERROR -> {e}")
finally:
    tmp.unlink(missing_ok=True)
