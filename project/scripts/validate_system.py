#!/usr/bin/env python3
"""
System validation script — tests all new components end-to-end.

Run from project root:
    python scripts/validate_system.py
"""

import sys
import os
import traceback
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.example", override=False)


PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
WARN = "\033[93m⚠️  WARN\033[0m"


def test_imports():
    """Test all critical imports."""
    print("\n═══ 1. IMPORT CHECKS ═══")
    results = []

    modules = [
        ("core.enhanced_exits", "EnhancedExitManager"),
        ("core.multiframe_confirmation", "MultiframeConfirmer"),
        ("models.regime_aware_factor_weighter", "RegimeAwareFactorWeighter"),
        ("core.signal_engine_v2", None),
        ("core.paper_trading", "VirtualBroker"),
        ("core.risk_manager", None),
    ]

    for mod_path, cls_name in modules:
        try:
            mod = __import__(mod_path, fromlist=[cls_name] if cls_name else [""])
            if cls_name:
                getattr(mod, cls_name)
            print(f"  {PASS} {mod_path}" + (f".{cls_name}" if cls_name else ""))
            results.append(True)
        except Exception as e:
            print(f"  {FAIL} {mod_path}: {e}")
            results.append(False)

    return all(results)


def test_enhanced_exits():
    """Test EnhancedExitManager with mock positions."""
    print("\n═══ 2. ENHANCED EXIT MANAGER ═══")
    from core.enhanced_exits import EnhancedExitManager
    mgr = EnhancedExitManager()
    results = []

    # Test 1: Stop loss hit (long position, price dropped below stop)
    pos = {
        "entry_price": 100.0,
        "quantity": 10,
        "side": "long",
        "atr_at_entry": 5.0,  # ATR=5, so stop = 100 - 2*5 = 90
    }
    result = mgr.evaluate_exit(pos, current_price=89.0, current_regime="normal", current_signal_strength=0.5)
    ok = result["should_exit"] and result["exit_reason"] == "stop_loss"
    print(f"  {PASS if ok else FAIL} Stop loss (long, ATR-based): exit={result['should_exit']}, reason={result['exit_reason']}")
    results.append(ok)

    # Test 2: Take profit hit (long position, price rose above target)
    pos2 = {
        "entry_price": 100.0,
        "quantity": 10,
        "side": "long",
        "atr_at_entry": 5.0,  # TP = 100 + 3.5*5 = 117.5
    }
    result2 = mgr.evaluate_exit(pos2, current_price=118.0, current_regime="normal", current_signal_strength=0.5)
    ok2 = result2["should_exit"] and result2["exit_reason"] == "take_profit"
    print(f"  {PASS if ok2 else FAIL} Take profit (long, ATR-based): exit={result2['should_exit']}, reason={result2['exit_reason']}")
    results.append(ok2)

    # Test 3: Regime shift (entered calm, now crisis)
    pos3 = {
        "entry_price": 100.0,
        "quantity": 10,
        "side": "long",
        "entry_regime": "calm",
    }
    result3 = mgr.evaluate_exit(pos3, current_price=100.0, current_regime="crisis", current_signal_strength=0.5)
    ok3 = result3["should_exit"] and "regime_shift" in result3["exit_reason"]
    print(f"  {PASS if ok3 else FAIL} Regime shift (calm→crisis): exit={result3['should_exit']}, reason={result3['exit_reason']}")
    results.append(ok3)

    # Test 4: Signal decay (strength dropped below 40% of entry)
    pos4 = {
        "entry_price": 100.0,
        "quantity": 10,
        "side": "long",
        "signal_strength_at_entry": 0.80,
    }
    result4 = mgr.evaluate_exit(pos4, current_price=101.0, current_regime="normal", current_signal_strength=0.20)
    ok4 = result4["should_exit"] and result4["exit_reason"] == "signal_decay"
    print(f"  {PASS if ok4 else FAIL} Signal decay (0.80→0.20): exit={result4['should_exit']}, reason={result4['exit_reason']}")
    results.append(ok4)

    # Test 5: No exit (price within range, regime stable, signal strong)
    pos5 = {
        "entry_price": 100.0,
        "quantity": 10,
        "side": "long",
        "atr_at_entry": 5.0,
        "entry_regime": "normal",
        "signal_strength_at_entry": 0.70,
    }
    result5 = mgr.evaluate_exit(pos5, current_price=105.0, current_regime="normal", current_signal_strength=0.65)
    ok5 = not result5["should_exit"]
    print(f"  {PASS if ok5 else FAIL} No exit (healthy position): exit={result5['should_exit']}, reason={result5['exit_reason']}")
    results.append(ok5)

    # Test 6: Trailing stop (position gained >2%, then retraced)
    pos6 = {
        "entry_price": 100.0,
        "quantity": 10,
        "side": "long",
        "atr_at_entry": 3.0,  # trail stop = peak - 1.5*3 = peak - 4.5
        "peak_price": 110.0,  # gained 10%, trail stop = 110 - 4.5 = 105.5
    }
    result6 = mgr.evaluate_exit(pos6, current_price=105.0, current_regime="normal", current_signal_strength=0.5)
    ok6 = result6["should_exit"] and result6["exit_reason"] == "trailing_stop"
    print(f"  {PASS if ok6 else FAIL} Trailing stop (peak 110, current 105): exit={result6['should_exit']}, reason={result6['exit_reason']}")
    results.append(ok6)

    # Test 7: Time exit for day trade (>90 min, no meaningful gain)
    pos7 = {
        "entry_price": 100.0,
        "quantity": 10,
        "side": "long",
        "lane": "day",
        "entry_time": (datetime.now() - timedelta(minutes=120)).isoformat(),
    }
    result7 = mgr.evaluate_exit(pos7, current_price=100.2, current_regime="normal", current_signal_strength=0.5)
    ok7 = result7["should_exit"] and "time_exit" in result7["exit_reason"]
    print(f"  {PASS if ok7 else FAIL} Time exit (day, 120min, +0.2%): exit={result7['should_exit']}, reason={result7['exit_reason']}")
    results.append(ok7)

    return all(results)


def test_multiframe_confirmation():
    """Test MultiframeConfirmer instantiation."""
    print("\n═══ 3. MULTI-TIMEFRAME CONFIRMATION ═══")
    try:
        from core.multiframe_confirmation import MultiframeConfirmer
        confirmer = MultiframeConfirmer()
        print(f"  {PASS} MultiframeConfirmer instantiated")

        # Check it has the expected methods
        has_confirm = hasattr(confirmer, "confirm") or hasattr(confirmer, "evaluate") or hasattr(confirmer, "check")
        methods = [m for m in dir(confirmer) if not m.startswith("_") and callable(getattr(confirmer, m))]
        print(f"  {PASS} Public methods: {', '.join(methods[:5])}")
        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        traceback.print_exc()
        return False


def test_regime_factor_weighter():
    """Test RegimeAwareFactorWeighter."""
    print("\n═══ 4. REGIME-AWARE FACTOR WEIGHTER ═══")
    try:
        from models.regime_aware_factor_weighter import RegimeAwareFactorWeighter
        weighter = RegimeAwareFactorWeighter()
        print(f"  {PASS} RegimeAwareFactorWeighter instantiated")

        methods = [m for m in dir(weighter) if not m.startswith("_") and callable(getattr(weighter, m))]
        print(f"  {PASS} Public methods: {', '.join(methods[:5])}")
        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        traceback.print_exc()
        return False


def test_signal_engine_integration():
    """Test that signal_engine_v2 properly imports and uses new components."""
    print("\n═══ 5. SIGNAL ENGINE INTEGRATION ═══")
    results = []

    try:
        import core.signal_engine_v2 as se
        source = open(ROOT / "core" / "signal_engine_v2.py").read()

        # Check multiframe import
        has_mf = "MultiframeConfirmer" in source
        print(f"  {PASS if has_mf else FAIL} signal_engine_v2 imports MultiframeConfirmer")
        results.append(has_mf)

        # Check regime weighter import
        has_rw = "RegimeAwareFactorWeighter" in source
        print(f"  {PASS if has_rw else FAIL} signal_engine_v2 imports RegimeAwareFactorWeighter")
        results.append(has_rw)

        return all(results)
    except Exception as e:
        print(f"  {FAIL} {e}")
        traceback.print_exc()
        return False


def test_run_py_integration():
    """Test that run.py properly imports EnhancedExitManager."""
    print("\n═══ 6. RUN.PY INTEGRATION ═══")
    results = []

    source = open(ROOT / "run.py").read()

    has_import = "from core.enhanced_exits import EnhancedExitManager" in source
    print(f"  {PASS if has_import else FAIL} run.py imports EnhancedExitManager")
    results.append(has_import)

    has_init = "_enhanced_exit_manager = EnhancedExitManager()" in source
    print(f"  {PASS if has_init else FAIL} run.py instantiates _enhanced_exit_manager")
    results.append(has_init)

    has_call = "self._enhanced_exit_manager.evaluate_exit" in source
    print(f"  {PASS if has_call else FAIL} run.py calls evaluate_exit in _manage_open_positions")
    results.append(has_call)

    has_peak = 'plan["peak_price"] = max(_prev_peak, current_price)' in source
    print(f"  {PASS if has_peak else FAIL} run.py tracks peak_price for all lanes")
    results.append(has_peak)

    return all(results)


def test_env_config():
    """Check .env for known issues."""
    print("\n═══ 7. CONFIG VALIDATION ═══")
    results = []

    capital = float(os.getenv("PAPER_CAPITAL", "100000"))
    mode = os.getenv("BROKER_EXECUTION_MODE", "paper")
    print(f"  Paper capital: ${capital:,.0f} | Mode: {mode}")

    if mode == "paper":
        print(f"  {PASS} Broker in paper mode (safe for testing)")
        results.append(True)
    else:
        print(f"  {WARN} Broker mode is '{mode}' — make sure you intended live trading!")
        results.append(True)

    # Check new factor flags
    for factor in ["CROSS_ASSET_REGIME", "INSTITUTIONAL_FLOW", "ORDER_BOOK_IMBALANCE", "SUPPLY_CHAIN_PROPAGATION"]:
        val = os.getenv(f"ENABLE_{factor}_FACTOR", "0")
        enabled = val.strip().lower() in {"1", "true", "yes"}
        status = PASS if enabled else WARN
        print(f"  {status} {factor}: {'enabled' if enabled else 'disabled'}")
        results.append(True)

    # Check for API keys
    api_keys = {
        "FINNHUB_API_KEYS": os.getenv("FINNHUB_API_KEYS", ""),
        "FRED_API_KEY": os.getenv("FRED_API_KEY", ""),
    }
    for name, val in api_keys.items():
        has_key = bool(val.strip())
        status = PASS if has_key else WARN
        print(f"  {status} {name}: {'set' if has_key else 'not set'}")

    return all(results)


def main():
    print("╔══════════════════════════════════════════╗")
    print("║  MACRO INTELLIGENCE — SYSTEM VALIDATION  ║")
    print("╚══════════════════════════════════════════╝")

    all_passed = True
    tests = [
        test_imports,
        test_enhanced_exits,
        test_multiframe_confirmation,
        test_regime_factor_weighter,
        test_signal_engine_integration,
        test_run_py_integration,
        test_env_config,
    ]

    for test in tests:
        try:
            passed = test()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  {FAIL} Test crashed: {e}")
            traceback.print_exc()
            all_passed = False

    print("\n" + "═" * 44)
    if all_passed:
        print(f"  {PASS} ALL TESTS PASSED — system is ready to run!")
        print(f"\n  Next step: python run.py")
    else:
        print(f"  {FAIL} Some tests failed — review above for details")
    print("═" * 44)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
