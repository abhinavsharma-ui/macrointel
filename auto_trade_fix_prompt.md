# Fix: Auto-Trade Executing 0 Positions Despite 21+ Trade-Ready Signals

## Context

The system shows 21+ buy signals as trade-ready on the dashboard but `0 open` positions.
The execution log shows no trade attempts at all — `_build_lane_candidates()` returns empty.
The broker is initialized, `_auto_trade()` runs every inference cycle, but nothing executes.

After tracing the full chain, there are **5 independent blockers** any one of which kills all trades.
Fix all 5. Do not change any other logic.

---

## ROOT CAUSE SUMMARY

The call chain is:
```
_auto_trade()
  → _build_lane_candidates()          # filters signals down to executable candidates
      → _iter_lane_signal_items()     # enumerates signals from signal store
      → for each signal:
          1. governor_decision.allow  → skip if False
          2. signal == "buy"          → skip if not buy
          3. conviction >= min_conviction → skip "weak_conviction"
          4. portfolio_target_pct > 0 OR fallback conditions → skip "optimizer_zero_weight"
          5. meta.take_trade == True  → skip with meta reason (THE PRIMARY BLOCKER)
  → _execute_lane_entries()           # only reached if candidates exist
```

The primary blocker is **#5**: when the meta model is in a degraded state
(`mean_precision=0.0`, `mean_taken_edge_pct=0.0`), every signal gets
`take_trade=False` from the engine, and line ~5682 of `run.py` skips them all.
Secondary blockers (#3 and #4) then prevent any fallback from working.

---

## FILE 1 — `project/run.py`

### Fix 1 (CRITICAL): Add degraded-model bypass in `_build_lane_candidates`

In `_build_lane_candidates`, find the block that checks `meta.take_trade` (around line 5682):

```python
if meta and not meta.get("take_trade", False):
    if (
        lane == "crypto"
        and self._event_window_mode
        ...
    ):
        signal["meta_override"] = "event_window_crypto"
    else:
        self._record_execution_event(
            symbol, "buy", "skipped",
            str(meta.get("reason", "meta_skip")),
            ...
        )
        continue
```

Replace the entire `if meta and not meta.get("take_trade", False):` block with:

```python
if meta and not meta.get("take_trade", False):
    # Check if the trained meta model is currently degraded (0% precision/edge).
    # When degraded, fall back to conviction-only gating so the system does not
    # grind to a halt while waiting for a new good checkpoint.
    meta_status = self._components.get("meta_model_status") or {}
    meta_precision = float(meta_status.get("mean_precision") or 0.0)
    meta_edge = float(meta_status.get("mean_taken_edge_pct") or 0.0)
    meta_degraded = meta_precision < 0.05 and meta_edge < 0.05
    degraded_bypass_enabled = os.getenv("META_DEGRADED_BYPASS_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
    degraded_min_conviction = max(0.5, float(os.getenv("META_DEGRADED_BYPASS_MIN_CONVICTION", "1.8")))

    if (
        lane == "crypto"
        and self._event_window_mode
        and take_probability >= (lane_config.get("min_take_probability", 0.0) * 0.85)
        and conviction >= (min_conviction * 0.90)
    ):
        signal["meta_override"] = "event_window_crypto"
    elif meta_degraded and degraded_bypass_enabled and conviction >= degraded_min_conviction:
        # Meta model is producing all-zero predictions — bypass and use conviction gate.
        signal["meta_override"] = "degraded_model_bypass"
        logger.debug(
            f"Meta degraded bypass: {symbol} lane={lane} conviction={conviction:.2f} "
            f"(meta precision={meta_precision:.3f} edge={meta_edge:.3f})"
        )
    else:
        self._record_execution_event(
            symbol,
            "buy",
            "skipped",
            str(meta.get("reason", "meta_skip")),
            signal=signal,
            position_key=position_key,
            score=rank_score,
            conviction=conviction,
        )
        continue
```

### Fix 2 (HIGH): Add optimizer zero-weight bypass env var

Find the `optimizer_zero_weight` skip block (around line 5668):

```python
if construction and portfolio_target_pct <= 0 and not (
    zero_weight_fallback or qualified_edge_fallback or intraday_daytrade_override
):
    self._record_execution_event(symbol, "buy", "skipped", "optimizer_zero_weight", ...)
    continue
```

Replace with:

```python
optimizer_bypass_enabled = os.getenv("OPTIMIZER_ZERO_WEIGHT_BYPASS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
if construction and portfolio_target_pct <= 0 and not (
    zero_weight_fallback or qualified_edge_fallback or intraday_daytrade_override or optimizer_bypass_enabled
):
    self._record_execution_event(symbol, "buy", "skipped", "optimizer_zero_weight", ...)
    continue
```

### Fix 3 (HIGH): Add `/api/trade_debug` diagnostic endpoint

In `dashboard/app.py`, add a new route that exposes the real-time blocking reason
for every signal currently in the store. Add this after the existing API routes:

```python
@app.route("/api/trade_debug")
def trade_debug():
    """Show why each trade-ready signal is or isn't being executed."""
    import os
    signal_store = _signal_store or {}
    rows = []
    for symbol, signal in list(signal_store.items()):
        if not isinstance(signal, dict):
            continue
        lane = str(signal.get("lane") or "normal").lower()
        meta = signal.get("meta_decision") or {}
        construction = signal.get("portfolio_construction") or {}
        rows.append({
            "symbol": symbol,
            "lane": lane,
            "signal_direction": signal.get("signal"),
            "conviction": round(float(signal.get("conviction_score", 0.0) or 0.0), 3),
            "trade_eligible": signal.get("trade_eligible", False),
            "take_trade": meta.get("take_trade", False),
            "take_probability": round(float(meta.get("take_probability", 0.0) or 0.0), 4),
            "meta_reason": meta.get("reason", ""),
            "meta_source": meta.get("source", ""),
            "expected_edge_pct": round(float(meta.get("expected_edge_pct", 0.0) or 0.0), 3),
            "portfolio_target_pct": round(float(construction.get("target_position_pct", 0.0) or 0.0), 4),
            "entry_readiness": (signal.get("entry_readiness") or {}).get("reason", ""),
            "entry_allowed": (signal.get("entry_readiness") or {}).get("allow", True),
            "meta_override": signal.get("meta_override", ""),
            "governor_allow": (signal.get("governor_decision") or {}).get("allow", True),
            "governor_reason": (signal.get("governor_decision") or {}).get("reason", ""),
        })

    trade_ready = [r for r in rows if r["trade_eligible"] or r["take_trade"]]
    buy_signals = [r for r in rows if r["signal_direction"] == "buy"]
    blocked_by_meta = [r for r in buy_signals if not r["take_trade"]]
    blocked_by_readiness = [r for r in buy_signals if not r["entry_allowed"]]
    blocked_by_governor = [r for r in buy_signals if not r["governor_allow"]]
    zero_weight = [r for r in buy_signals if r["portfolio_target_pct"] <= 0]

    return jsonify({
        "summary": {
            "total_signals": len(rows),
            "buy_signals": len(buy_signals),
            "trade_ready": len(trade_ready),
            "blocked_by_meta_skip": len(blocked_by_meta),
            "blocked_by_entry_readiness": len(blocked_by_readiness),
            "blocked_by_governor": len(blocked_by_governor),
            "blocked_by_zero_weight": len(zero_weight),
        },
        "buy_signals_sample": sorted(buy_signals, key=lambda r: r["conviction"], reverse=True)[:20],
    })
```

---

## FILE 2 — `project/.env`

### Fix 4 (HIGH): Lower conviction threshold

Change:
```
AUTO_TRADE_MIN_CONVICTION=2.6
```
To:
```
AUTO_TRADE_MIN_CONVICTION=1.5
```

Also add these new env vars (append to end of .env):
```
# Meta degraded bypass — allows conviction-gated trades when trained model has 0% precision
META_DEGRADED_BYPASS_ENABLED=1
META_DEGRADED_BYPASS_MIN_CONVICTION=1.8

# Set to 1 temporarily if optimizer is zeroing out all weights
OPTIMIZER_ZERO_WEIGHT_BYPASS_ENABLED=0
```

---

## FILE 3 — `project/core/signal_engine_v2.py`

### Fix 5 (MEDIUM): Ensure heuristic path produces `take_trade=True` for strong signals

Find `_conviction_floor` and `_min_edge_pct` initialization. Verify the heuristic path
can produce `take_trade=True` when the trained model is not available.

In the `evaluate_signal` method, find where `take_trade` is computed (around line 972):

```python
take_trade = (
    take_prob >= effective_threshold
    and expected_edge_pct > min_edge_required
    and edge_ratio_live >= min_ratio_required
    and conviction >= min_conviction
)
```

Add an env-var-controlled conviction-only override for when the trained model is absent
and heuristic thresholds are too tight:

```python
heuristic_conviction_override = (
    decision_source == "heuristic"
    and os.getenv("HEURISTIC_CONVICTION_OVERRIDE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    and conviction >= float(os.getenv("HEURISTIC_CONVICTION_OVERRIDE_FLOOR", "2.5"))
    and direction != "neutral"
    and take_prob >= 0.30
)
take_trade = (
    take_trade or heuristic_conviction_override
)
if heuristic_conviction_override and not (
    take_prob >= effective_threshold
    and expected_edge_pct > min_edge_required
):
    reason = "heuristic_conviction_override"
```

Also add to `.env`:
```
# Override: use conviction alone when trained model is absent (heuristic mode only)
HEURISTIC_CONVICTION_OVERRIDE_ENABLED=0
HEURISTIC_CONVICTION_OVERRIDE_FLOOR=2.5
```

---

## Verification

After applying all fixes, restart the system and check:

1. **Diagnose immediately** — hit the new endpoint:
   ```bash
   curl -s http://localhost:8888/api/trade_debug | python3 -m json.tool
   ```
   The `summary` block will show exactly which gate is blocking how many signals.

2. **Confirm meta bypass is firing** — check logs:
   ```bash
   grep "degraded_model_bypass\|meta_degraded" logs/system.jsonl | tail -10
   ```

3. **Confirm trades open** — within 2-3 inference cycles (60-90 seconds),
   the dashboard should show `X open` > 0 on Crypto Scalper.

4. **If still 0 after fix 1-4**, set `OPTIMIZER_ZERO_WEIGHT_BYPASS_ENABLED=1` in .env
   and restart — this forces trades through even if the optimizer gives 0% allocation.

5. **If still 0**, enable heuristic override:
   `HEURISTIC_CONVICTION_OVERRIDE_ENABLED=1` and `HEURISTIC_CONVICTION_OVERRIDE_FLOOR=2.0`

6. Once trades flow, revert `OPTIMIZER_ZERO_WEIGHT_BYPASS_ENABLED` to `0` so the
   portfolio optimizer resumes controlling position sizing correctly.
