# Go-Live Strategy — Macro Intelligence System
**Capital: <$5K | Markets: NSE (Upstox) + Crypto | Timeline: 1 Month**

> ⚠️ **Disclaimer:** This is a strategic planning document, not financial advice. Real money trading involves substantial risk of loss. Backtest and paper trading results do not guarantee live performance.

---

## What Your System Already Has (That Most People Don't)

Before anything else — your system is genuinely well-built. The things I'd normally tell someone to add first are already in your codebase:

- **`core/risk_manager.py`** — Full production risk controls: Kelly sizing, daily/weekly/monthly P&L limits, correlation limits, sector exposure, circuit breakers
- **`core/paper_trading.py` + `unified_paper_trading.py`** — Realistic simulation with slippage (0.05%) and commission modeling for both India and Crypto
- **`core/signal_engine_v2.py`** — Regime detection (calm/normal/stressed/crisis) with position multipliers already wired in
- **`core/brokerages.py` → `UpstoxExecutionBroker`** — The live execution adapter exists, just needs to be enabled
- **`core/external_execution.py`** — Shadow execution and dry-run validation layer already built
- **`config/precision_target.env`** — Calibrated for 70% directional accuracy, stop loss 2%, take profit 5%

The gap between where you are and going live is smaller than you think. It's mostly about **3 critical mismatches** between your paper environment and the live environment.

---

## The 3 Critical Mismatches to Fix Before Going Live

### Mismatch 1: Capital Scale ($100K paper → <$5K live)

Your `paper_broker_state.json` shows `"initial_capital": 100000.0`. Your portfolio construction (`InstitutionalPortfolioConstructor`) has:

```python
max_weight = 0.18      # = $18,000 max position on $100K paper
min_weight = 0.03      # = $3,000 min position on $100K paper
max_names = 12
```

At $5K live capital, those same weights become:
- Max position: $900 (0.18 × $5K)
- Min position: $150 (0.03 × $5K)
- Max names at full deployment: 12 positions × avg $400 = $4,800

**At ₹5K USD or ₹4 lakh INR, NSE lot sizes and minimum tick values matter.** A ₹150 position in many NSE stocks is literally 1 share. This is fine — but you need to be aware your `quantity` calculations will often round to 1–3 shares.

**Action before Week 1:**
```python
# In InstitutionalPortfolioConstructor, for live $5K account:
max_names = 5           # not 12 — you can't meaningfully split $5K 12 ways
max_weight = 0.20       # allow slightly larger concentration per name
gross_target_pct = 0.50 # only deploy 50% of capital at first (regime-aware)
```

And in `RiskLimits`:
```python
max_concurrent_positions = 4   # was 5, but at $5K keep it tight
max_position_size_pct = 0.15   # was 0.10, at small capital this is too restrictive
```

---

### Mismatch 2: Market Orders → Limit Orders (Your #1 Slippage Fix)

Your paper trading logs show **all orders are `"order_type": "market"`**. Your simulated slippage is 0.05% (5 bps). In live NSE markets:

| Condition | Realistic Slippage (Market Order) |
|-----------|----------------------------------|
| Normal liquid mid-cap (NSE) | 0.05–0.15% |
| Stressed market (like 2026 recent conditions) | 0.15–0.40% |
| Small-cap or low-volume | 0.30–1.0%+ |

Your signal engine is running in `"regime": "stressed"` (from your paper trade logs). At 0.30% round-trip slippage, a trade targeting 5% take profit has already burned 6% of its edge before you start.

**Action before Week 1:** Add a limit order path to `UpstoxExecutionBroker`. For entries, use limit orders at mid-price + small buffer:

```python
# For BUY entries (add to external_execution.py or brokerages.py):
if order.order_type == "limit":
    limit_price = current_bid + (spread * 0.3)  # 30% into the spread
else:
    # Warn loudly in stressed regimes
    if regime == "stressed":
        logger.warning(f"Market order in STRESSED regime: {symbol}. Consider limit.")
```

**Practical rule for live trading:**
- Liquid NSE large-caps (NIFTY 50 names): limit orders at mid ± 0.05%
- NSE mid-cap or crypto: limit orders at mid ± 0.10%
- Never use market orders in stressed or crisis regime

---

### Mismatch 3: Upstox Adapter is Disabled — Activation Checklist

In `core/brokerages.py`, `UpstoxExecutionBroker` initializes with:
```python
enabled: bool = False
dry_run: bool = True
```

Before going live, you need to work through this checklist in sequence:

**Phase A — Shadow Mode (Week 2–3):**
```python
# Set environment variables (NOT in code):
UPSTOX_ACCESS_TOKEN=your_token_here
UPSTOX_DRY_RUN=true        # shadow mode: logs orders, doesn't send them
UPSTOX_ENABLED=true
```
In shadow mode, your system logs every order it *would* send to Upstox. Compare these against your paper broker output to confirm they match.

**Phase B — Live with Kill Switch (Week 3–4):**
```python
UPSTOX_DRY_RUN=false       # real orders now
MAX_LIVE_POSITION_VALUE=500 # hard cap: no single order > ₹500 equivalent at first
```

The `UpstoxExecutionBroker` already handles instrument token resolution (NSE.json.gz) and the symbol alias map for tickers like `ZOMATO.NS → ETERNAL`. Verify your `data/upstox_instruments.json` is populated before enabling.

---

## Week-by-Week Plan

### Week 1 — Recalibrate for $5K (Days 1–7)

**Goal:** Fix the 3 mismatches above before any live money touches the market.

- [ ] Update `InstitutionalPortfolioConstructor` params for $5K as shown above
- [ ] Update `RiskLimits` in `risk_manager.py` for small account
- [ ] Add limit order logic to your Upstox execution path
- [ ] Set `UPSTOX_ACCESS_TOKEN` and test instrument map resolution
- [ ] Run `external_execution.py` in dry-run against live NSE data for 48 hours
- [ ] Recalibrate `paper_broker_state.json` — reset initial capital to your actual live amount so your P&L tracking is meaningful from day 1

**Regime check:** Your signal engine shows the current regime is `"stressed"`. Per your config:
```
REGIME_STRESSED_POSITION=0.30
```
This means at $5K, you should only have ~$1,500 deployed in week 1. This is correct and the system already enforces it — just make sure you don't override it manually.

---

### Week 2 — Shadow Execution (Days 8–14)

**Goal:** Run live signals through `UpstoxExecutionBroker` in shadow/dry-run mode and compare against paper broker.

Enable shadow mode:
```bash
UPSTOX_ENABLED=true UPSTOX_DRY_RUN=true python run.py
```

For each signal, track:
1. **Signal time** (when `signal_engine_v2.py` generates direction)
2. **Order log time** (when `external_execution.py` would submit to Upstox)
3. **Expected fill price** (from paper broker simulation)
4. **What the real market price was** at that moment (get from WebSocket tick buffer)

The delta between (3) and (4) is your **live slippage estimate** — before you've spent a rupee.

If the delta is consistently >2x your simulated 0.05%, adjust your slippage assumption in `paper_trading.py` before Week 3.

---

### Week 3 — Go Live at 25% Size (Days 15–21)

**Goal:** Real Upstox orders, but at 25% of what your system recommends.

```python
UPSTOX_DRY_RUN=false
LIVE_SIZE_MULTIPLIER=0.25   # Add this to your execution layer
```

Implement `LIVE_SIZE_MULTIPLIER` in `external_execution.py`:
```python
quantity = max(1, int(recommended_quantity * float(os.getenv("LIVE_SIZE_MULTIPLIER", "1.0"))))
```

**Risk gates for Week 3:**
- Daily loss limit: ₹2,000 (your `max_daily_loss_pct=0.02` × $5K = $100 ≈ ₹8,300 — but manually cap lower in week 3)
- If circuit breaker fires: don't restart same day. Review in dashboard.
- Track fill quality: every live fill vs. your paper broker's expected fill

---

### Week 4 — Scale to 50%, Evaluate (Days 22–30)

**Scale up if Week 3 shows:**
- Live fills within 2x of paper slippage assumption (i.e., <0.10% average per trade)
- No circuit breaker violations
- Regime hasn't moved to `"crisis"` (your multiplier goes to 0.10 in crisis — trust it)
- Signal accuracy directionally correct on >55% of closed trades (below your 70% target is expected early on, but 55%+ is needed to not lose money)

**Raise to 50% size:**
```python
LIVE_SIZE_MULTIPLIER=0.50
```

**Go/No-Go at Day 30:**

| Signal | Go Full | Extend Calibration | Pause |
|--------|---------|-------------------|-------|
| Live slippage vs. paper | < 1.5x | 1.5–3x | > 3x |
| Win rate (closed trades) | ≥ 55% | 45–55% | < 45% |
| Max drawdown hit | Never | Once | Multiple |
| Circuit breaker triggered | Never | Once | Multiple |
| Regime at Day 30 | calm/normal | stressed | crisis |

---

## Specific Config Changes for $5K Live Account

Edit `config/precision_target.env`:

```env
# Capital-adjusted risk (was calibrated for $100K paper)
MAX_POSITION_SIZE=0.15           # was 0.10 — too restrictive at $5K
STOP_LOSS_PCT=2.0                # keep as-is ✓
TAKE_PROFIT_PCT=5.0              # keep as-is ✓

# Regime position sizing — these are already correct, don't touch
REGIME_CRISIS_POSITION=0.10
REGIME_STRESSED_POSITION=0.30    # = $1,500 deployed in stressed market ✓
REGIME_NORMAL_POSITION=0.60      # = $3,000 deployed in normal market ✓
REGIME_CALM_POSITION=0.80        # = $4,000 deployed in calm market ✓

# Confidence thresholds — consider raising slightly for live trading
BUY_CONFIDENCE_THRESHOLD=0.65    # was 0.60 — be more selective with real money
SELL_CONFIDENCE_THRESHOLD=0.65   # was 0.60
MIN_CONFIDENCE_FOR_TRADE=0.60    # was 0.55
```

---

## Slippage Budget by Asset Class (Live NSE + Crypto)

| Asset | Entry Slippage Budget | Exit Slippage Budget | Total Round-Trip Budget |
|-------|----------------------|----------------------|------------------------|
| NIFTY 50 stocks (Reliance, TCS, etc.) | 0.05% | 0.05% | 0.10% |
| NSE mid-cap (liquid) | 0.10% | 0.10% | 0.20% |
| BTC/ETH on major exchange | 0.05% | 0.05% | 0.10% |
| Altcoins / low-volume | 0.20% | 0.20% | 0.40% |

Your system's `TAKE_PROFIT_PCT=5.0` and `STOP_LOSS_PCT=2.0` give you enough room even with worst-case slippage on NIFTY 50 stocks. The risk is mid-cap and crypto altcoins — keep those to minimal size in weeks 2–3.

---

## Things You Don't Need to Build (Already Done)

- ❌ Kill switch — `RiskManager.circuit_breaker_triggered` already handles this
- ❌ Drawdown limits — `max_daily_loss_pct`, `max_weekly_loss_pct`, `max_monthly_loss_pct` already in `RiskLimits`
- ❌ Regime-based position scaling — `REGIME_*_POSITION` in config already wired to `signal_engine_v2.py`
- ❌ Slippage simulation — `paper_trading.py` already models it
- ❌ Commission modeling — already in your paper broker
- ❌ Dashboard monitoring — your WebSocket dashboard already exists

---

## One-Page Month Summary

| Week | What You're Doing | Live Money? | Size |
|------|-------------------|-------------|------|
| 1 | Fix 3 mismatches, recalibrate for $5K | No | — |
| 2 | Shadow execution via Upstox dry-run | No | — |
| 3 | First live orders | Yes | 25% of system recommendation |
| 4 | Scale up + evaluate | Yes | 50% → full if go/no-go passes |

---

*Generated after reading: `core/risk_manager.py`, `core/signal_engine_v2.py`, `core/brokerages.py`, `core/external_execution.py`, `core/paper_trading.py`, `core/unified_paper_trading.py`, `core/portfolio_construction.py`, `core/realtime_engine.py`, `config/precision_target.env`, `data/paper_broker_state.json` | April 2026*
