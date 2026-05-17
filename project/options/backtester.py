"""
Options Backtester — Real Historical Data
==========================================
Replays the signal system against real Kaggle historical options chain data
(S&P 500 Daily Options 2010-2023) to measure actual edge.

Data source (free, one-time download):
    import kagglehub
    path = kagglehub.dataset_download("shubhamcodez/s-and-p-500-daily-options-data-2010-2023")

The Kaggle CSV schema (typical):
    QUOTE_DATE, EXPIRE_DATE, UNDERLYING_LAST, STRIKE, C_IV, C_DELTA, C_BID,
    C_ASK, C_VOLUME, C_OPEN_INT, P_IV, P_DELTA, P_BID, P_ASK, P_VOLUME, P_OPEN_INT

Usage:
    # point at the downloaded CSV
    python -m options.backtester --data /path/to/spy_options.csv --symbol SPY

    # or from Python
    from options.backtester import OptionsBacktester
    bt = OptionsBacktester("/path/to/spy_options.csv")
    results = bt.run_backtest(symbol="SPY", start="2020-01-01", end="2023-12-31")
    bt.print_report(results)

What it tests:
  ✓ IV rank gate — does filtering by IVR < 60% improve win rate?
  ✓ Strike selection — how does the 60%-OTM rule perform vs ATM?
  ✓ Exit rules — 2× profit, 50% stop, day-8 exit performance
  ✓ Expected-move filter — what happens when underlying moves < 3%?
  ✓ Full P&L in dollar terms with Kelly sizing
"""

import argparse
import glob
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent so we can import our own modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from options.black_scholes import price_option, implied_volatility

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Signal simulation — we don't have the ML model's historical predictions,
# so we simulate signals via two methods:
#   1. MOMENTUM signals: stock up >3% in prior 5 days → simulate "bullish call"
#   2. SYNTHETIC signals: randomly sample 10% of trading days (stress test)

SIGNAL_MODE           = "momentum"   # "momentum" | "synthetic"
MOMENTUM_LOOKBACK     = 5            # days of prior price history to check
MOMENTUM_THRESHOLD    = 0.03         # 3% gain in prior 5 days → buy call

# IV rank gate (replicated from options_trader.py)
MAX_IVR_ENTRY         = 60.0         # skip if IV rank > 60%

# Strike selection
OTM_RATIO             = 0.60         # how far OTM as fraction of expected move
EXPECTED_MOVE_PCT     = 0.07         # 7% — matches your stock system target
MIN_DELTA             = 0.20
MAX_DELTA             = 0.55

# Expiry selection
# NOTE: 2010-2013 SPY had monthly options only — nearest expiry is often 20-35 DTE.
# Widen MAX_DTE to 45 so we don't filter out all available contracts.
TARGET_DTE            = 20           # relaxed for monthly-only era
MIN_DTE               = 5
MAX_DTE               = 45

# Exit rules
PROFIT_TAKE_MULT      = 2.0          # 2× premium → take profit
STOP_LOSS_MULT        = 0.50         # 50% loss → stop
MAX_HOLD_DAYS         = 8
MIN_DTE_EXIT          = 3            # exit if < 3 DTE remaining

# Position sizing
PORTFOLIO_VALUE       = 50_000.0
MAX_RISK_PCT          = 0.02         # 2% of portfolio per trade
KELLY_FRACTION        = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol:       str
    entry_date:   str
    expiry_date:  str
    strike:       float
    option_type:  str          # 'call' | 'put'
    entry_price:  float        # option premium paid
    contracts:    int          # number of contracts (100 shares each)
    entry_stock:  float        # underlying price at entry
    iv_at_entry:  float        # IV% at entry
    iv_rank:      float        # IVR% at entry
    dte_at_entry: int

    exit_date:    str   = ""
    exit_price:   float = 0.0
    exit_reason:  str   = ""   # "profit_take" | "stop_loss" | "time_exit" | "dte_exit"
    pnl_dollars:  float = 0.0
    pnl_pct:      float = 0.0
    underlying_move_pct: float = 0.0

@dataclass
class BacktestResults:
    trades:          List[BacktestTrade] = field(default_factory=list)
    total_trades:    int   = 0
    winning_trades:  int   = 0
    losing_trades:   int   = 0
    total_pnl:       float = 0.0
    avg_pnl:         float = 0.0
    win_rate:        float = 0.0
    profit_factor:   float = 0.0
    avg_winner:      float = 0.0
    avg_loser:       float = 0.0
    max_drawdown:    float = 0.0
    sharpe:          float = 0.0
    # breakdown by exit reason
    profit_takes:    int   = 0
    stop_losses:     int   = 0
    time_exits:      int   = 0
    dte_exits:       int   = 0
    # IV filter stats
    signals_total:   int   = 0
    signals_blocked_iv: int = 0
    signals_blocked_liquidity: int = 0
    signals_executed:  int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Core backtester
# ─────────────────────────────────────────────────────────────────────────────

class OptionsBacktester:
    """
    Replays the options strategy against real historical chains.

    The CSV must have at minimum:
        QUOTE_DATE, EXPIRE_DATE, UNDERLYING_LAST, STRIKE,
        C_IV, C_DELTA, C_BID, C_ASK, C_VOLUME, C_OPEN_INT
    """

    # Column name aliases — handles variations in Kaggle dataset naming
    COL_ALIASES = {
        "quote_date":    ["QUOTE_DATE", "quote_date", "QuoteDate", "date"],
        "expire_date":   ["EXPIRE_DATE", "expire_date", "ExpiryDate", "expiry"],
        "underlying":    ["UNDERLYING_LAST", "underlying_last", "S", "stock_price", "close"],
        "strike":        ["STRIKE", "strike", "Strike", "STRIKE_PRICE"],
        "c_iv":          ["C_IV", "call_iv", "impl_volatility", "IV"],
        "c_delta":       ["C_DELTA", "call_delta", "delta"],
        "c_bid":         ["C_BID", "call_bid", "bid"],
        "c_ask":         ["C_ASK", "call_ask", "ask"],
        "c_volume":      ["C_VOLUME", "call_volume", "volume"],
        "c_oi":          ["C_OPEN_INT", "call_oi", "open_interest", "OI"],
    }

    def __init__(self, data_path: str, symbol: str = "SPY"):
        self.data_path = data_path
        self.symbol    = symbol
        self.df: Optional[pd.DataFrame] = None
        self._iv_history: Dict[str, List[float]] = {}   # date → list of IVs

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load and normalise the Kaggle CSV."""
        path = Path(self.data_path)
        if path.is_dir():
            # auto-detect CSV files in directory
            csvs = list(path.glob("*.csv"))
            if not csvs:
                raise FileNotFoundError(f"No CSV files found in {path}")
            # pick the largest file (most data)
            path = max(csvs, key=lambda p: p.stat().st_size)
            logger.info(f"Auto-selected: {path.name} ({path.stat().st_size/1e6:.1f} MB)")

        logger.info(f"Loading {path} …")
        self.df = pd.read_csv(path, low_memory=False)
        logger.info(f"  Raw rows: {len(self.df):,}")

        self._normalise_columns()
        self._parse_dates()
        self._clean()

        logger.info(f"  After cleaning: {len(self.df):,} rows")
        logger.info(f"  Date range: {self.df['quote_date'].min()} → {self.df['quote_date'].max()}")
        logger.info(f"  Unique expiries: {self.df['expire_date'].nunique()}")
        logger.info(f"  Strikes: {self.df['strike'].min():.0f} – {self.df['strike'].max():.0f}")
        logger.info(f"  Columns: {list(self.df.columns)}")

        # Show DTE distribution for first quote date — critical for tuning MAX_DTE
        first_date = self.df["quote_date"].min()
        sample = self.df[self.df["quote_date"] == first_date].copy()
        sample["dte"] = (sample["expire_date"] - first_date).dt.days
        dte_vals = sorted(sample["dte"].unique())
        logger.info(f"  Sample DTEs on {first_date.date()}: {dte_vals[:10]}")

        # Check if delta column has real values
        if "c_delta" in self.df.columns:
            nonzero_delta = (self.df["c_delta"].abs() > 0.01).sum()
            logger.info(f"  Delta non-zero rows: {nonzero_delta:,} / {len(self.df):,}")

    def _normalise_columns(self) -> None:
        """Map whatever column names the CSV has → our standard names."""
        col_map = {}
        existing = set(self.df.columns)
        for std_name, aliases in self.COL_ALIASES.items():
            for alias in aliases:
                if alias in existing:
                    col_map[alias] = std_name
                    break
        self.df.rename(columns=col_map, inplace=True)

        # check required columns present
        required = ["quote_date", "expire_date", "underlying", "strike",
                    "c_iv", "c_delta", "c_bid", "c_ask"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            logger.warning(f"Available columns: {list(self.df.columns)}")
            # try to survive with what we have

    def _parse_dates(self) -> None:
        for col in ["quote_date", "expire_date"]:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
        self.df.dropna(subset=["quote_date"], inplace=True)

    def _clean(self) -> None:
        """Remove obviously bad rows."""
        df = self.df
        n0 = len(df)

        # numeric coerce
        for col in ["underlying", "strike", "c_iv", "c_delta", "c_bid", "c_ask"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # remove NaN fundamentals
        df.dropna(subset=["underlying", "strike", "c_iv"], inplace=True)
        logger.debug(f"  After NaN drop: {len(df):,} (removed {n0-len(df):,})")

        # remove zero/negative prices
        n1 = len(df)
        df = df[(df["underlying"] > 0) & (df["strike"] > 0)]
        df = df[(df["c_bid"] >= 0) & (df["c_ask"] > 0)]
        logger.debug(f"  After price filter: {len(df):,} (removed {n1-len(df):,})")

        # remove IV artefacts (IV > 500% is a data error)
        # Note: Kaggle SPY data may store IV as decimal (0.25 = 25%) or percent (25.0)
        # detect which format: if median > 2.0, it's in percent form — divide by 100
        n2 = len(df)
        iv_median = df["c_iv"].median()
        if iv_median > 2.0:
            logger.info(f"  IV appears to be in percent form (median={iv_median:.1f}) — converting to decimal")
            df = df.copy()
            df["c_iv"] = df["c_iv"] / 100.0
        df = df[(df["c_iv"] > 0.01) & (df["c_iv"] < 5.0)]
        logger.debug(f"  After IV filter: {len(df):,} (removed {n2-len(df):,})")

        # remove rows where ask/bid spread is > 80% of ask
        # (relaxed from 50% — 2010-era options had wider spreads)
        n3 = len(df)
        if "c_bid" in df.columns:
            spread_pct = (df["c_ask"] - df["c_bid"]) / df["c_ask"].clip(lower=0.01)
            df = df[spread_pct < 0.80]
        logger.debug(f"  After spread filter: {len(df):,} (removed {n3-len(df):,})")

        self.df = df.reset_index(drop=True)

    # ── IV Rank calculation ───────────────────────────────────────────────────

    def _build_iv_history(self) -> None:
        """
        For each quote date, compute the median ATM IV across all contracts
        for that day. This is our proxy for the 'daily IV' of the underlying.
        Used to compute IV rank over a rolling 252-day window.
        """
        logger.info("Building IV history (may take a moment)…")
        daily_iv = (
            self.df.groupby("quote_date")["c_iv"]
            .median()
            .sort_index()
        )
        self._daily_iv_series = daily_iv
        logger.info(f"  IV history: {len(daily_iv)} trading days")

    def _get_iv_rank(self, quote_date: pd.Timestamp) -> Tuple[float, float]:
        """
        Returns (iv_rank_pct, iv_percentile_pct) for the given date.
        Uses the prior 252 trading days.
        """
        series = self._daily_iv_series
        idx = series.index.searchsorted(quote_date)
        if idx < 20:
            return 50.0, 50.0   # not enough history yet

        window = series.iloc[max(0, idx - 252): idx]
        current_iv = series.get(quote_date, window.iloc[-1])

        iv_high = window.max()
        iv_low  = window.min()
        if iv_high <= iv_low:
            return 50.0, 50.0

        iv_rank = (current_iv - iv_low) / (iv_high - iv_low) * 100.0
        iv_pct  = (window < current_iv).mean() * 100.0
        return float(iv_rank), float(iv_pct)

    # ── Signal generation (simulated) ────────────────────────────────────────

    def _generate_signals(
        self, start: str, end: str
    ) -> List[Tuple[pd.Timestamp, str]]:
        """
        Returns list of (date, direction) signal tuples in the date range.

        MOMENTUM mode: if underlying rose > MOMENTUM_THRESHOLD over prior
        MOMENTUM_LOOKBACK days, emit a 'call' signal.
        """
        dates = sorted(self.df["quote_date"].unique())
        dates = [d for d in dates if start <= str(d.date()) <= end]

        if not dates:
            data_start = str(self.df["quote_date"].min().date())
            data_end   = str(self.df["quote_date"].max().date())
            logger.error(
                f"No trading dates found in range {start} → {end}.\n"
                f"  Dataset covers: {data_start} → {data_end}\n"
                f"  Fix: pass --start {data_start[:7]}-01 --end {data_end}"
            )
            return []

        # get daily underlying price — use the first row per day (all rows on
        # same day have the same underlying_last, so any agg works; first is fastest)
        daily_price = (
            self.df.groupby("quote_date")["underlying"]
            .first()      # not median — underlying_last is same for all rows on a date
            .sort_index()
        )

        signals = []
        for i, dt in enumerate(dates):
            if SIGNAL_MODE == "momentum":
                # need prior LOOKBACK days
                lookback_date = dt - timedelta(days=MOMENTUM_LOOKBACK * 2)
                prior = daily_price.loc[
                    (daily_price.index >= lookback_date) &
                    (daily_price.index < dt)
                ]
                if len(prior) < MOMENTUM_LOOKBACK:
                    continue
                prior_price = prior.iloc[-MOMENTUM_LOOKBACK]
                current     = daily_price.get(dt, None)
                if current is None:
                    continue
                move = (current - prior_price) / prior_price
                if move >= MOMENTUM_THRESHOLD:
                    signals.append((dt, "call"))
                elif move <= -MOMENTUM_THRESHOLD:
                    signals.append((dt, "put"))
            elif SIGNAL_MODE == "synthetic":
                if np.random.random() < 0.10:
                    direction = "call" if np.random.random() > 0.3 else "put"
                    signals.append((dt, direction))

        logger.info(f"Generated {len(signals)} {SIGNAL_MODE} signals ({start} → {end})")
        return signals

    # ── Contract selection ────────────────────────────────────────────────────

    def _select_contract(
        self,
        quote_date: pd.Timestamp,
        direction: str,   # 'call' | 'put'
        underlying_price: float,
    ) -> Optional[pd.Series]:
        """
        From the real chain on quote_date, pick the best contract matching
        our strike/DTE/delta criteria.
        """
        chain = self.df[self.df["quote_date"] == quote_date].copy()
        if chain.empty:
            return None

        # compute DTE
        chain["dte"] = (chain["expire_date"] - quote_date).dt.days

        # filter DTE window
        n_before_dte = len(chain)
        chain = chain[(chain["dte"] >= MIN_DTE) & (chain["dte"] <= MAX_DTE)]
        if chain.empty:
            # diagnostic: show what DTEs were available
            all_chain = self.df[self.df["quote_date"] == quote_date].copy()
            all_chain["dte"] = (all_chain["expire_date"] - quote_date).dt.days
            available_dtes = sorted(all_chain["dte"].unique())
            logger.debug(f"  {str(quote_date.date())}: {n_before_dte} contracts, "
                         f"no DTE in [{MIN_DTE},{MAX_DTE}]. Available: {available_dtes[:8]}")
            return None

        # target strike: OTM by 60% of 7% expected move
        otm_pct    = EXPECTED_MOVE_PCT * OTM_RATIO
        if direction == "call":
            target_strike = underlying_price * (1 + otm_pct)
        else:
            target_strike = underlying_price * (1 - otm_pct)

        # compute mid price
        chain = chain.copy()
        chain["mid"] = (chain["c_bid"] + chain["c_ask"]) / 2.0
        chain = chain[chain["mid"] > 0.01]   # skip sub-penny contracts (relaxed from 0.05)

        if chain.empty:
            return None

        # delta filter — ONLY apply if delta column has real non-zero values
        # (Kaggle dataset may not include delta, or may be all zeros)
        if "c_delta" in chain.columns:
            delta_vals = chain["c_delta"].abs()
            has_real_deltas = (delta_vals > 0.01).sum() > len(chain) * 0.5
            if has_real_deltas:
                if direction == "call":
                    chain = chain[(delta_vals >= MIN_DELTA) & (delta_vals <= MAX_DELTA)]
                else:
                    chain = chain[(delta_vals >= MIN_DELTA) & (delta_vals <= MAX_DELTA)]
            else:
                logger.debug(f"  Delta column present but mostly zero — skipping delta filter")

        if chain.empty:
            return None

        # scoring: proximity to target strike (40%), DTE proximity (30%), liquidity (30%)
        chain["strike_score"] = 1.0 - (
            (chain["strike"] - target_strike).abs() / underlying_price
        ).clip(0, 1)

        target_dte = TARGET_DTE
        chain["dte_score"] = 1.0 - (
            (chain["dte"] - target_dte).abs() / target_dte
        ).clip(0, 1)

        if "c_oi" in chain.columns:
            oi_norm = chain["c_oi"] / chain["c_oi"].max().clip(lower=1)
            chain["liq_score"] = oi_norm.clip(0, 1)
        else:
            chain["liq_score"] = 0.5

        chain["total_score"] = (
            0.40 * chain["strike_score"] +
            0.30 * chain["dte_score"]    +
            0.30 * chain["liq_score"]
        )

        best = chain.nlargest(1, "total_score")
        if best.empty:
            return None
        return best.iloc[0]

    # ── Exit evaluation ───────────────────────────────────────────────────────

    def _bsm_price(
        self,
        trade: BacktestTrade,
        check_date: pd.Timestamp,
        current_stock: float,
    ) -> Optional[float]:
        """
        Synthetic option price via Black-Scholes when the contract isn't
        in the chain on check_date. Uses entry IV (held constant) as sigma.
        This lets us apply exit rules even between chain snapshots.
        """
        try:
            dte = (pd.Timestamp(trade.expiry_date) - check_date).days
            if dte <= 0:
                return 0.0
            sigma = max(trade.iv_at_entry / 100.0, 0.05)
            result = price_option(
                S=current_stock,
                K=trade.strike,
                T=dte,
                sigma=sigma,
                option_type=trade.option_type,
            )
            return result.price
        except Exception:
            return None

    def _evaluate_exit(
        self,
        trade: BacktestTrade,
        check_date: pd.Timestamp,
    ) -> Tuple[bool, float, str]:
        """
        Given a live trade and a check date, return (should_exit, current_price, reason).

        Priority:
          1. Look up real bid/ask from the historical chain (most accurate)
          2. Fall back to BSM synthetic price if contract not in chain that day
             (dataset is sparse — only ~1 snapshot per 3 days in 2010-2013)
          3. Always apply hold_days exit regardless of price source
        """
        dte       = (pd.Timestamp(trade.expiry_date) - check_date).days
        hold_days = (check_date - pd.Timestamp(trade.entry_date)).days

        # ── Try real chain first ───────────────────────────────────────────
        mid = None
        chain = self.df[self.df["quote_date"] == check_date]
        if not chain.empty:
            contract = chain[
                (chain["expire_date"] == pd.Timestamp(trade.expiry_date)) &
                (chain["strike"].round(1) == round(trade.strike, 1))
            ]
            if not contract.empty:
                row = contract.iloc[0]
                raw_mid = (row["c_bid"] + row["c_ask"]) / 2.0
                if raw_mid > 0:
                    mid = raw_mid

        # ── Fall back to BSM if contract not found in chain ───────────────
        if mid is None:
            # get current underlying from any row on this date
            day_rows = self.df[self.df["quote_date"] == check_date]
            if not day_rows.empty:
                current_stock = float(day_rows["underlying"].iloc[0])
                mid = self._bsm_price(trade, check_date, current_stock)

        # ── If still no price, can only apply calendar-based exits ────────
        if mid is None:
            # DTE exit (can check without a price)
            if dte <= MIN_DTE_EXIT:
                return True, 0.0, "dte_exit"
            # Day-8 time exit — exit at last known price (entry price as proxy)
            if hold_days >= MAX_HOLD_DAYS:
                return True, trade.entry_price, "time_exit"
            return False, trade.entry_price, ""

        # ── Apply all exit rules ──────────────────────────────────────────
        # 1. Profit take: 2× premium
        if mid >= trade.entry_price * PROFIT_TAKE_MULT:
            return True, mid, "profit_take"

        # 2. Stop loss: 50% of premium gone
        if mid <= trade.entry_price * STOP_LOSS_MULT:
            return True, mid, "stop_loss"

        # 3. DTE protection
        if dte <= MIN_DTE_EXIT:
            return True, mid, "dte_exit"

        # 4. Day 8 time exit
        if hold_days >= MAX_HOLD_DAYS:
            return True, mid, "time_exit"

        return False, mid, ""

    # ── Main backtest loop ────────────────────────────────────────────────────

    def run_backtest(
        self,
        symbol: str = "SPY",
        start:  str = "2018-01-01",
        end:    str = "2023-12-31",
    ) -> BacktestResults:

        if self.df is None:
            self.load()
        self._build_iv_history()

        results = BacktestResults()
        signals = self._generate_signals(start, end)
        results.signals_total = len(signals)

        # Build signal lookup: date → direction (for O(1) entry check per day)
        signal_map: Dict[pd.Timestamp, str] = {dt: d for dt, d in signals}

        open_trades: List[BacktestTrade] = []

        # ALL trading dates in range — exits checked EVERY day, not just signal days
        trading_dates = sorted(self.df["quote_date"].unique())
        trading_dates = [d for d in trading_dates if start <= str(d.date()) <= end]

        # daily equity curve for drawdown / Sharpe
        daily_pnl: List[float] = []

        for today in trading_dates:

            # ── 1. Check exits for all open trades (runs EVERY trading day) ──
            still_open = []
            day_pnl = 0.0
            for trade in open_trades:
                exited, exit_px, reason = self._evaluate_exit(trade, today)
                if exited:
                    trade.exit_date   = str(today.date())
                    trade.exit_price  = exit_px
                    trade.exit_reason = reason
                    trade.pnl_dollars = (exit_px - trade.entry_price) * 100 * trade.contracts
                    trade.pnl_pct     = (exit_px - trade.entry_price) / trade.entry_price * 100

                    # record underlying move
                    day_rows = self.df[self.df["quote_date"] == today]
                    if not day_rows.empty:
                        exit_stock = float(day_rows["underlying"].iloc[0])
                        trade.underlying_move_pct = (
                            (exit_stock - trade.entry_stock) / trade.entry_stock * 100
                        )

                    results.trades.append(trade)
                    day_pnl += trade.pnl_dollars

                    if trade.pnl_dollars > 0:
                        results.winning_trades += 1
                    else:
                        results.losing_trades  += 1

                    if reason == "profit_take": results.profit_takes += 1
                    elif reason == "stop_loss": results.stop_losses  += 1
                    elif reason == "time_exit": results.time_exits   += 1
                    elif reason == "dte_exit":  results.dte_exits    += 1
                else:
                    still_open.append(trade)
            open_trades = still_open
            daily_pnl.append(day_pnl)

            # ── 2. Evaluate new entry — only on signal days ────────────────
            if today not in signal_map:
                continue
            direction = signal_map[today]

            # Max positions check
            if len(open_trades) >= 5:
                continue

            # IV rank gate
            iv_rank, iv_pct = self._get_iv_rank(today)
            if iv_rank > MAX_IVR_ENTRY:
                results.signals_blocked_iv += 1
                continue

            # Get underlying price
            day_rows = self.df[self.df["quote_date"] == today]
            if day_rows.empty:
                continue
            daily_px = float(day_rows["underlying"].iloc[0])
            if pd.isna(daily_px) or daily_px <= 0:
                continue

            # Select contract
            contract = self._select_contract(today, direction, daily_px)
            if contract is None:
                results.signals_blocked_liquidity += 1
                continue

            # Kelly sizing
            premium   = (contract["c_bid"] + contract["c_ask"]) / 2.0
            max_risk  = PORTFOLIO_VALUE * MAX_RISK_PCT
            contracts = max(1, int((max_risk * KELLY_FRACTION) / (premium * 100)))
            contracts = min(contracts, 10)

            trade = BacktestTrade(
                symbol       = symbol,
                entry_date   = str(today.date()),
                expiry_date  = str(contract["expire_date"].date()),
                strike       = float(contract["strike"]),
                option_type  = direction,
                entry_price  = premium,
                contracts    = contracts,
                entry_stock  = daily_px,
                iv_at_entry  = float(contract["c_iv"]) * 100,
                iv_rank      = iv_rank,
                dte_at_entry = int(contract["dte"]),
            )
            open_trades.append(trade)
            results.signals_executed += 1

        # ── Force-close any remaining open trades at last available date ──
        if trading_dates:
            last_date = max(d for d in trading_dates if str(d.date()) <= end)
            for trade in open_trades:
                exited, exit_px, reason = self._evaluate_exit(trade, last_date)
                if not exited:
                    # close at last known mid or entry (no gain/loss)
                    exit_px = trade.entry_price
                    reason  = "end_of_backtest"
                trade.exit_date   = str(last_date.date())
                trade.exit_price  = exit_px
                trade.exit_reason = reason
                trade.pnl_dollars = (exit_px - trade.entry_price) * 100 * trade.contracts
                trade.pnl_pct     = (exit_px - trade.entry_price) / trade.entry_price * 100

                # record underlying move for closed trades
                entry_px  = trade.entry_stock
                exit_rows = self.df[self.df["quote_date"] == last_date]
                exit_chain = float(exit_rows["underlying"].iloc[0]) if not exit_rows.empty else float("nan")
                if not pd.isna(exit_chain):
                    trade.underlying_move_pct = (exit_chain - entry_px) / entry_px * 100

                results.trades.append(trade)
                if trade.pnl_dollars > 0:
                    results.winning_trades += 1
                else:
                    results.losing_trades += 1

        # ── Compute aggregate stats ────────────────────────────────────────
        results.total_trades = len(results.trades)
        if results.total_trades == 0:
            logger.warning("No trades executed — check data path and date range.")
            return results

        pnls = [t.pnl_dollars for t in results.trades]
        results.total_pnl   = sum(pnls)
        results.avg_pnl     = results.total_pnl / results.total_trades
        results.win_rate    = results.winning_trades / results.total_trades * 100

        winners = [p for p in pnls if p > 0]
        losers  = [p for p in pnls if p <= 0]
        results.avg_winner  = sum(winners) / len(winners) if winners else 0.0
        results.avg_loser   = sum(losers)  / len(losers)  if losers  else 0.0
        results.profit_factor = (
            abs(sum(winners)) / abs(sum(losers))
            if losers else float("inf")
        )

        # max drawdown
        equity = np.cumsum(pnls)
        peak   = np.maximum.accumulate(equity)
        dd     = equity - peak
        results.max_drawdown = float(dd.min()) if len(dd) else 0.0

        # Sharpe (annualised, assuming ~252 trading days, using daily pnl)
        if len(daily_pnl) > 2:
            arr = np.array(daily_pnl)
            mean_d = arr.mean()
            std_d  = arr.std()
            results.sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0

        return results

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self, results: BacktestResults) -> None:
        r = results
        sep = "─" * 60

        print(f"\n{'═'*60}")
        print(f"  OPTIONS BACKTEST RESULTS  —  {self.symbol}")
        print(f"{'═'*60}")

        print(f"\n── Signal Funnel {'─'*43}")
        print(f"  Signals generated     : {r.signals_total:>6}")
        print(f"  Blocked by IV rank    : {r.signals_blocked_iv:>6}  ({r.signals_blocked_iv/max(r.signals_total,1)*100:.1f}%)")
        print(f"  Blocked by liquidity  : {r.signals_blocked_liquidity:>6}  ({r.signals_blocked_liquidity/max(r.signals_total,1)*100:.1f}%)")
        print(f"  Executed trades       : {r.signals_executed:>6}  ({r.signals_executed/max(r.signals_total,1)*100:.1f}%)")

        print(f"\n── Trade Summary {'─'*43}")
        print(f"  Total trades          : {r.total_trades:>6}")
        print(f"  Winning trades        : {r.winning_trades:>6}")
        print(f"  Losing trades         : {r.losing_trades:>6}")
        print(f"  Win rate              : {r.win_rate:>6.1f}%")

        print(f"\n── P&L Summary {'─'*45}")
        print(f"  Total P&L             : ${r.total_pnl:>+10,.2f}")
        print(f"  Avg P&L per trade     : ${r.avg_pnl:>+10,.2f}")
        print(f"  Avg winner            : ${r.avg_winner:>+10,.2f}")
        print(f"  Avg loser             : ${r.avg_loser:>+10,.2f}")
        print(f"  Profit factor         : {r.profit_factor:>10.2f}x")
        print(f"  Max drawdown          : ${r.max_drawdown:>10,.2f}")
        print(f"  Sharpe ratio          : {r.sharpe:>10.2f}")

        print(f"\n── Exit Reason Breakdown {'─'*35}")
        total_closed = r.profit_takes + r.stop_losses + r.time_exits + r.dte_exits
        if total_closed > 0:
            print(f"  Profit take (2×)      : {r.profit_takes:>4}  ({r.profit_takes/total_closed*100:.1f}%)")
            print(f"  Stop loss (50%)       : {r.stop_losses:>4}  ({r.stop_losses/total_closed*100:.1f}%)")
            print(f"  Time exit (day 8)     : {r.time_exits:>4}  ({r.time_exits/total_closed*100:.1f}%)")
            print(f"  DTE exit (<3d)        : {r.dte_exits:>4}  ({r.dte_exits/total_closed*100:.1f}%)")

        print(f"\n── Best & Worst Trades {'─'*37}")
        if r.trades:
            by_pnl = sorted(r.trades, key=lambda t: t.pnl_dollars, reverse=True)
            for t in by_pnl[:3]:
                move = f"{t.underlying_move_pct:+.1f}%" if t.underlying_move_pct != 0 else "n/a"
                print(f"  ✅  {t.entry_date}  {t.option_type.upper():4s}  K={t.strike:.0f}  "
                      f"DTE={t.dte_at_entry}  P&L=${t.pnl_dollars:+,.0f}  "
                      f"({t.pnl_pct:+.0f}%)  underlying {move}  [{t.exit_reason}]")
            print()
            for t in by_pnl[-3:]:
                move = f"{t.underlying_move_pct:+.1f}%" if t.underlying_move_pct != 0 else "n/a"
                print(f"  ❌  {t.entry_date}  {t.option_type.upper():4s}  K={t.strike:.0f}  "
                      f"DTE={t.dte_at_entry}  P&L=${t.pnl_dollars:+,.0f}  "
                      f"({t.pnl_pct:+.0f}%)  underlying {move}  [{t.exit_reason}]")

        print(f"\n{'═'*60}\n")

    def save_trades_csv(self, results: BacktestResults, output_path: str = "backtest_trades.csv") -> None:
        """Save all trades to CSV for further analysis."""
        import dataclasses
        rows = [dataclasses.asdict(t) for t in results.trades]
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info(f"Saved {len(rows)} trades to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity analysis — how does win rate change with IV rank threshold?
# ─────────────────────────────────────────────────────────────────────────────

def run_iv_rank_sensitivity(bt: OptionsBacktester, symbol: str,
                             start: str, end: str) -> None:
    """
    Sweep IV rank threshold from 30% to 80% and show how it affects results.
    Useful for calibrating the MAX_IVR_ENTRY constant.
    """
    global MAX_IVR_ENTRY
    print("\n── IV Rank Threshold Sensitivity ────────────────────────────────────")
    print(f"{'IVR Thresh':>12} │ {'Trades':>6} │ {'Win%':>6} │ {'Total P&L':>12} │ {'Sharpe':>7}")
    print("─" * 55)

    for threshold in [30, 40, 50, 60, 70, 80]:
        MAX_IVR_ENTRY = float(threshold)
        r = bt.run_backtest(symbol=symbol, start=start, end=end)
        print(f"  IVR ≤ {threshold:2d}%    │ {r.total_trades:>6} │ {r.win_rate:>5.1f}% │ "
              f"${r.total_pnl:>+11,.0f} │ {r.sharpe:>6.2f}")

    MAX_IVR_ENTRY = 60.0   # reset to default


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Options strategy backtester")
    parser.add_argument("--data",    required=True, help="Path to Kaggle CSV file or directory")
    parser.add_argument("--symbol",  default="SPY",         help="Ticker symbol")
    parser.add_argument("--start",   default="2018-01-01",  help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end",     default="2023-12-31",  help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--mode",    default="momentum",    help="Signal mode: momentum | synthetic")
    parser.add_argument("--sensitivity", action="store_true", help="Run IV rank sensitivity sweep")
    parser.add_argument("--save-csv",    action="store_true", help="Save trade log to CSV")
    args = parser.parse_args()

    global SIGNAL_MODE
    SIGNAL_MODE = args.mode

    bt = OptionsBacktester(args.data, args.symbol)
    bt.load()

    if args.sensitivity:
        run_iv_rank_sensitivity(bt, args.symbol, args.start, args.end)
    else:
        results = bt.run_backtest(args.symbol, args.start, args.end)
        bt.print_report(results)
        if args.save_csv:
            csv_path = f"backtest_{args.symbol}_{args.start[:4]}_{args.end[:4]}.csv"
            bt.save_trades_csv(results, csv_path)


if __name__ == "__main__":
    main()
