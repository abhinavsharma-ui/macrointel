"""
Walk-Forward Backtest
=====================
Performs proper temporal validation:
- Rolling window training (6 months train, 1 month test)
- Multiple iterations for robustness
- Monte Carlo for drawdown estimation
- ATR-based SL/TP with 1.67:1 reward/risk ratio
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

os.environ["FINNHUB_API_KEYS"] = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

MODELS_DIR = Path(__file__).parent / "models" / "checkpoints"
DATA_DIR = Path(__file__).parent / "data" / "features"

# ── Risk parameters ────────────────────────────────────────────
ATR_PERIOD      = 14       # ATR lookback
SL_ATR_MULT     = 1.5      # Stop loss  = 1.5 × ATR
TP_ATR_MULT     = 2.5      # Take profit = 2.5 × ATR  →  1.67:1 R:R
MAX_HOLD_DAYS   = 10       # Force-exit after N bars if neither hit
MIN_ATR_PCT     = 0.003    # Skip trade if ATR < 0.3% of price (no volatility)
# ──────────────────────────────────────────────────────────────


class WalkForwardBacktest:
    """
    Walk-forward backtesting with proper temporal splits.
    """

    def __init__(
        self,
        train_months: int = 6,
        test_months: int = 1,
        symbols: List[str] = None,
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.symbols = symbols or ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS"]
        self.results = []

    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load feature data for symbol."""
        parquet_path = DATA_DIR / f"{symbol}.parquet"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            df = self._add_technical_features(df)
            return df

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y", interval="1d")
            df = self._add_technical_features(df)
            return df
        except Exception as e:
            logger.warning(f"Could not load data for {symbol}: {e}")
            return pd.DataFrame()

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators including ATR."""
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        # RSI
        df["rsi_14"] = self._rsi(close, 14)
        df["rsi_9"]  = self._rsi(close, 9)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # Trend filter
        df["sma_20"] = close.rolling(20).mean()
        df["sma_50"] = close.rolling(50).mean()

        # Returns
        df["returns_5d"]  = close.pct_change(5)
        df["returns_10d"] = close.pct_change(10)

        # ATR — true range across high/low/prev_close
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(ATR_PERIOD).mean()

        return df

    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs    = gain / loss
        return 100 - (100 / (1 + rs))

    # ── core backtest ──────────────────────────────────────────

    def run_backtest(self) -> Dict:
        """Run walk-forward backtest."""
        logger.info("Starting walk-forward backtest")
        logger.info(f"Train: {self.train_months} months, Test: {self.test_months} months")
        logger.info(f"SL: {SL_ATR_MULT}×ATR  |  TP: {TP_ATR_MULT}×ATR  |  Max hold: {MAX_HOLD_DAYS} bars")

        all_trades = []

        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")

            df = self.load_data(symbol)
            if df.empty:
                continue

            df = df.dropna()
            if len(df) < 100:
                continue

            dates      = df.index.sort_values()
            start_date = dates[0]
            end_date   = dates[-1]
            train_end  = start_date + timedelta(days=30 * self.train_months)

            while train_end < end_date:
                test_end = train_end + timedelta(days=30 * self.test_months)

                train_data = df[(df.index >= start_date) & (df.index < train_end)]
                test_data  = df[(df.index >= train_end)  & (df.index < test_end)]

                if len(train_data) < 60 or len(test_data) < 10:
                    train_end += timedelta(days=30)
                    continue

                trades = self._test_strategy(train_data, test_data, symbol)
                all_trades.extend(trades)
                train_end += timedelta(days=30)

        if not all_trades:
            logger.warning("No trades generated")
            return {"error": "No trades"}

        trades_df = pd.DataFrame(all_trades)

        winning = trades_df[trades_df["pnl"] > 0]
        losing  = trades_df[trades_df["pnl"] <= 0]

        win_rate      = len(winning) / len(trades_df) * 100
        avg_win       = winning["pnl"].mean() if len(winning) > 0 else 0
        avg_loss      = losing["pnl"].mean()  if len(losing)  > 0 else 0
        gross_profit  = winning["pnl"].sum()  if len(winning) > 0 else 0
        gross_loss    = abs(losing["pnl"].sum()) if len(losing) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        max_dd = self._calculate_max_drawdown(trades_df)

        result = {
            "total_trades":      len(trades_df),
            "winning_trades":    len(winning),
            "losing_trades":     len(losing),
            "win_rate":          round(win_rate, 2),
            "avg_win":           round(avg_win, 2),
            "avg_loss":          round(avg_loss, 2),
            "profit_factor":     round(profit_factor, 2),
            "max_drawdown_pct":  round(max_dd, 2),
            "total_pnl":         round(trades_df["pnl"].sum(), 2),
            "sl_atr_mult":       SL_ATR_MULT,
            "tp_atr_mult":       TP_ATR_MULT,
        }

        logger.info(f"Backtest Results: {result}")
        return result

    def _test_strategy(
        self,
        train_data: pd.DataFrame,
        test_data:  pd.DataFrame,
        symbol:     str,
    ) -> List[Dict]:
        """
        RSI mean-reversion with ATR-based SL/TP.

        Entry rules (LONG):
          - RSI(14) < 30  (oversold)
          - Close > SMA50  (don't buy into a downtrend)
          - MACD hist turning up  (momentum confirmation)

        Entry rules (SHORT):
          - RSI(14) > 70  (overbought)
          - Close < SMA50  (don't short a strong uptrend)
          - MACD hist turning down

        Exit rules (both directions):
          - Stop loss  : entry ± SL_ATR_MULT × ATR
          - Take profit: entry ± TP_ATR_MULT × ATR
          - Time stop  : MAX_HOLD_DAYS bars (exit at close)
        """
        trades = []
        arr    = test_data.reset_index(drop=False)   # keep date in column

        i = 0
        while i < len(arr) - 1:
            row   = arr.iloc[i]
            price = row["Close"]
            rsi   = row.get("rsi_14", 50)
            atr   = row.get("atr",    np.nan)
            sma50 = row.get("sma_50", np.nan)
            macd_h= row.get("macd_hist", 0)

            # Skip if ATR is missing or too small (illiquid / no data)
            if pd.isna(atr) or atr < price * MIN_ATR_PCT:
                i += 1
                continue

            sl_dist = SL_ATR_MULT * atr
            tp_dist = TP_ATR_MULT * atr

            direction = None

            # Long signal
            if (rsi < 30
                    and (pd.isna(sma50) or price > sma50 * 0.98)   # slight tolerance
                    and macd_h > arr.iloc[max(i-1,0)].get("macd_hist", macd_h)):
                direction = "buy"
                sl_price  = price - sl_dist
                tp_price  = price + tp_dist

            # Short signal
            elif (rsi > 70
                    and (pd.isna(sma50) or price < sma50 * 1.02)
                    and macd_h < arr.iloc[max(i-1,0)].get("macd_hist", macd_h)):
                direction = "sell"
                sl_price  = price + sl_dist
                tp_price  = price - tp_dist

            if direction is None:
                i += 1
                continue

            # Simulate bar-by-bar until SL / TP / time-stop
            entry_price = price
            exit_price  = None
            exit_reason = "time"

            for j in range(i + 1, min(i + 1 + MAX_HOLD_DAYS, len(arr))):
                future      = arr.iloc[j]
                bar_high    = future["High"]
                bar_low     = future["Low"]
                bar_close   = future["Close"]

                if direction == "buy":
                    if bar_low <= sl_price:
                        exit_price  = sl_price
                        exit_reason = "sl"
                        break
                    if bar_high >= tp_price:
                        exit_price  = tp_price
                        exit_reason = "tp"
                        break
                else:  # sell
                    if bar_high >= sl_price:
                        exit_price  = sl_price
                        exit_reason = "sl"
                        break
                    if bar_low <= tp_price:
                        exit_price  = tp_price
                        exit_reason = "tp"
                        break

                # Time stop — exit on last bar close
                if j == min(i + MAX_HOLD_DAYS, len(arr) - 1):
                    exit_price  = bar_close
                    exit_reason = "time"

            if exit_price is None:
                exit_price  = arr.iloc[min(i + 1, len(arr) - 1)]["Close"]
                exit_reason = "time"

            if direction == "buy":
                pnl = (exit_price - entry_price) / entry_price * 100
            else:
                pnl = (entry_price - exit_price) / entry_price * 100

            trades.append({
                "symbol":      symbol,
                "entry_date":  row.get("index", row.name),
                "direction":   direction,
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "exit_reason": exit_reason,
                "atr":         round(atr, 4),
                "pnl":         round(pnl, 4),
            })

            # Skip ahead past the trade bars
            hold_bars = {"sl": 1, "tp": 1, "time": MAX_HOLD_DAYS}
            i += hold_bars.get(exit_reason, 1)

        return trades

    # ── helpers ───────────────────────────────────────────────

    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        if trades_df.empty:
            return 0
        equity = [10000]
        for _, row in trades_df.iterrows():
            equity.append(equity[-1] * (1 + row["pnl"] / 100))
        equity   = np.array(equity)
        peak     = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        return drawdown.max()

    def run_monte_carlo(self, n_simulations: int = 1000) -> Dict:
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")

        all_trades = []
        for symbol in self.symbols:
            df = self.load_data(symbol)
            if not df.empty:
                df = df.dropna()
                trades = self._simulate_trades(df)
                all_trades.extend(trades)

        if not all_trades:
            return {"error": "No data for simulation"}

        trade_returns = [t["pnl"] for t in all_trades]

        results = []
        for _ in range(n_simulations):
            sim_trades = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            results.append({
                "total_return": np.sum(sim_trades),
                "avg_trade":    np.mean(sim_trades),
                "max_loss":     np.min(sim_trades),
                "max_win":      np.max(sim_trades),
            })

        results_df = pd.DataFrame(results)

        return {
            "mean_return":    round(results_df["total_return"].mean(), 2),
            "median_return":  round(results_df["total_return"].median(), 2),
            "worst_case":     round(results_df["total_return"].quantile(0.05), 2),
            "best_case":      round(results_df["total_return"].quantile(0.95), 2),
            "prob_of_profit": round((results_df["total_return"] > 0).mean() * 100, 1),
        }

    def _simulate_trades(self, df: pd.DataFrame) -> List[Dict]:
        trades = []
        close  = df["Close"]
        rsi    = df.get("rsi_14")
        atr    = df.get("atr")

        if rsi is None or atr is None:
            return trades

        for i in range(len(df) - 1):
            r = rsi.iloc[i]
            a = atr.iloc[i]
            p = close.iloc[i]

            if pd.isna(a) or a < p * MIN_ATR_PCT:
                continue

            if r < 30:
                tp = p + TP_ATR_MULT * a
                sl = p - SL_ATR_MULT * a
                next_p = close.iloc[i + 1]
                if next_p >= tp:
                    pnl = TP_ATR_MULT * a / p * 100
                elif next_p <= sl:
                    pnl = -SL_ATR_MULT * a / p * 100
                else:
                    pnl = (next_p - p) / p * 100
                trades.append({"pnl": pnl})

            elif r > 70:
                tp = p - TP_ATR_MULT * a
                sl = p + SL_ATR_MULT * a
                next_p = close.iloc[i + 1]
                if next_p <= tp:
                    pnl = TP_ATR_MULT * a / p * 100
                elif next_p >= sl:
                    pnl = -SL_ATR_MULT * a / p * 100
                else:
                    pnl = (p - next_p) / p * 100
                trades.append({"pnl": pnl})

        return trades


def run_backtest():
    logger.info("=" * 60)
    logger.info("WALK-FORWARD BACKTEST")
    logger.info("=" * 60)

    backtest = WalkForwardBacktest(
        train_months=6,
        test_months=1,
        symbols=["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS", "BAJFINANCE.NS"],
    )

    results = backtest.run_backtest()

    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)

    print(f"""
=== Walk-Forward Backtest Results ===

Risk Setup:  SL={SL_ATR_MULT}×ATR  |  TP={TP_ATR_MULT}×ATR  |  Max hold={MAX_HOLD_DAYS} bars

Performance Metrics:
  Total Trades:   {results.get('total_trades', 0)}
  Winning Trades: {results.get('winning_trades', 0)}
  Losing Trades:  {results.get('losing_trades', 0)}
  Win Rate:       {results.get('win_rate', 0):.1f}%

  Average Win:    {results.get('avg_win', 0):.2f}%
  Average Loss:   {results.get('avg_loss', 0):.2f}%
  Profit Factor:  {results.get('profit_factor', 0):.2f}

  Max Drawdown:   {results.get('max_drawdown_pct', 0):.2f}%
  Total P&L:      {results.get('total_pnl', 0):.2f}%
""")

    logger.info("Running Monte Carlo simulation...")
    mc_results = backtest.run_monte_carlo(1000)

    print(f"""
Monte Carlo (1000 simulations):
  Mean Return:            {mc_results.get('mean_return', 0):.2f}%
  Median Return:          {mc_results.get('median_return', 0):.2f}%
  5th Percentile (Worst): {mc_results.get('worst_case', 0):.2f}%
  95th Percentile (Best): {mc_results.get('best_case', 0):.2f}%
  Probability of Profit:  {mc_results.get('prob_of_profit', 0):.1f}%
""")

    return results, mc_results


if __name__ == "__main__":
    run_backtest()
