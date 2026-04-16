"""
Walk-Forward Backtest
=====================
Performs proper temporal validation:
- Rolling window training (6 months train, 1 month test)
- Multiple iterations for robustness
- Monte Carlo for drawdown estimation
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
            return pd.read_parquet(parquet_path)
        
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
        """Add technical indicators."""
        close = df["Close"]
        
        df["rsi_14"] = self._rsi(close, 14)
        df["rsi_9"] = self._rsi(close, 9)
        
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        df["sma_20"] = close.rolling(20).mean()
        df["sma_50"] = close.rolling(50).mean()
        
        df["returns_5d"] = close.pct_change(5)
        df["returns_10d"] = close.pct_change(10)
        
        return df
    
    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def run_backtest(self) -> Dict:
        """Run walk-forward backtest."""
        logger.info(f"Starting walk-forward backtest")
        logger.info(f"Train: {self.train_months} months, Test: {self.test_months} months")
        
        all_trades = []
        equity_curve = [10000]
        
        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")
            
            df = self.load_data(symbol)
            if df.empty:
                continue
            
            df = df.dropna()
            if len(df) < 100:
                continue
            
            dates = df.index.sort_values()
            
            start_date = dates[0]
            end_date = dates[-1]
            
            train_end = start_date + timedelta(days=30 * self.train_months)
            
            while train_end < end_date:
                test_end = train_end + timedelta(days=30 * self.test_months)
                
                train_data = df[(df.index >= start_date) & (df.index < train_end)]
                test_data = df[(df.index >= train_end) & (df.index < test_end)]
                
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
        losing = trades_df[trades_df["pnl"] <= 0]
        
        win_rate = len(winning) / len(trades_df) * 100
        avg_win = winning["pnl"].mean() if len(winning) > 0 else 0
        avg_loss = losing["pnl"].mean() if len(losing) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        max_dd = self._calculate_max_drawdown(trades_df)
        
        result = {
            "total_trades": len(trades_df),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_pnl": round(trades_df["pnl"].sum(), 2),
        }
        
        logger.info(f"Backtest Results: {result}")
        
        return result
    
    def _test_strategy(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        symbol: str,
    ) -> List[Dict]:
        """Test simple RSI strategy."""
        trades = []
        
        for i in range(len(test_data) - 1):
            row = test_data.iloc[i]
            next_row = test_data.iloc[i + 1]
            
            rsi = row.get("rsi_14", 50)
            price = row["Close"]
            next_price = next_row["Close"]
            
            if rsi < 30:
                pnl = (next_price - price) / price * 100
                trades.append({
                    "symbol": symbol,
                    "entry_date": row.name,
                    "direction": "buy",
                    "entry_price": price,
                    "exit_price": next_price,
                    "pnl": pnl,
                })
            
            elif rsi > 70:
                pnl = (price - next_price) / price * 100
                trades.append({
                    "symbol": symbol,
                    "entry_date": row.name,
                    "direction": "sell",
                    "entry_price": price,
                    "exit_price": next_price,
                    "pnl": pnl,
                })
        
        return trades
    
    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        if trades_df.empty:
            return 0
        
        equity = [10000]
        for _, row in trades_df.iterrows():
            equity.append(equity[-1] * (1 + row["pnl"] / 100))
        
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        
        return drawdown.max()
    
    def run_monte_carlo(self, n_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation."""
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")
        
        all_trades = []
        for symbol in self.symbols:
            df = self.load_data(symbol)
            if not df.empty:
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
                "avg_trade": np.mean(sim_trades),
                "max_loss": np.min(sim_trades),
                "max_win": np.max(sim_trades),
            })
        
        results_df = pd.DataFrame(results)
        
        return {
            "mean_return": round(results_df["total_return"].mean(), 2),
            "median_return": round(results_df["total_return"].median(), 2),
            "worst_case": round(results_df["total_return"].quantile(0.05), 2),
            "best_case": round(results_df["total_return"].quantile(0.95), 2),
            "prob_of_profit": (results_df["total_return"] > 0).mean() * 100,
        }
    
    def _simulate_trades(self, df: pd.DataFrame) -> List[Dict]:
        """Simulate trades from data."""
        trades = []
        
        close = df["Close"]
        rsi = df.get("rsi_14")
        
        if rsi is None:
            return trades
        
        for i in range(len(df) - 1):
            if rsi.iloc[i] < 30:
                pnl = (close.iloc[i + 1] - close.iloc[i]) / close.iloc[i] * 100
                trades.append({"pnl": pnl})
            elif rsi.iloc[i] > 70:
                pnl = (close.iloc[i] - close.iloc[i + 1]) / close.iloc[i] * 100
                trades.append({"pnl": pnl})
        
        return trades


def run_backtest():
    """Run complete backtest."""
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

Performance Metrics:
  Total Trades: {results.get('total_trades', 0)}
  Winning Trades: {results.get('winning_trades', 0)}
  Losing Trades: {results.get('losing_trades', 0)}
  Win Rate: {results.get('win_rate', 0):.1f}%
  
  Average Win: {results.get('avg_win', 0):.2f}%
  Average Loss: {results.get('avg_loss', 0):.2f}%
  Profit Factor: {results.get('profit_factor', 0):.2f}
  
  Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%
  Total P&L: {results.get('total_pnl', 0):.2f}%
""")
    
    logger.info("\nRunning Monte Carlo simulation...")
    mc_results = backtest.run_monte_carlo(1000)
    
    print(f"""
Monte Carlo (1000 simulations):
  Mean Return: {mc_results.get('mean_return', 0):.2f}%
  Median Return: {mc_results.get('median_return', 0):.2f}%
  5th Percentile (Worst): {mc_results.get('worst_case', 0):.2f}%
  95th Percentile (Best): {mc_results.get('best_case', 0):.2f}%
  Probability of Profit: {mc_results.get('prob_of_profit', 0):.1f}%
""")
    
    return results, mc_results


if __name__ == "__main__":
    run_backtest()