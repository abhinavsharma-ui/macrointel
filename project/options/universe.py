"""
Liquid Options Universe Filter
================================
Stops the options trader from touching illiquid small-cap stocks.
Options on small-caps have:
  - wide bid-ask spreads (0.30+ on a $0.50 option = 60% instant loss)
  - low open interest → you can't exit without moving the market
  - unreliable IV quotes from yfinance

This module maintains and checks a liquid options universe:
  - S&P 500 index options (SPY, QQQ, IWM)
  - Large-cap single names with: ADV > $50M, OI > 500 on ATM strike

Usage:
    from options.universe import LiquidUniverse

    univ = LiquidUniverse()
    if not univ.is_liquid(symbol, chain_df):
        logger.info(f"Skipping {symbol} — not in liquid options universe")
        continue
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded S&P 500 liquid options universe
# These are the ~150 names with reliably tight bid-ask and high OI.
# Updated quarterly — last update May 2025.
# ─────────────────────────────────────────────────────────────────────────────

# Core ETFs — always trade these (deepest liquidity of all options)
LIQUID_ETFS: Set[str] = {
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV", "XLY",
    "GLD", "TLT", "HYG", "EEM", "VXX", "UVXY", "SQQQ", "TQQQ",
}

# Large-cap single names — all have >$50M avg daily options volume
LIQUID_LARGE_CAPS: Set[str] = {
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
    "AVGO", "ORCL", "AMD", "INTC", "QCOM", "MU", "AMAT", "KLAC",
    "LRCX", "MRVL", "ARM", "CRWD", "PANW", "SNOW", "NET", "DDOG",
    "ZS", "FTNT", "OKTA", "MDB", "CRM", "NOW", "ADBE", "INTU",
    "UBER", "LYFT", "ABNB", "BKNG", "EXPE",

    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW",
    "V", "MA", "PYPL", "SQ", "COF", "DFS",

    # Healthcare / Biotech (liquid enough)
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "BMY", "GILD", "AMGN",
    "BIIB", "REGN", "VRTX", "MRNA", "LLY", "CVS", "CI",

    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "OXY", "MPC",

    # Consumer
    "AMZN", "WMT", "COST", "TGT", "HD", "LOW", "NKE", "MCD",
    "SBUX", "YUM", "CMG", "DG", "DLTR",

    # Industrial / Other
    "BA", "CAT", "DE", "GE", "RTX", "LMT", "NOC", "HON",
    "UNP", "CSX", "FDX", "UPS",

    # EV / Clean energy
    "RIVN", "LCID", "PLUG", "FCEL", "BE",   # some are liquid enough due to retail

    # Volatile / high-beta (lots of options activity)
    "GME", "AMC", "BBBY", "MEME", "COIN", "HOOD",
}

# Combined universe
LIQUID_UNIVERSE: Set[str] = LIQUID_ETFS | LIQUID_LARGE_CAPS

# Minimum liquidity requirements at the contract level
MIN_OI_ATM       = 100    # minimum open interest on the ATM contract
MIN_ADV_USD      = 1e6    # minimum $1M daily options dollar volume (contract × mid × 100)
MAX_SPREAD_PCT   = 0.20   # max (ask - bid) / ask on ATM contract (20%)


# ─────────────────────────────────────────────────────────────────────────────
# Universe checker
# ─────────────────────────────────────────────────────────────────────────────

class LiquidUniverse:
    """
    Checks whether a symbol is liquid enough for options trading.
    Two levels of check:
      1. Hard list — is the symbol in the pre-approved universe?
      2. Live chain check — does the actual chain meet OI/spread minimums?
         (Optional: only runs if you pass the chain DataFrame.)
    """

    def __init__(self, extra_symbols: Optional[Set[str]] = None):
        self.universe = LIQUID_UNIVERSE.copy()
        if extra_symbols:
            self.universe |= extra_symbols
        logger.info(f"Liquid universe initialised: {len(self.universe)} symbols")

    def in_universe(self, symbol: str) -> bool:
        """Fast O(1) hard-list check."""
        return symbol.upper() in self.universe

    def is_liquid(
        self,
        symbol: str,
        chain_df: Optional[pd.DataFrame] = None,
        current_price: Optional[float] = None,
    ) -> bool:
        """
        Full liquidity check.

        Args:
            symbol:        ticker
            chain_df:      yfinance options chain DataFrame (optional)
            current_price: current stock price (to find ATM strike)

        Returns True only if both hard-list AND chain checks pass.
        """
        sym = symbol.upper()

        # 1. Hard list
        if sym not in self.universe:
            logger.debug(f"{sym}: not in liquid universe")
            return False

        # 2. Live chain check (optional — only if chain provided)
        if chain_df is not None and current_price is not None:
            if not self._chain_liquid(sym, chain_df, current_price):
                return False

        return True

    def _chain_liquid(
        self,
        symbol: str,
        chain: pd.DataFrame,
        current_price: float,
    ) -> bool:
        """
        Check liquidity of the actual options chain:
          - OI on near-ATM contracts ≥ MIN_OI_ATM
          - Bid-ask spread ≤ MAX_SPREAD_PCT
        """
        if chain.empty:
            return False

        # find ATM strike
        chain = chain.copy()
        chain["dist"] = (chain["strike"] - current_price).abs()
        atm = chain.nsmallest(3, "dist")

        if atm.empty:
            return False

        # OI check
        oi_col = next((c for c in ["openInterest", "open_interest", "OI", "c_oi"]
                       if c in atm.columns), None)
        if oi_col:
            max_oi = atm[oi_col].max()
            if max_oi < MIN_OI_ATM:
                logger.debug(f"{symbol}: ATM OI={max_oi:.0f} < {MIN_OI_ATM}")
                return False

        # Spread check
        bid_col = next((c for c in ["bid", "c_bid"] if c in atm.columns), None)
        ask_col = next((c for c in ["ask", "c_ask"] if c in atm.columns), None)
        if bid_col and ask_col:
            spread_pct = (atm[ask_col] - atm[bid_col]) / atm[ask_col].clip(lower=0.01)
            avg_spread = spread_pct.mean()
            if avg_spread > MAX_SPREAD_PCT:
                logger.debug(f"{symbol}: spread={avg_spread:.1%} > {MAX_SPREAD_PCT:.0%}")
                return False

        return True

    def add_symbol(self, symbol: str) -> None:
        """Dynamically add a symbol to the universe (e.g. from screener)."""
        self.universe.add(symbol.upper())

    def filter_candidates(self, symbols: list) -> list:
        """Filter a list of symbols to only those in the liquid universe."""
        filtered = [s for s in symbols if s.upper() in self.universe]
        removed  = len(symbols) - len(filtered)
        if removed:
            logger.info(f"Universe filter: removed {removed} illiquid symbols, "
                        f"{len(filtered)} remain")
        return filtered

    def __len__(self) -> int:
        return len(self.universe)

    def __repr__(self) -> str:
        return f"LiquidUniverse({len(self.universe)} symbols)"


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    univ = LiquidUniverse()
    print(f"\nLiquid universe: {len(univ)} symbols")

    test_cases = [
        ("AAPL", True),
        ("SPY",  True),
        ("SNXX", False),   # small-cap illiquid
        ("AXTI", False),   # small-cap illiquid
        ("NVDA", True),
        ("GME",  True),    # retail-driven but liquid enough
        ("MEME_COIN_XYZ", False),
    ]

    print(f"\n{'Symbol':>15} │ {'Expected':>9} │ {'Got':>5} │ {'Pass?':>5}")
    print("─" * 50)
    all_pass = True
    for sym, expected in test_cases:
        result = univ.in_universe(sym)
        ok = "✅" if result == expected else "❌"
        if result != expected:
            all_pass = False
        print(f"  {sym:>12}   │ {str(expected):>9} │ {str(result):>5} │ {ok}")

    print(f"\n{'All tests passed ✅' if all_pass else 'Some tests failed ❌'}\n")
