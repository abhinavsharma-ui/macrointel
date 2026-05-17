"""
IV Rank & IV Percentile Calculator
====================================
The professional filter that tells you whether options are cheap or expensive
BEFORE you buy them.

Two metrics, both important:

  IV Rank (IVR):
    (current IV - 52w low) / (52w high - 52w low)
    0% = cheapest options have been all year
    100% = most expensive options have been all year
    BUY when IVR < 50% (ideally < 30%)

  IV Percentile (IVP):
    % of trading days in the past year where IV was LOWER than today
    IVP = 30% means IV was lower on 30% of days → currently elevated
    BUY when IVP < 50%

Why this matters for your system:
  Your Form 4 / ClinicalTrials signals fire BEFORE the crowd knows.
  That means IV should still be low when your signal fires.
  If IV is already elevated (IVR > 60%), it means someone else already
  detected the catalyst and you're paying their markup. Pass on that trade.

Usage:
    from options.iv_rank import IVRankCalculator, iv_rank_filter

    calc = IVRankCalculator()
    result = calc.get_iv_rank("AAPL")
    print(result)

    # Gate check — returns True if safe to buy options
    ok = iv_rank_filter("AAPL", max_ivr=50.0)
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

LOOKBACK_DAYS   = 252    # 1 trading year
CACHE_TTL_HOURS = 4      # re-fetch IV data every 4 hours
MAX_IVR_ENTRY   = 60.0   # don't buy options if IVR > 60%
MAX_IVP_ENTRY   = 70.0   # don't buy options if IVP > 70%
MIN_DATA_POINTS = 60     # need at least 60 days of IV history to trust ranking


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IVRankResult:
    symbol:        str
    current_iv:    float    # today's average IV across near-term chain
    iv_52w_high:   float
    iv_52w_low:    float
    iv_rank:       float    # 0-100, lower = cheaper options
    iv_percentile: float    # 0-100, lower = cheaper options
    data_points:   int      # days of history used
    timestamp:     str

    # Derived
    regime:        str      # 'cheap', 'normal', 'elevated', 'expensive'
    buy_signal:    bool     # True = safe to buy options
    note:          str      # human-readable explanation

    def __str__(self) -> str:
        flag = "✅ BUY OK" if self.buy_signal else "⛔ SKIP"
        return (
            f"{self.symbol:8s} | IV={self.current_iv*100:.1f}%  "
            f"IVR={self.iv_rank:.0f}%  IVP={self.iv_percentile:.0f}%  "
            f"[{self.regime.upper()}]  {flag}  {self.note}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# IV history fetcher
# ─────────────────────────────────────────────────────────────────────────────

class IVHistoryFetcher:
    """
    Fetches implied volatility history for a symbol.

    Primary source: yfinance options chains (free, real-time)
    Method: compute average IV across near-term strikes each day by
            sampling the current chain and using historical close prices
            to approximate historical IV via HV proxy when chain history
            is unavailable.

    Wall Street shops use ORATS, OptionMetrics, or Cboe LiveVol for
    proper historical IV chains — if you upgrade to one of those APIs,
    swap this class out. The interface stays the same.
    """

    def __init__(self):
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError("pip install yfinance")

    def fetch_current_iv(self, symbol: str) -> Optional[float]:
        """
        Get current average implied volatility from near-term options chain.
        Averages IV across ATM +/- 2 strikes on the nearest expiry.
        """
        try:
            ticker = self._yf.Ticker(symbol)
            exps   = ticker.options
            if not exps:
                return None

            # Use nearest expiry with > 5 DTE for cleaner IV
            info  = ticker.fast_info
            S     = getattr(info, 'last_price', None) or getattr(info, 'regularMarketPrice', None)
            if not S or S <= 0:
                return None

            chain  = ticker.option_chain(exps[0])
            calls  = chain.calls
            if calls.empty:
                return None

            # Filter to near-ATM strikes (within 5% of current price)
            near_atm = calls[
                (calls['strike'] >= S * 0.95) &
                (calls['strike'] <= S * 1.05) &
                (calls['impliedVolatility'] > 0.01) &
                (calls['impliedVolatility'] < 5.0)   # filter bad data
            ]

            if near_atm.empty:
                near_atm = calls[calls['impliedVolatility'] > 0.01]

            if near_atm.empty:
                return None

            # Volume-weighted average IV (ATM strikes have most volume)
            vols    = near_atm['volume'].fillna(1).clip(lower=1)
            iv_vals = near_atm['impliedVolatility']
            iv_avg  = float(np.average(iv_vals, weights=vols))
            return round(iv_avg, 4)

        except Exception as e:
            logger.debug(f"IV fetch failed for {symbol}: {e}")
            return None

    def fetch_hv_proxy(self, symbol: str, window: int = LOOKBACK_DAYS) -> Optional[pd.Series]:
        """
        Fetch historical volatility as a proxy for historical IV.
        HV and IV are correlated (IV mean-reverts to HV over time).
        Used when historical options chain data isn't available.

        Returns a pd.Series indexed by date with daily HV values.
        """
        try:
            ticker  = self._yf.Ticker(symbol)
            hist    = ticker.history(period="2y", interval="1d", auto_adjust=True)

            if hist.empty or len(hist) < 30:
                return None

            close   = hist['Close'].dropna()
            returns = np.log(close / close.shift(1)).dropna()

            # 30-day rolling HV annualised
            hv_30 = returns.rolling(30).std() * np.sqrt(252)
            hv_30 = hv_30.dropna().tail(window)

            # Add a mean-reversion premium (~20%) since IV typically > HV
            # This is a rough approximation — replace with real IV history
            # from ORATS or OptionMetrics for production
            iv_proxy = (hv_30 * 1.20).clip(0.05, 3.0)
            return iv_proxy

        except Exception as e:
            logger.debug(f"HV proxy failed for {symbol}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# IV Rank Calculator
# ─────────────────────────────────────────────────────────────────────────────

class IVRankCalculator:
    """
    Computes IV Rank and IV Percentile for any US stock.

    Caches results in memory to avoid hammering yfinance on every signal.
    Cache TTL is 4 hours (IV rank doesn't change minute-to-minute).
    """

    def __init__(
        self,
        max_ivr:      float = MAX_IVR_ENTRY,
        max_ivp:      float = MAX_IVP_ENTRY,
        cache_dir:    Optional[str] = None,
    ):
        self.max_ivr  = max_ivr
        self.max_ivp  = max_ivp
        self._fetcher = IVHistoryFetcher()
        self._cache:  Dict[str, IVRankResult] = {}
        self._cache_ts: Dict[str, float] = {}

        # Optional on-disk cache for persistence across runs
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _is_cache_fresh(self, symbol: str) -> bool:
        ts = self._cache_ts.get(symbol, 0)
        return (time.time() - ts) < (CACHE_TTL_HOURS * 3600)

    @staticmethod
    def _classify_regime(ivr: float, ivp: float) -> str:
        avg = (ivr + ivp) / 2
        if avg < 20:  return 'cheap'
        if avg < 40:  return 'normal'
        if avg < 60:  return 'elevated'
        return 'expensive'

    @staticmethod
    def _make_note(ivr: float, ivp: float, buy_signal: bool) -> str:
        if buy_signal:
            if ivr < 20:
                return "Options very cheap — excellent entry for buying premium"
            return "IV rank acceptable — reasonable premium to pay"
        if ivr > 80:
            return "IV near 52w high — you'd be buying at the top; wait for IV crush"
        if ivp > 70 and ivr <= 50:
            return f"IVP={ivp:.0f}% elevated — IV has been low most of the year, rising now"
        return f"IVR={ivr:.0f}% / IVP={ivp:.0f}% too high — market already pricing in the move"

    def get_iv_rank(self, symbol: str) -> Optional[IVRankResult]:
        """
        Compute IV Rank and Percentile for a symbol.
        Returns None if insufficient data.
        """
        # Check in-memory cache
        if symbol in self._cache and self._is_cache_fresh(symbol):
            return self._cache[symbol]

        # Get current IV
        current_iv = self._fetcher.fetch_current_iv(symbol)
        if current_iv is None or current_iv <= 0:
            logger.debug(f"No current IV for {symbol}")
            return None

        # Get historical IV (via HV proxy if real history unavailable)
        iv_history = self._fetcher.fetch_hv_proxy(symbol)
        if iv_history is None or len(iv_history) < MIN_DATA_POINTS:
            logger.debug(f"Insufficient IV history for {symbol} ({len(iv_history) if iv_history is not None else 0} points)")
            # Fall back: use current IV only, assume normal rank
            result = IVRankResult(
                symbol=symbol,
                current_iv=current_iv,
                iv_52w_high=current_iv * 1.5,   # estimated
                iv_52w_low=current_iv * 0.6,    # estimated
                iv_rank=50.0,                    # unknown → assume neutral
                iv_percentile=50.0,
                data_points=0,
                timestamp=datetime.now().isoformat(),
                regime='normal',
                buy_signal=True,                 # don't filter if no data
                note="Insufficient IV history — proceeding with caution",
            )
            self._cache[symbol]    = result
            self._cache_ts[symbol] = time.time()
            return result

        # Combine history with current IV
        all_iv = pd.concat([iv_history, pd.Series([current_iv])]).tail(LOOKBACK_DAYS)

        iv_high = float(all_iv.max())
        iv_low  = float(all_iv.min())
        n       = len(all_iv)

        # IV Rank
        denom  = iv_high - iv_low
        ivr    = ((current_iv - iv_low) / denom * 100) if denom > 0.001 else 50.0
        ivr    = float(np.clip(ivr, 0, 100))

        # IV Percentile — % of days where IV was LOWER than current
        ivp    = float((all_iv < current_iv).mean() * 100)

        regime     = self._classify_regime(ivr, ivp)
        buy_signal = (ivr <= self.max_ivr) and (ivp <= self.max_ivp)
        note       = self._make_note(ivr, ivp, buy_signal)

        result = IVRankResult(
            symbol=symbol,
            current_iv=current_iv,
            iv_52w_high=round(iv_high, 4),
            iv_52w_low=round(iv_low, 4),
            iv_rank=round(ivr, 1),
            iv_percentile=round(ivp, 1),
            data_points=n,
            timestamp=datetime.now().isoformat(),
            regime=regime,
            buy_signal=buy_signal,
            note=note,
        )

        self._cache[symbol]    = result
        self._cache_ts[symbol] = time.time()
        return result

    def batch_rank(self, symbols: list, verbose: bool = True) -> Dict[str, IVRankResult]:
        """
        Get IV rank for a list of symbols.
        Returns dict of {symbol: IVRankResult} for symbols with data.
        Skips symbols where data is unavailable without crashing.
        """
        results = {}
        for symbol in symbols:
            try:
                r = self.get_iv_rank(symbol)
                if r is not None:
                    results[symbol] = r
                    if verbose:
                        print(r)
            except Exception as e:
                logger.debug(f"IV rank failed for {symbol}: {e}")
        return results

    def filter_for_options(self, symbols: list) -> list:
        """
        Filter a list of symbols to only those where buying options makes sense.
        Returns list of symbols passing the IV rank gate.
        """
        passing = []
        ranks   = self.batch_rank(symbols, verbose=False)
        for symbol, r in ranks.items():
            if r.buy_signal:
                passing.append(symbol)
        logger.info(f"IV rank filter: {len(passing)}/{len(symbols)} symbols pass "
                    f"(IVR≤{self.max_ivr}%, IVP≤{self.max_ivp}%)")
        return passing


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function — drop-in gate check
# ─────────────────────────────────────────────────────────────────────────────

_default_calc = None

def iv_rank_filter(
    symbol: str,
    max_ivr: float = MAX_IVR_ENTRY,
    max_ivp: float = MAX_IVP_ENTRY,
) -> bool:
    """
    Simple gate: return True if it's safe to buy options on this symbol.
    Uses a module-level cached calculator.

    Usage:
        if iv_rank_filter("AAPL"):
            # proceed with options trade
    """
    global _default_calc
    if _default_calc is None:
        _default_calc = IVRankCalculator(max_ivr=max_ivr, max_ivp=max_ivp)

    result = _default_calc.get_iv_rank(symbol)
    if result is None:
        return True   # no data = don't block the trade
    return result.buy_signal


# ─────────────────────────────────────────────────────────────────────────────
# IV term structure — for understanding the shape of IV across expiries
# ─────────────────────────────────────────────────────────────────────────────

def get_iv_term_structure(symbol: str) -> Optional[pd.DataFrame]:
    """
    Get IV for each available expiry.
    Useful for detecting backwardation (near-term IV > far-term IV),
    which signals an imminent event the market is pricing.

    Returns DataFrame with columns: expiry, dte, atm_iv, volume_calls, volume_puts
    Backwardation (near IV spike) confirms your catalyst is live.
    """
    try:
        import yfinance as yf
        ticker  = yf.Ticker(symbol)
        exps    = ticker.options
        if not exps:
            return None

        info = ticker.fast_info
        S    = getattr(info, 'last_price', None) or getattr(info, 'regularMarketPrice', None)
        if not S:
            return None

        today  = pd.Timestamp.now().normalize()
        rows   = []

        for exp in exps[:8]:   # first 8 expiries
            try:
                chain  = ticker.option_chain(exp)
                calls  = chain.calls
                if calls.empty:
                    continue

                near_atm = calls[
                    (calls['strike'] >= S * 0.97) &
                    (calls['strike'] <= S * 1.03) &
                    (calls['impliedVolatility'] > 0.01)
                ]
                if near_atm.empty:
                    continue

                atm_iv = float(near_atm['impliedVolatility'].mean())
                exp_dt = pd.Timestamp(exp)
                dte    = max((exp_dt - today).days, 0)

                rows.append({
                    'expiry':      exp,
                    'dte':         dte,
                    'atm_iv':      round(atm_iv, 4),
                    'atm_iv_pct':  round(atm_iv * 100, 2),
                    'volume_calls': int(calls['volume'].sum()),
                })
            except Exception:
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows).sort_values('dte').reset_index(drop=True)

        # Detect backwardation
        if len(df) >= 2:
            near_iv = df['atm_iv'].iloc[0]
            far_iv  = df['atm_iv'].iloc[1]
            df['backwardation'] = near_iv > far_iv * 1.05

        return df

    except Exception as e:
        logger.debug(f"Term structure failed for {symbol}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== IV RANK CALCULATOR SELF-TEST ===\n")

    calc = IVRankCalculator(max_ivr=50.0, max_ivp=50.0)

    symbols = ["AAPL", "MSFT", "NVDA", "META"]
    print(f"Testing {len(symbols)} symbols:\n")

    for sym in symbols:
        r = calc.get_iv_rank(sym)
        if r:
            print(r)
        else:
            print(f"{sym}: no data")

    print("\n--- Term structure for AAPL ---")
    ts = get_iv_term_structure("AAPL")
    if ts is not None:
        print(ts[['expiry', 'dte', 'atm_iv_pct', 'volume_calls']].to_string(index=False))
    else:
        print("No term structure data")

    print("\n--- Gate check ---")
    for sym in symbols:
        ok = iv_rank_filter(sym)
        print(f"{sym}: {'✅ PASS' if ok else '⛔ SKIP'}")
