"""
Strike & Expiry Selector
=========================
Given a directional signal + current stock price + expected move,
picks the optimal options contract to trade.

This is the execution-edge layer. Being directionally right is not enough —
buying the wrong strike or wrong expiry destroys your edge even when the
underlying moves exactly as predicted.

Selection philosophy (calibrated to your 7% / 8-day hold system):
  - Strike: 60% of the expected move OTM (captures most upside with leverage)
  - Expiry: hold_days + 4-day buffer, rounded UP to nearest weekly expiry
  - Delta target: 0.25–0.40 (sweet spot of leverage vs cost)
  - Never buy < 7 DTE (theta is brutal)
  - Prefer high open interest (liquidity = tighter bid/ask)

Usage:
    from options.strike_selector import StrikeSelector, ContractSpec

    selector = StrikeSelector()
    contract = selector.select(
        symbol="AAPL",
        current_price=150.0,
        expected_move_pct=7.0,
        hold_days=8,
        direction='call',
    )
    print(contract)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from options.black_scholes import price_option, OptionResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — tuned for the 7%/8-day strategy
# ─────────────────────────────────────────────────────────────────────────────

# Strike selection: how far OTM relative to expected move
# 0.60 = buy strike at 60% of target distance (e.g. 4.2% OTM for 7% target)
OTM_RATIO          = 0.60

# Delta bounds — reject contracts outside this range
MIN_DELTA          = 0.20   # too far OTM → needs huge move
MAX_DELTA          = 0.50   # too close ATM → not enough leverage

# DTE bounds
MIN_DTE            = 7      # never buy < 7 DTE (theta crush)
BUFFER_DAYS        = 4      # extra days beyond hold period
MAX_DTE            = 45     # don't buy far-dated (theta too slow to show gains)

# Minimum open interest for liquidity
MIN_OPEN_INTEREST  = 100

# Minimum volume for liquidity
MIN_VOLUME         = 10

# Strike rounding — most liquid US options have strikes every $1 below $25,
# $2.50 between $25-$200, $5 above $200 (approximate)
def _round_to_nearest_strike(price: float, target: float) -> float:
    """Round target to nearest standard strike increment."""
    if price < 25:
        increment = 1.0
    elif price < 200:
        increment = 2.5
    else:
        increment = 5.0
    return round(round(target / increment) * increment, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContractSpec:
    """The selected options contract with full rationale."""
    symbol:           str
    direction:        str        # 'call' or 'put'
    strike:           float
    expiry:           str        # YYYY-MM-DD
    dte:              int        # days to expiry from today

    # Pricing (from Black-Scholes using current IV)
    theoretical_price: float    # per share
    contract_cost:    float     # × 100 shares = one contract cost
    delta:            float
    theta_per_day:    float     # daily decay in dollars per share
    theta_pct_daily:  float     # daily decay as % of premium

    # Trade rationale
    otm_pct:          float     # how far OTM
    expected_move_pct: float    # what your signal predicts
    breakeven_move_pct: float   # stock must move this % to break even at expiry

    # Liquidity (from live chain, if available)
    open_interest:    int = 0
    bid:              float = 0.0
    ask:              float = 0.0
    bid_ask_spread:   float = 0.0   # as % of mid — lower is better

    # Quality score (0-100)
    score:            float = 0.0
    score_breakdown:  dict = None
    warnings:         list = None

    def __post_init__(self):
        if self.score_breakdown is None:
            self.score_breakdown = {}
        if self.warnings is None:
            self.warnings = []

    def __str__(self) -> str:
        spread_str = f"  Spread: {self.bid_ask_spread:.1f}%" if self.bid_ask_spread > 0 else ""
        oi_str     = f"  OI: {self.open_interest:,}" if self.open_interest > 0 else ""
        warn_str   = f"\n  ⚠ {' | '.join(self.warnings)}" if self.warnings else ""
        return (
            f"\n{'═'*60}\n"
            f"  SELECTED CONTRACT: {self.symbol} {self.direction.upper()} "
            f"${self.strike} exp {self.expiry} ({self.dte} DTE)\n"
            f"{'─'*60}\n"
            f"  Cost    : ${self.theoretical_price:.2f}/share  "
            f"(${self.contract_cost:.0f}/contract)\n"
            f"  Delta   : {self.delta:+.3f}   OTM: {self.otm_pct:+.1f}%\n"
            f"  Theta   : ${self.theta_per_day:+.4f}/day  "
            f"({self.theta_pct_daily:.1f}%/day)\n"
            f"  Signal  : +{self.expected_move_pct:.1f}% expected move\n"
            f"  B/E     : stock must move {self.breakeven_move_pct:+.1f}% to profit at expiry"
            f"{spread_str}{oi_str}\n"
            f"  Score   : {self.score:.0f}/100\n"
            f"{'═'*60}{warn_str}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Expiry finder
# ─────────────────────────────────────────────────────────────────────────────

class ExpiryFinder:
    """
    Finds the best expiry from available options chain.
    Prefers weekly expiries (every Friday for most liquid names).
    """

    def __init__(self):
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError("pip install yfinance")

    def get_available_expiries(self, symbol: str) -> List[Tuple[str, int]]:
        """
        Returns list of (expiry_date_str, dte) tuples sorted by DTE.
        """
        try:
            ticker = self._yf.Ticker(symbol)
            exps   = ticker.options
            if not exps:
                return []

            today  = datetime.now().date()
            result = []
            for exp in exps:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                dte      = (exp_date - today).days
                if dte >= MIN_DTE:
                    result.append((exp, dte))
            return result
        except Exception as e:
            logger.debug(f"Expiry fetch failed for {symbol}: {e}")
            return []

    def select_expiry(
        self,
        symbol:     str,
        hold_days:  int,
        target_dte: Optional[int] = None,
    ) -> Optional[Tuple[str, int]]:
        """
        Select the optimal expiry.
        Target DTE = hold_days + BUFFER_DAYS, rounded to nearest available.
        """
        target = target_dte or (hold_days + BUFFER_DAYS)
        target = max(target, MIN_DTE)

        expiries = self.get_available_expiries(symbol)
        if not expiries:
            # Fallback: construct a synthetic expiry date
            exp_date = datetime.now().date() + timedelta(days=target)
            # Roll to nearest Friday
            while exp_date.weekday() != 4:
                exp_date += timedelta(days=1)
            return (exp_date.strftime("%Y-%m-%d"), target)

        # Pick the nearest expiry to our target that's >= MIN_DTE
        valid = [(exp, dte) for exp, dte in expiries if dte >= MIN_DTE]
        if not valid:
            return None

        # Prefer expiries close to target, slightly biased toward more time
        scored = sorted(valid, key=lambda x: abs(x[1] - target) + max(0, target - x[1]) * 0.3)
        return scored[0]


# ─────────────────────────────────────────────────────────────────────────────
# Strike selector
# ─────────────────────────────────────────────────────────────────────────────

class StrikeSelector:
    """
    Selects the optimal options contract given a directional signal.

    Two modes:
    1. Live mode: fetches real options chain from yfinance, scores each
       available contract, picks the best one.
    2. Synthetic mode: constructs a theoretical contract from Black-Scholes
       when live chain is unavailable (used in backtesting).
    """

    def __init__(self, use_live_chain: bool = True):
        self.use_live_chain = use_live_chain
        self._expiry_finder = ExpiryFinder()

        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError("pip install yfinance")

    # ── Live chain selection ──────────────────────────────────────────────

    def _get_live_chain(
        self,
        symbol:    str,
        expiry:    str,
        direction: str,
        S:         float,
    ) -> Optional[pd.DataFrame]:
        """Fetch real options chain for the selected expiry."""
        try:
            ticker = self._yf.Ticker(symbol)
            chain  = ticker.option_chain(expiry)
            df     = chain.calls if direction == 'call' else chain.puts
            if df.empty:
                return None

            # Clean and filter
            df = df.copy()
            df = df[df['strike'] > 0]
            df = df[df['impliedVolatility'] > 0.01]
            df = df[df['impliedVolatility'] < 5.0]    # filter bad data
            df['bid_ask_spread_pct'] = (
                (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2 + 1e-6) * 100
            ).clip(0, 100)
            return df.reset_index(drop=True)

        except Exception as e:
            logger.debug(f"Live chain failed for {symbol} {expiry}: {e}")
            return None

    def _score_contract(
        self,
        row:              pd.Series,
        S:                float,
        expected_move_pct: float,
        dte:              int,
        direction:        str,
    ) -> float:
        """
        Score a candidate contract 0-100.
        Higher = better for buying premium.
        """
        score = 0.0
        breakdown = {}

        strike = float(row['strike'])
        iv     = float(row['impliedVolatility'])
        oi     = int(row.get('openInterest', 0) or 0)
        vol    = int(row.get('volume', 0) or 0)
        spread = float(row.get('bid_ask_spread_pct', 50))

        # Target strike (60% of expected move OTM)
        if direction == 'call':
            target_strike = S * (1 + expected_move_pct / 100 * OTM_RATIO)
            otm_pct       = (strike - S) / S * 100
        else:
            target_strike = S * (1 - expected_move_pct / 100 * OTM_RATIO)
            otm_pct       = (S - strike) / S * 100

        # 1. Strike proximity to ideal (max 40 points)
        strike_diff_pct = abs(strike - target_strike) / S * 100
        strike_score    = max(0, 40 - strike_diff_pct * 8)
        score          += strike_score
        breakdown['strike_proximity'] = round(strike_score, 1)

        # 2. Delta range (max 25 points) — want 0.25-0.40
        # Approximate delta from IV and moneyness
        pricing = price_option(S=S, K=strike, T=dte, sigma=iv, option_type=direction)
        delta   = abs(pricing.greeks.delta)

        if MIN_DELTA <= delta <= MAX_DELTA:
            delta_score = 25.0
        elif delta < MIN_DELTA:
            delta_score = max(0, 25 - (MIN_DELTA - delta) * 300)
        else:
            delta_score = max(0, 25 - (delta - MAX_DELTA) * 300)
        score          += delta_score
        breakdown['delta_score'] = round(delta_score, 1)

        # 3. Liquidity (max 25 points)
        oi_score     = min(15, oi / 1000 * 15)     # full points at OI=1000
        vol_score    = min(5,  vol / 200 * 5)       # full points at vol=200
        spread_score = max(0, 5 - spread / 10 * 5)  # penalise wide spreads
        liq_score    = oi_score + vol_score + spread_score
        score       += liq_score
        breakdown['liquidity'] = round(liq_score, 1)

        # 4. DTE penalty (max 10 points) — prefer 10-20 DTE
        if 10 <= dte <= 20:
            dte_score = 10
        elif dte < 10:
            dte_score = max(0, 10 - (10 - dte) * 2)
        else:
            dte_score = max(0, 10 - (dte - 20) * 0.5)
        score += dte_score
        breakdown['dte_score'] = round(dte_score, 1)

        return float(np.clip(score, 0, 100)), breakdown, pricing

    # ── Main selection function ───────────────────────────────────────────

    def select(
        self,
        symbol:            str,
        current_price:     float,
        expected_move_pct: float,   # e.g. 7.0 for 7% expected move
        hold_days:         int = 8,
        direction:         str = 'call',
        current_iv:        Optional[float] = None,  # if None, fetched from chain
    ) -> Optional[ContractSpec]:
        """
        Select the optimal options contract.

        Args:
            symbol            : Stock ticker
            current_price     : Current stock price (S)
            expected_move_pct : Your model's predicted move in %
            hold_days         : How long you'll hold (default 8 days)
            direction         : 'call' (bullish) or 'put' (bearish)
            current_iv        : Override IV (e.g. from iv_rank module)

        Returns:
            ContractSpec with the selected contract, or None if no valid contract.
        """
        S = current_price

        # 1. Find expiry
        expiry_info = self._expiry_finder.select_expiry(symbol, hold_days)
        if expiry_info is None:
            logger.warning(f"No valid expiry found for {symbol}")
            return None
        expiry, dte = expiry_info

        # 2. Compute ideal strike
        if direction == 'call':
            ideal_strike = S * (1 + expected_move_pct / 100 * OTM_RATIO)
        else:
            ideal_strike = S * (1 - expected_move_pct / 100 * OTM_RATIO)
        ideal_strike = _round_to_nearest_strike(S, ideal_strike)

        # 3. Try live chain first
        if self.use_live_chain:
            contract = self._select_from_live_chain(
                symbol, expiry, dte, direction, S, ideal_strike,
                expected_move_pct, current_iv
            )
            if contract is not None:
                return contract

        # 4. Fallback: synthetic contract from Black-Scholes
        logger.info(f"Using synthetic contract for {symbol} (no live chain)")
        return self._make_synthetic_contract(
            symbol, S, ideal_strike, dte, expiry,
            direction, expected_move_pct, current_iv or 0.35
        )

    def _select_from_live_chain(
        self,
        symbol:            str,
        expiry:            str,
        dte:               int,
        direction:         str,
        S:                 float,
        ideal_strike:      float,
        expected_move_pct: float,
        current_iv:        Optional[float],
    ) -> Optional[ContractSpec]:
        """Score every available strike and pick the best."""
        chain = self._get_live_chain(symbol, expiry, direction, S)
        if chain is None or chain.empty:
            return None

        # Filter to reasonable strike range (±15% of current price)
        chain = chain[
            (chain['strike'] >= S * 0.85) &
            (chain['strike'] <= S * 1.15)
        ]
        if chain.empty:
            return None

        best_score    = -1.0
        best_row      = None
        best_pricing  = None
        best_breakdown = {}

        for _, row in chain.iterrows():
            try:
                score, breakdown, pricing = self._score_contract(
                    row, S, expected_move_pct, dte, direction
                )
                if score > best_score:
                    best_score     = score
                    best_row       = row
                    best_pricing   = pricing
                    best_breakdown = breakdown
            except Exception as e:
                logger.debug(f"Score failed for strike {row.get('strike')}: {e}")
                continue

        if best_row is None or best_pricing is None:
            return None

        strike   = float(best_row['strike'])
        iv       = float(best_row['impliedVolatility'])
        bid      = float(best_row.get('bid', 0) or 0)
        ask      = float(best_row.get('ask', 0) or 0)
        oi       = int(best_row.get('openInterest', 0) or 0)
        spread   = float(best_row.get('bid_ask_spread_pct', 0) or 0)

        # Breakeven move needed at expiry
        prem     = best_pricing.theoretical_price
        if direction == 'call':
            be_price   = strike + prem
            be_move    = (be_price - S) / S * 100
        else:
            be_price   = strike - prem
            be_move    = (S - be_price) / S * 100

        warnings = list(best_pricing.warnings)
        if oi < MIN_OPEN_INTEREST:
            warnings.append(f"Low OI ({oi}) — expect wide spreads at exit")
        if spread > 15:
            warnings.append(f"Wide bid/ask spread ({spread:.1f}%) — use limit orders")

        return ContractSpec(
            symbol=symbol,
            direction=direction,
            strike=strike,
            expiry=expiry,
            dte=dte,
            theoretical_price=prem,
            contract_cost=round(prem * 100, 2),
            delta=best_pricing.greeks.delta,
            theta_per_day=best_pricing.greeks.theta,
            theta_pct_daily=best_pricing.theta_pct_daily,
            otm_pct=best_pricing.otm_pct,
            expected_move_pct=expected_move_pct,
            breakeven_move_pct=round(be_move, 2),
            open_interest=oi,
            bid=bid,
            ask=ask,
            bid_ask_spread=spread,
            score=best_score,
            score_breakdown=best_breakdown,
            warnings=warnings,
        )

    def _make_synthetic_contract(
        self,
        symbol:            str,
        S:                 float,
        strike:            float,
        dte:               int,
        expiry:            str,
        direction:         str,
        expected_move_pct: float,
        sigma:             float,
    ) -> ContractSpec:
        """Build a synthetic contract from Black-Scholes when live chain unavailable."""
        pricing = price_option(S=S, K=strike, T=dte, sigma=sigma,
                               option_type=direction, symbol=symbol)
        prem    = pricing.theoretical_price

        if direction == 'call':
            be_move = ((strike + prem) - S) / S * 100
        else:
            be_move = (S - (strike - prem)) / S * 100

        return ContractSpec(
            symbol=symbol,
            direction=direction,
            strike=strike,
            expiry=expiry,
            dte=dte,
            theoretical_price=prem,
            contract_cost=round(prem * 100, 2),
            delta=pricing.greeks.delta,
            theta_per_day=pricing.greeks.theta,
            theta_pct_daily=pricing.theta_pct_daily,
            otm_pct=pricing.otm_pct,
            expected_move_pct=expected_move_pct,
            breakeven_move_pct=round(be_move, 2),
            open_interest=0,
            score=50.0,   # neutral score for synthetic
            warnings=pricing.warnings + ["Synthetic pricing — verify against live chain"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Delta-adjusted Kelly position sizing
# ─────────────────────────────────────────────────────────────────────────────

def delta_adjusted_kelly(
    win_prob:          float,   # ML model confidence (0-1)
    expected_move_pct: float,   # predicted stock move
    contract:          ContractSpec,
    portfolio_value:   float,   # total portfolio in dollars
    max_risk_pct:      float = 0.02,   # max 2% of portfolio per trade
    kelly_fraction:    float = 0.25,   # quarter-Kelly for options (more conservative)
) -> dict:
    """
    Calculate options position size using delta-adjusted Kelly criterion.

    Standard Kelly: f = (p*b - q) / b
    But for options, the R:R (b) is much larger, which makes Kelly suggest
    a large position. We use delta-adjusted notional to account for the
    fact that options have embedded leverage.

    Delta-adjusted Kelly:
      1. Calculate theoretical R:R for the option (3x to 8x for your setup)
      2. Apply Kelly → get % of portfolio
      3. Divide by delta to get true notional exposure
      4. Cap at max_risk_pct (never risk more than 2% of portfolio on one option)

    Returns dict with sizing in contracts, dollars, and risk metrics.
    """
    # Theoretical R:R: if stock moves expected_move_pct, option multiplies by ~3-8x
    # We use a conservative 2.5x as baseline (actual is higher but accounts for theta)
    estimated_option_multiplier = max(
        1.5,
        expected_move_pct / 100 * abs(contract.delta) * 100 / contract.theoretical_price
        * (1 - contract.theta_pct_daily / 100 * 8)  # theta cost over hold period
    )

    win_p  = min(max(win_prob, 0.01), 0.99)
    lose_p = 1 - win_p
    b      = estimated_option_multiplier   # reward per unit risked

    # Raw Kelly
    kelly_raw = (win_p * b - lose_p) / b
    kelly_raw = max(0, kelly_raw)

    # Apply fraction (quarter-Kelly is standard for high-leverage instruments)
    kelly_f = kelly_raw * kelly_fraction

    # Dollar amount to risk (premium paid = max loss on options)
    max_premium_risk = portfolio_value * max_risk_pct
    kelly_premium    = portfolio_value * kelly_f
    premium_to_risk  = min(max_premium_risk, kelly_premium)

    # Number of contracts
    cost_per_contract = contract.contract_cost
    if cost_per_contract <= 0:
        return {'contracts': 0, 'premium_risk': 0, 'delta_notional': 0}

    n_contracts = max(1, int(premium_to_risk / cost_per_contract))

    # Cap again at max risk
    actual_risk = n_contracts * cost_per_contract
    if actual_risk > max_premium_risk:
        n_contracts = max(1, int(max_premium_risk / cost_per_contract))
        actual_risk = n_contracts * cost_per_contract

    # Delta-adjusted notional (stock equivalent exposure)
    shares_equivalent = n_contracts * 100 * abs(contract.delta)
    delta_notional    = shares_equivalent * (contract.strike + contract.theoretical_price)

    return {
        'contracts':          n_contracts,
        'premium_risk':       round(actual_risk, 2),
        'premium_risk_pct':   round(actual_risk / portfolio_value * 100, 2),
        'delta_notional':     round(delta_notional, 2),
        'delta_notional_pct': round(delta_notional / portfolio_value * 100, 2),
        'kelly_raw':          round(kelly_raw, 4),
        'kelly_fraction_used': kelly_fraction,
        'estimated_multiplier': round(estimated_option_multiplier, 2),
        'max_profit_scenario': round(actual_risk * estimated_option_multiplier - actual_risk, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== STRIKE SELECTOR SELF-TEST ===\n")

    selector = StrikeSelector(use_live_chain=True)

    # Simulate your system firing a signal:
    # ML confidence 0.72, expected 7% move in 8 days
    symbol   = "AAPL"
    S        = 210.0   # approximate current price
    move     = 7.0
    hold     = 8

    print(f"Signal: {symbol} BUY  |  Expected move: +{move}%  |  Hold: {hold} days")
    print(f"Ideal OTM strike: ${S * (1 + move/100 * OTM_RATIO):.2f}  "
          f"({move * OTM_RATIO:.1f}% OTM)\n")

    contract = selector.select(
        symbol=symbol,
        current_price=S,
        expected_move_pct=move,
        hold_days=hold,
        direction='call',
    )

    if contract:
        print(contract)

        # Sizing
        sizing = delta_adjusted_kelly(
            win_prob=0.72,
            expected_move_pct=move,
            contract=contract,
            portfolio_value=50_000,
        )
        print(f"\nPosition Sizing (portfolio: $50,000):")
        print(f"  Contracts      : {sizing['contracts']}")
        print(f"  Premium at risk: ${sizing['premium_risk']:,.2f}  "
              f"({sizing['premium_risk_pct']:.2f}% of portfolio)")
        print(f"  Delta notional : ${sizing['delta_notional']:,.2f}  "
              f"({sizing['delta_notional_pct']:.1f}% stock-equivalent exposure)")
        print(f"  Est. multiplier: {sizing['estimated_multiplier']}x")
        print(f"  Max profit est : ${sizing['max_profit_scenario']:,.2f}")
    else:
        print("No contract selected (live data unavailable?)")

    # Synthetic mode test (for backtesting)
    print("\n--- Synthetic mode (no live chain) ---")
    selector_bt = StrikeSelector(use_live_chain=False)
    synth = selector_bt.select(
        symbol="SYNTH",
        current_price=100.0,
        expected_move_pct=7.0,
        hold_days=8,
        direction='call',
        current_iv=0.35,
    )
    if synth:
        print(synth)
