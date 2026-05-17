"""
Options Pricing Engine — Black-Scholes + Greeks
================================================
Foundation layer for the options trading module.

Provides:
  - Black-Scholes call/put pricing
  - Full Greeks: delta, gamma, theta, vega, rho
  - Implied volatility solver (Newton-Raphson)
  - Option diagnostics: intrinsic, extrinsic, breakeven, leverage ratio

All Greeks follow market convention:
  - Theta is expressed per CALENDAR DAY (not annualised)
  - Vega is expressed per 1 percentage-point change in IV (not per unit)
  - Delta is 0→1 for calls, -1→0 for puts

Usage:
    from options.black_scholes import price_option, implied_volatility, OptionResult

    result = price_option(
        S=150.0,    # stock price
        K=157.0,    # strike (4.7% OTM)
        T=12,       # days to expiry
        sigma=0.35, # implied vol (35%)
        option_type='call',
        r=0.053,    # risk-free rate (current ~5.3%)
    )
    print(result)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Risk-free rate default — US 3-month T-bill, update periodically
DEFAULT_RISK_FREE_RATE = 0.053  # 5.3% as of 2025


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Greeks:
    """All first and second-order option sensitivities."""
    delta: float   # ∂V/∂S — how much option moves per $1 stock move
    gamma: float   # ∂²V/∂S² — how fast delta changes
    theta: float   # ∂V/∂t per calendar day — daily time decay (negative)
    vega:  float   # ∂V/∂σ per 1% IV change — sensitivity to volatility
    rho:   float   # ∂V/∂r per 1% rate change — sensitivity to rates

    def __str__(self) -> str:
        return (
            f"Δ={self.delta:+.3f}  Γ={self.gamma:.5f}  "
            f"Θ={self.theta:+.4f}/day  "
            f"ν={self.vega:.4f}/1%IV  ρ={self.rho:.4f}"
        )


@dataclass
class OptionResult:
    """
    Full pricing result for one option contract (covers 100 shares).
    All dollar figures are per-share unless noted.
    """
    # Inputs (stored for reference)
    symbol:      str
    option_type: str        # 'call' or 'put'
    S:           float      # current stock price
    K:           float      # strike price
    T_days:      int        # calendar days to expiry
    sigma:       float      # implied vol used
    r:           float      # risk-free rate used

    # Pricing
    theoretical_price: float   # Black-Scholes fair value per share
    intrinsic:         float   # max(S-K, 0) for call
    extrinsic:         float   # time value = price - intrinsic
    contract_cost:     float   # theoretical_price × 100 (one contract)

    # Greeks
    greeks: Greeks

    # Trade diagnostics
    otm_pct:         float   # how far OTM as % of stock price
    breakeven:       float   # stock price needed at expiry to break even
    leverage_ratio:  float   # % gain on option per % gain on stock
    theta_pct_daily: float   # daily theta as % of option premium (decay rate)

    # Risk flags
    warnings: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"{'─'*55}",
            f"  {self.symbol} {self.option_type.upper()} ${self.K} "
            f"exp {self.T_days}d  |  Stock: ${self.S:.2f}",
            f"{'─'*55}",
            f"  Fair Value : ${self.theoretical_price:.2f}/share  "
            f"(${self.contract_cost:.0f}/contract)",
            f"  Intrinsic  : ${self.intrinsic:.2f}   "
            f"Extrinsic: ${self.extrinsic:.2f}",
            f"  OTM        : {self.otm_pct:+.1f}%   "
            f"Breakeven: ${self.breakeven:.2f}",
            f"  Leverage   : {self.leverage_ratio:.1f}x   "
            f"Theta: {self.theta_pct_daily:.1f}%/day",
            f"  {self.greeks}",
        ]
        if self.warnings:
            lines.append(f"  ⚠ {' | '.join(self.warnings)}")
        lines.append(f"{'─'*55}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core Black-Scholes math
# ─────────────────────────────────────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """Compute d1 and d2 terms. T is in years."""
    log_term = math.log(S / K)
    d1 = (log_term + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def _bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price. T in years."""
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price via put-call parity."""
    call = _bs_call_price(S, K, T, r, sigma)
    return call - S + K * math.exp(-r * T)


def _compute_greeks(
    S: float,
    K: float,
    T: float,      # in years
    r: float,
    sigma: float,
    option_type: str,
) -> Greeks:
    """Calculate all five Greeks. T in years."""
    if T <= 1e-6:
        T = 1 / 365  # floor to avoid division by zero on expiry day

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    sqrt_T  = math.sqrt(T)

    # ── Delta ──────────────────────────────────────────────
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1.0

    # ── Gamma (same for calls and puts) ───────────────────
    gamma = pdf_d1 / (S * sigma * sqrt_T)

    # ── Theta (per calendar day) ───────────────────────────
    base_theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T)
    if option_type == 'call':
        theta = (base_theta - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (base_theta + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

    # ── Vega (per 1 percentage-point of IV) ───────────────
    vega = S * pdf_d1 * sqrt_T / 100.0

    # ── Rho (per 1 percentage-point of rate) ──────────────
    if option_type == 'call':
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100.0

    return Greeks(
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        theta=round(theta, 4),
        vega=round(vega, 4),
        rho=round(rho, 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public pricing function
# ─────────────────────────────────────────────────────────────────────────────

def price_option(
    S: float,
    K: float,
    T: int,                              # calendar days to expiry
    sigma: float,                        # implied vol, e.g. 0.35 for 35%
    option_type: Literal['call', 'put'] = 'call',
    r: float = DEFAULT_RISK_FREE_RATE,
    symbol: str = "UNKNOWN",
) -> OptionResult:
    """
    Price an option and return full Greeks + diagnostics.

    Args:
        S           : Current stock price
        K           : Strike price
        T           : Days to expiry (calendar days)
        sigma       : Implied volatility as decimal (0.35 = 35%)
        option_type : 'call' or 'put'
        r           : Risk-free rate as decimal (default 5.3%)
        symbol      : Ticker for display

    Returns:
        OptionResult with price, Greeks, and trade diagnostics
    """
    if S <= 0 or K <= 0 or sigma <= 0:
        raise ValueError(f"Invalid inputs: S={S}, K={K}, sigma={sigma}")
    if T < 0:
        raise ValueError(f"T={T} days — can't price expired option")

    T_years = max(T, 0) / 365.0

    # Pricing
    if option_type == 'call':
        price     = _bs_call_price(S, K, T_years, r, sigma)
        intrinsic = max(S - K, 0.0)
    else:
        price     = _bs_put_price(S, K, T_years, r, sigma)
        intrinsic = max(K - S, 0.0)

    extrinsic = max(price - intrinsic, 0.0)

    # Greeks
    greeks = _compute_greeks(S, K, T_years, r, sigma, option_type)

    # Trade diagnostics
    otm_pct = ((K - S) / S * 100) if option_type == 'call' else ((S - K) / S * 100)
    breakeven = (K + price) if option_type == 'call' else (K - price)

    # Leverage: (delta × S) / price  — stock-equivalent exposure per dollar spent
    leverage_ratio = (abs(greeks.delta) * S / price) if price > 0 else 0.0

    # Daily theta as % of premium — tells you how fast the option is decaying
    theta_pct_daily = (abs(greeks.theta) / price * 100) if price > 0 else 0.0

    # Risk warnings
    warnings = []
    if theta_pct_daily > 3.0:
        warnings.append(f"High theta decay {theta_pct_daily:.1f}%/day — move must be fast")
    if otm_pct > 8.0:
        warnings.append(f"Deep OTM ({otm_pct:.1f}%) — needs large move to profit")
    if T <= 5:
        warnings.append(f"Only {T} DTE — theta accelerating, exit or roll now")
    if sigma > 0.70:
        warnings.append(f"Very high IV ({sigma*100:.0f}%) — premium is expensive")

    return OptionResult(
        symbol=symbol,
        option_type=option_type,
        S=round(S, 2),
        K=round(K, 2),
        T_days=T,
        sigma=sigma,
        r=r,
        theoretical_price=round(price, 4),
        intrinsic=round(intrinsic, 4),
        extrinsic=round(extrinsic, 4),
        contract_cost=round(price * 100, 2),
        greeks=greeks,
        otm_pct=round(otm_pct, 2),
        breakeven=round(breakeven, 2),
        leverage_ratio=round(leverage_ratio, 2),
        theta_pct_daily=round(theta_pct_daily, 2),
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Implied volatility solver
# ─────────────────────────────────────────────────────────────────────────────

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: int,                              # calendar days to expiry
    option_type: Literal['call', 'put'] = 'call',
    r: float = DEFAULT_RISK_FREE_RATE,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> Optional[float]:
    """
    Solve for implied volatility given the observed market price.
    Uses Newton-Raphson with Brent's method fallback.

    Returns IV as decimal (e.g. 0.35 for 35%), or None if no solution found.
    """
    T_years = max(T, 1) / 365.0

    # Sanity check — price must be above intrinsic
    if option_type == 'call':
        intrinsic = max(S - K * math.exp(-r * T_years), 0.0)
    else:
        intrinsic = max(K * math.exp(-r * T_years) - S, 0.0)

    if market_price < intrinsic - 0.01:
        logger.warning(f"Market price ${market_price:.2f} below intrinsic ${intrinsic:.2f}")
        return None

    # Initial sigma guess using Brenner-Subrahmanyam approximation
    sigma = (market_price / S) * math.sqrt(2 * math.pi / T_years)
    sigma = max(0.05, min(sigma, 3.0))

    def bs_price(s):
        if option_type == 'call':
            return _bs_call_price(S, K, T_years, r, s)
        return _bs_put_price(S, K, T_years, r, s)

    def bs_vega(s):
        d1, _ = _d1_d2(S, K, T_years, r, s)
        return S * norm.pdf(d1) * math.sqrt(T_years)

    # Newton-Raphson
    for i in range(max_iterations):
        price = bs_price(sigma)
        diff  = price - market_price

        if abs(diff) < tolerance:
            return round(sigma, 4)

        vega = bs_vega(sigma)
        if vega < 1e-10:
            break  # flat region, switch to bisection

        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 5.0))

    # Brent's method fallback if Newton fails
    try:
        from scipy.optimize import brentq
        iv = brentq(lambda s: bs_price(s) - market_price, 0.001, 5.0, xtol=1e-6)
        return round(float(iv), 4)
    except Exception:
        logger.warning(f"IV solver failed for market_price={market_price}, S={S}, K={K}, T={T}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Payoff projection — what the option is worth at various stock prices
# ─────────────────────────────────────────────────────────────────────────────

def payoff_at_expiry(
    K: float,
    premium_paid: float,
    option_type: Literal['call', 'put'] = 'call',
    stock_prices: Optional[np.ndarray] = None,
) -> dict:
    """
    Calculate P&L at expiry for a range of stock prices.
    Returns dict with arrays for plotting or analysis.

    Useful for visualising the risk/reward of a position before entering.
    """
    if stock_prices is None:
        # Default: K ± 20%
        low  = K * 0.80
        high = K * 1.20
        stock_prices = np.linspace(low, high, 100)

    if option_type == 'call':
        intrinsic = np.maximum(stock_prices - K, 0.0)
    else:
        intrinsic = np.maximum(K - stock_prices, 0.0)

    pnl_per_share = intrinsic - premium_paid
    pnl_contract  = pnl_per_share * 100  # 1 contract = 100 shares
    pnl_pct       = pnl_per_share / premium_paid * 100

    breakeven = K + premium_paid if option_type == 'call' else K - premium_paid

    return {
        'stock_prices':    stock_prices,
        'pnl_per_share':   pnl_per_share,
        'pnl_contract':    pnl_contract,
        'pnl_pct':         pnl_pct,
        'breakeven':       breakeven,
        'max_loss':        -premium_paid * 100,
        'max_loss_pct':    -100.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario analysis — P&L at a future date (not just expiry)
# ─────────────────────────────────────────────────────────────────────────────

def scenario_pnl(
    S_entry:     float,    # stock price when you bought
    K:           float,    # strike
    T_entry:     int,      # DTE when you bought
    premium_paid: float,   # what you paid per share
    sigma:       float,    # current IV
    option_type: Literal['call', 'put'] = 'call',
    r:           float = DEFAULT_RISK_FREE_RATE,
    days_held:   int = 8,
    stock_moves: Optional[list] = None,   # % moves to scenario, e.g. [-5, 0, 3, 7, 10]
) -> list:
    """
    Show P&L at a specific future date (not expiry) for different stock moves.
    This is the key analysis for your 8-day hold strategy.

    Returns list of scenario dicts with move %, option value, P&L, P&L%.
    """
    if stock_moves is None:
        stock_moves = [-10, -7, -5, -3, 0, 3, 5, 7, 10, 15]

    T_exit = max(T_entry - days_held, 0)
    scenarios = []

    for move_pct in stock_moves:
        S_exit = S_entry * (1 + move_pct / 100)
        result = price_option(S_exit, K, T_exit, sigma, option_type, r)
        option_value = result.theoretical_price

        pnl_share    = option_value - premium_paid
        pnl_contract = pnl_share * 100
        pnl_pct      = pnl_share / premium_paid * 100 if premium_paid > 0 else 0.0

        scenarios.append({
            'stock_move_pct':  move_pct,
            'S_exit':          round(S_exit, 2),
            'option_value':    round(option_value, 4),
            'pnl_per_share':   round(pnl_share, 4),
            'pnl_contract':    round(pnl_contract, 2),
            'pnl_pct':         round(pnl_pct, 1),
            'option_delta':    result.greeks.delta,
        })

    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Quick diagnostics print
# ─────────────────────────────────────────────────────────────────────────────

def print_scenario_table(scenarios: list, days_held: int = 8) -> None:
    """Pretty-print the scenario P&L table."""
    print(f"\n{'Stock Move':>12} {'Stock Price':>12} {'Option Value':>13} "
          f"{'P&L/contract':>13} {'P&L %':>8}")
    print("─" * 62)
    for s in scenarios:
        marker = " ◄ target" if s['stock_move_pct'] == 7 else ""
        print(
            f"{s['stock_move_pct']:>+11.0f}%  "
            f"${s['S_exit']:>10.2f}  "
            f"${s['option_value']:>11.2f}  "
            f"${s['pnl_contract']:>+11.2f}  "
            f"{s['pnl_pct']:>+7.1f}%"
            f"{marker}"
        )
    print("─" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== BLACK-SCHOLES ENGINE SELF-TEST ===\n")

    # --- Example matching your system's use case ---
    # Stock at $150, ML model says +7% move in 8 days
    # Buy call 4.7% OTM (strike $157), 12 DTE, IV=35%

    S     = 150.00
    K     = 157.00
    T     = 12        # 12 days to expiry
    sigma = 0.35      # 35% IV
    prem  = None

    result = price_option(S=S, K=K, T=T, sigma=sigma, symbol="EXAMPLE")
    print(result)

    prem = result.theoretical_price

    # --- IV round-trip test ---
    iv_solved = implied_volatility(market_price=prem, S=S, K=K, T=T)
    print(f"IV round-trip: input={sigma:.4f}  solved={iv_solved:.4f}  "
          f"diff={abs(sigma - iv_solved):.6f}")
    assert abs(sigma - iv_solved) < 0.001, "IV solver round-trip failed"
    print("✅ IV solver OK\n")

    # --- 8-day scenario table ---
    print(f"P&L scenarios after {8} days (stock moves, option still has {T-8} DTE):\n")
    scenarios = scenario_pnl(
        S_entry=S,
        K=K,
        T_entry=T,
        premium_paid=prem,
        sigma=sigma,
        days_held=8,
    )
    print_scenario_table(scenarios, days_held=8)

    # --- Greeks check (ATM option) ---
    print("\n--- ATM option Greeks check ---")
    atm = price_option(S=100, K=100, T=30, sigma=0.30, symbol="ATM_TEST")
    print(atm)
    print(f"Delta ATM call should be ~0.50: {atm.greeks.delta:.3f}  "
          f"{'✅' if 0.45 < atm.greeks.delta < 0.55 else '❌'}")
