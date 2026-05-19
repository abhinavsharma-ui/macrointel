"""
Options Trading Module
======================
Adds an options layer on top of the existing ML + catalyst signal system.

Modules:
  black_scholes   — Black-Scholes pricing + all Greeks + IV solver
  iv_rank         — IV Rank and IV Percentile (buy when cheap)
  strike_selector — Pick optimal strike/expiry + delta-adjusted Kelly sizing
  options_trader  — Full integration: catalyst.db + ML + IV gate + execution

Quick start:
    cd /home/abhinavsharma1359/macro_intelligence_complete
    python -m project.options.black_scholes      # test pricing engine
    python -m project.options.iv_rank            # test IV rank for AAPL/MSFT
    python -m project.options.strike_selector    # test contract selection
    python -m project.options.options_trader     # inspect DB + run full flow
"""

from options.black_scholes import price_option, implied_volatility, scenario_pnl
from options.iv_rank import IVRankCalculator, iv_rank_filter
from options.strike_selector import StrikeSelector, delta_adjusted_kelly
from options.options_trader import OptionsTrader, run_options_cron
from options.backtester import OptionsBacktester, BacktestResults
from options.universe import LiquidUniverse, LIQUID_UNIVERSE
from options.tradier_client import TradierClient, get_options_chain_yf_compat

__all__ = [
    'price_option',
    'implied_volatility',
    'scenario_pnl',
    'IVRankCalculator',
    'iv_rank_filter',
    'StrikeSelector',
    'delta_adjusted_kelly',
    'OptionsTrader',
    'run_options_cron',
    'OptionsBacktester',
    'BacktestResults',
    'LiquidUniverse',
    'LIQUID_UNIVERSE',
    'TradierClient',
    'get_options_chain_yf_compat',
]
