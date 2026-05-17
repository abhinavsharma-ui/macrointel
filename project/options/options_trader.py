"""
Options Trader — Full Integration Layer
=========================================
Connects all pieces into one runnable system:

  macrointel-catalyst/catalyst.db   → catalyst score (Form 4, ClinicalTrials, etc.)
  signal_generator.py               → ML confidence score
  screener/screener.py              → UOA confirmation (optional)
  iv_rank.py                        → IV rank gate
  strike_selector.py                → contract selection + Kelly sizing
  ↓
  OptionsTrader.run()               → entry signals with full contract specs
  OptionsTrader.check_exits()       → exit logic for open positions

Exit rules (when to sell the option):
  1. 2× premium → take profit (never let a winner fully decay)
  2. 50% loss on premium → stop loss (options go to zero fast)
  3. Day 8 → time exit regardless (matches your stock system's hold period)
  4. < 3 DTE → exit regardless (theta acceleration is too dangerous)

Usage (in your existing cron / 8:30pm run):
    from options.options_trader import OptionsTrader

    trader = OptionsTrader(
        catalyst_db_path="/home/abhinavsharma1359/macrointel-catalyst/catalyst.db",
        portfolio_value=50_000,
    )
    signals = trader.run()
    exits   = trader.check_exits()
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from options.black_scholes import price_option, scenario_pnl, print_scenario_table
from options.iv_rank import IVRankCalculator
from options.strike_selector import StrikeSelector, ContractSpec, delta_adjusted_kelly

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Signal thresholds — same logic as your stock system
MIN_ML_CONFIDENCE   = 0.55   # minimum ML confidence to consider
MIN_CATALYST_SCORE  = 0.50   # minimum catalyst score from catalyst.db
MIN_COMBINED_SCORE  = 0.60   # weighted combination must clear this

# IV rank gate
MAX_IVR_ENTRY       = 50.0   # skip if IV rank > 50%

# Portfolio risk
DEFAULT_PORTFOLIO   = 50_000.0
MAX_RISK_PER_TRADE  = 0.02   # max 2% of portfolio per options trade
MAX_OPEN_POSITIONS  = 5      # maximum concurrent options positions
KELLY_FRACTION      = 0.25   # quarter-Kelly (conservative for options)

# Exit thresholds
PROFIT_TAKE_MULT    = 2.0    # exit when option doubles (200% gain)
STOP_LOSS_MULT      = 0.50   # exit when option loses 50% of premium
MAX_HOLD_DAYS       = 8      # force exit at day 8
MIN_DTE_EXIT        = 3      # force exit if < 3 DTE remaining

# Positions file (persists across cron runs)
POSITIONS_FILE      = "options_positions.json"


# ─────────────────────────────────────────────────────────────────────────────
# Position tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionsPosition:
    """An open options position."""
    symbol:          str
    direction:       str          # 'call' or 'put'
    strike:          float
    expiry:          str          # YYYY-MM-DD
    dte_at_entry:    int
    entry_date:      str          # ISO datetime
    entry_premium:   float        # per share (what you paid)
    contracts:       int          # number of contracts
    total_cost:      float        # entry_premium × contracts × 100
    current_premium: float = 0.0  # updated daily
    pnl:             float = 0.0
    pnl_pct:         float = 0.0
    days_held:       int   = 0
    status:          str   = 'open'  # 'open', 'closed_profit', 'closed_loss', 'closed_time'
    exit_date:       str   = ''
    exit_premium:    float = 0.0
    exit_reason:     str   = ''

    # Signal context (for logging/analysis)
    ml_confidence:   float = 0.0
    catalyst_score:  float = 0.0
    iv_rank_entry:   float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Catalyst reader
# ─────────────────────────────────────────────────────────────────────────────

class CatalystReader:
    """
    Reads catalyst signals from macrointel-catalyst/catalyst.db.
    Maps to your existing Form 4 / ClinicalTrials / OpenInsider / EDGAR data.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            logger.warning(f"Catalyst DB not found: {db_path}")

    def get_catalyst_scores(self, lookback_hours: int = 24) -> Dict[str, float]:
        """
        Get catalyst scores for symbols with recent activity.
        Returns {symbol: score} where score is 0-1.

        Schema (confirmed from catalyst.db inspection):
          table  : catalyst_events
          ticker : ticker
          score  : strength  (0.0 - 1.0)
          time   : ingested_at
          extra  : event_type, source, headline, payload

        Also reads option_baselines for UOA confirmation bonus.
        """
        if not self.db_path.exists():
            return {}

        scores: Dict[str, float] = {}

        try:
            conn   = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # ── Primary: catalyst_events with strength score ───────────────
            # Lookback window — for options we want signals from last 48h
            # (catalyst may fire a day before the move)
            cutoff_primary = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()
            cutoff_wide    = (datetime.now() - timedelta(hours=48)).isoformat()

            cursor.execute(f"""
                SELECT
                    ticker,
                    MAX(strength)                          AS max_strength,
                    COUNT(*)                               AS event_count,
                    GROUP_CONCAT(DISTINCT event_type)      AS event_types,
                    GROUP_CONCAT(DISTINCT source)          AS sources
                FROM catalyst_events
                WHERE ingested_at >= ?
                  AND strength IS NOT NULL
                  AND strength > 0
                GROUP BY ticker
                ORDER BY max_strength DESC
            """, (cutoff_wide,))

            rows = cursor.fetchall()
            for row in rows:
                ticker      = str(row[0]).upper().strip()
                strength    = float(row[1] or 0)
                count       = int(row[2] or 1)
                event_types = str(row[3] or '')
                sources     = str(row[4] or '')

                if not ticker or strength <= 0:
                    continue

                # Boost score for multiple corroborating events
                count_bonus = min(0.10, (count - 1) * 0.03)

                # Boost for high-conviction event types
                type_bonus = 0.0
                if 'insider' in event_types.lower() or 'form4' in event_types.lower():
                    type_bonus += 0.05
                if 'clinical' in event_types.lower() or 'trial' in event_types.lower():
                    type_bonus += 0.05
                if 'edgar' in event_types.lower() or 'sec' in event_types.lower():
                    type_bonus += 0.03

                final_score = min(1.0, strength + count_bonus + type_bonus)
                scores[ticker] = round(final_score, 4)

            # ── UOA bonus: option_baselines unusual activity ───────────────
            # If call volume is anomalously high vs baseline, add a small boost
            try:
                cursor.execute("""
                    SELECT ticker,
                           call_volume,
                           put_volume
                    FROM option_baselines
                    WHERE as_of_date >= date('now', '-2 days')
                """)
                uoa_rows = cursor.fetchall()
                for urow in uoa_rows:
                    ticker     = str(urow[0]).upper().strip()
                    call_vol   = float(urow[1] or 0)
                    put_vol    = float(urow[2] or 0)
                    total      = call_vol + put_vol
                    if total > 0 and call_vol / total > 0.65:
                        # Unusual call skew — smart money buying calls
                        if ticker in scores:
                            scores[ticker] = min(1.0, scores[ticker] + 0.08)
                        else:
                            scores[ticker] = 0.55   # UOA-only signal
            except sqlite3.OperationalError:
                pass  # option_baselines may be empty

            conn.close()

        except Exception as e:
            logger.error(f"Catalyst DB read failed: {e}")

        logger.info(f"Catalyst scores loaded: {len(scores)} symbols with activity")
        return scores

    def get_table_names(self) -> List[str]:
        """Inspect what tables exist in catalyst.db."""
        if not self.db_path.exists():
            return []
        try:
            conn   = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]
            conn.close()
            return tables
        except Exception as e:
            logger.error(f"Table inspection failed: {e}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# ML signal reader
# ─────────────────────────────────────────────────────────────────────────────

class MLSignalReader:
    """
    Reads ML confidence scores from your existing signal_generator output.
    Falls back to reading the signal cache or running inference directly.
    """

    def __init__(self, signals_cache_path: Optional[str] = None):
        self.cache_path = Path(signals_cache_path) if signals_cache_path else None

    # Common locations where your pipeline writes signal output
    _SIGNAL_SEARCH_PATHS = [
        "data/signals.json",
        "data/runtime_state.json",
        "data/latest_signals.json",
        "logs/signals.json",
        "signals.json",
    ]

    def get_ml_signals(self) -> Dict[str, dict]:
        """
        Returns {symbol: {'confidence': float, 'signal': str, 'conviction': float}}
        Reads from the SIGNALS global in signal_generator or a JSON/JSONL cache.
        """
        # Try explicit cache path first
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    raw = json.load(f)
                # Handle both list and dict formats
                if isinstance(raw, list):
                    result = {s['symbol']: s for s in raw if s.get('signal') == 'buy'}
                elif isinstance(raw, dict):
                    # runtime_state.json may have signals nested
                    signals_list = raw.get('signals', raw.get('buy_signals', []))
                    result = {s['symbol']: s for s in signals_list if s.get('signal') == 'buy'}
                else:
                    result = {}
                logger.info(f"ML signals loaded from {self.cache_path}: {len(result)} buy signals")
                return result
            except Exception as e:
                logger.warning(f"ML cache read failed: {e}")

        # Search common paths relative to the project directory
        project_root = Path(__file__).parents[1]
        for rel_path in self._SIGNAL_SEARCH_PATHS:
            candidate = project_root / rel_path
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        content = f.read().strip()
                    # Handle JSONL format (one JSON object per line)
                    if content.startswith('{'):
                        raw = json.loads(content)
                        signals_list = raw.get('signals', raw.get('buy_signals', []))
                        if isinstance(signals_list, list):
                            result = {s['symbol']: s for s in signals_list if s.get('signal') == 'buy'}
                            if result:
                                logger.info(f"ML signals from {candidate}: {len(result)} buy signals")
                                return result
                    elif content.startswith('['):
                        raw = json.loads(content)
                        result = {s['symbol']: s for s in raw if s.get('signal') == 'buy'}
                        if result:
                            logger.info(f"ML signals from {candidate}: {len(result)} buy signals")
                            return result
                except Exception as e:
                    logger.debug(f"Signal read failed at {candidate}: {e}")

        # Try importing directly from signal_generator (works if cron just ran)
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parents[1]))
            from pipeline.signal_generator import SIGNALS
            result = {}
            for sig in SIGNALS:
                if sig.get('signal') == 'buy' and sig.get('confidence', 0) >= MIN_ML_CONFIDENCE:
                    result[sig['symbol']] = sig
            if result:
                logger.info(f"ML signals from live generator: {len(result)} buy signals")
                return result
        except Exception as e:
            logger.debug(f"Direct ML import failed: {e}")

        logger.info("ML signals: 0 — pipeline hasn't run yet today or no buy signals")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Main Options Trader
# ─────────────────────────────────────────────────────────────────────────────

class OptionsTrader:
    """
    Full options trading system that plugs into your existing pipeline.

    run()         → generate new options entry signals
    check_exits() → evaluate open positions against exit rules
    """

    def __init__(
        self,
        catalyst_db_path:   str = "/home/abhinavsharma1359/macrointel-catalyst/catalyst.db",
        ml_cache_path:      Optional[str] = None,
        positions_file:     str = POSITIONS_FILE,
        portfolio_value:    float = DEFAULT_PORTFOLIO,
        max_ivr:            float = MAX_IVR_ENTRY,
    ):
        self.portfolio_value  = portfolio_value
        self.positions_file   = Path(positions_file)

        self.catalyst_reader  = CatalystReader(catalyst_db_path)
        self.ml_reader        = MLSignalReader(ml_cache_path)
        self.iv_calc          = IVRankCalculator(max_ivr=max_ivr)
        self.strike_selector  = StrikeSelector(use_live_chain=True)

        self._positions: List[OptionsPosition] = []
        self._load_positions()

        logger.info(f"OptionsTrader initialised | portfolio=${portfolio_value:,.0f} | "
                    f"open positions={len(self.open_positions)}")

    # ── Position persistence ──────────────────────────────────────────────

    @property
    def open_positions(self) -> List[OptionsPosition]:
        return [p for p in self._positions if p.status == 'open']

    def _load_positions(self):
        if self.positions_file.exists():
            try:
                with open(self.positions_file) as f:
                    raw = json.load(f)
                self._positions = [OptionsPosition(**p) for p in raw]
                logger.info(f"Loaded {len(self._positions)} positions from {self.positions_file}")
            except Exception as e:
                logger.error(f"Position load failed: {e}")
                self._positions = []

    def _save_positions(self):
        try:
            with open(self.positions_file, 'w') as f:
                json.dump([asdict(p) for p in self._positions], f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Position save failed: {e}")

    # ── Signal combination ────────────────────────────────────────────────

    def _combined_score(
        self,
        ml_confidence:  float,
        catalyst_score: float,
        iv_rank:        float,   # 0-100
    ) -> float:
        """
        Combine ML confidence + catalyst score into a final entry score.
        IV rank acts as a multiplier penalty (higher IV rank = reduce score).

        Weights (calibrated to mirror your stock system):
          ML confidence : 60%
          Catalyst score: 40%
          IV penalty    : up to -20% if IV rank is high
        """
        base  = 0.60 * ml_confidence + 0.40 * catalyst_score
        # IV rank penalty: 0 at IVR=0, -0.20 at IVR=100
        iv_penalty = (iv_rank / 100) * 0.20
        return float(np.clip(base - iv_penalty, 0, 1))

    # ── Entry signal generation ───────────────────────────────────────────

    def run(
        self,
        expected_move_pct: float = 7.0,
        hold_days:         int   = 8,
        verbose:           bool  = True,
    ) -> List[dict]:
        """
        Main entry point. Call this from your 8:30pm cron.

        1. Read ML buy signals
        2. Read catalyst scores
        3. Cross-reference: only trade symbols with both signals
        4. IV rank gate
        5. Select strike/expiry
        6. Size via delta-adjusted Kelly
        7. Log and return trade specs

        Returns list of trade dicts ready to execute.
        """
        print(f"\n{'═'*60}")
        print(f"  OPTIONS TRADER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Portfolio: ${self.portfolio_value:,.0f}   "
              f"Open: {len(self.open_positions)}/{MAX_OPEN_POSITIONS}")
        print(f"{'═'*60}\n")

        # Check capacity
        capacity = MAX_OPEN_POSITIONS - len(self.open_positions)
        if capacity <= 0:
            print("⛔ Max open positions reached — no new entries")
            return []

        # 1. Get signals
        ml_signals      = self.ml_reader.get_ml_signals()
        catalyst_scores = self.catalyst_reader.get_catalyst_scores()

        if not ml_signals:
            print("ℹ No ML buy signals today")
            return []

        print(f"ML signals: {len(ml_signals)} buy  |  "
              f"Catalyst events: {len(catalyst_scores)} symbols\n")

        # 2. Find overlap
        candidates = []
        for symbol, ml in ml_signals.items():
            ml_conf    = float(ml.get('confidence', 0))
            cat_score  = catalyst_scores.get(symbol, 0.0)
            if ml_conf < MIN_ML_CONFIDENCE:
                continue
            if cat_score < MIN_CATALYST_SCORE and len(catalyst_scores) > 0:
                # If we have catalyst data but this symbol isn't in it, skip
                logger.debug(f"{symbol}: skipped — no catalyst signal ({cat_score:.2f})")
                continue
            candidates.append({
                'symbol':       symbol,
                'ml_confidence': ml_conf,
                'catalyst_score': cat_score,
                'conviction':   float(ml.get('conviction_score', 5)),
            })

        # If no catalyst DB overlap (new setup), fall through with ML-only signals
        if not candidates and catalyst_scores:
            print("⚠ No symbols in both ML and catalyst signals — check DB connection")
            return []
        elif not candidates:
            # No catalyst data available — use ML-only mode
            candidates = [
                {
                    'symbol':        s,
                    'ml_confidence': float(ml.get('confidence', 0)),
                    'catalyst_score': 0.5,   # default when no catalyst DB
                    'conviction':    float(ml.get('conviction_score', 5)),
                }
                for s, ml in ml_signals.items()
                if float(ml.get('confidence', 0)) >= MIN_ML_CONFIDENCE
            ]
            print("ℹ Running ML-only mode (no catalyst DB)")

        # Sort by ML confidence descending
        candidates.sort(key=lambda x: x['ml_confidence'], reverse=True)

        trades = []
        for cand in candidates[:capacity * 2]:   # evaluate 2× capacity to find best
            if len(trades) >= capacity:
                break

            symbol     = cand['symbol']
            ml_conf    = cand['ml_confidence']
            cat_score  = cand['catalyst_score']

            # Skip if already in position
            if any(p.symbol == symbol for p in self.open_positions):
                logger.debug(f"{symbol}: already in position")
                continue

            # 3. IV rank gate
            iv_result = self.iv_calc.get_iv_rank(symbol)
            iv_rank   = iv_result.iv_rank if iv_result else 50.0
            current_iv = iv_result.current_iv if iv_result else 0.35

            if iv_result and not iv_result.buy_signal:
                print(f"⛔ {symbol}: IV rank too high ({iv_rank:.0f}%) — skipping")
                continue

            # 4. Combined score gate
            combined = self._combined_score(ml_conf, cat_score, iv_rank)
            if combined < MIN_COMBINED_SCORE:
                print(f"⛔ {symbol}: combined score {combined:.2f} < {MIN_COMBINED_SCORE}")
                continue

            # 5. Select contract
            try:
                import yfinance as yf
                info = yf.Ticker(symbol).fast_info
                S    = getattr(info, 'last_price', None) or getattr(info, 'regularMarketPrice', None)
                if not S or S <= 0:
                    logger.warning(f"{symbol}: can't get current price")
                    continue
            except Exception:
                continue

            contract = self.strike_selector.select(
                symbol=symbol,
                current_price=float(S),
                expected_move_pct=expected_move_pct,
                hold_days=hold_days,
                direction='call',
                current_iv=current_iv,
            )
            if contract is None:
                print(f"⚠ {symbol}: no valid contract found")
                continue

            # 6. Size position
            sizing = delta_adjusted_kelly(
                win_prob=ml_conf,
                expected_move_pct=expected_move_pct,
                contract=contract,
                portfolio_value=self.portfolio_value,
                max_risk_pct=MAX_RISK_PER_TRADE,
                kelly_fraction=KELLY_FRACTION,
            )

            if sizing['contracts'] == 0:
                continue

            # 7. Build trade spec
            trade = {
                'symbol':          symbol,
                'action':          'BUY_CALL',
                'strike':          contract.strike,
                'expiry':          contract.expiry,
                'dte':             contract.dte,
                'contracts':       sizing['contracts'],
                'limit_price':     round(contract.theoretical_price * 1.02, 2),  # 2% above mid
                'total_cost':      sizing['premium_risk'],
                'delta':           contract.delta,
                'iv_rank':         iv_rank,
                'combined_score':  round(combined, 3),
                'ml_confidence':   ml_conf,
                'catalyst_score':  cat_score,
                'theta_pct_daily': contract.theta_pct_daily,
                'breakeven_move':  contract.breakeven_move_pct,
                'score':           contract.score,
                'sizing':          sizing,
                'timestamp':       datetime.now().isoformat(),
                'warnings':        contract.warnings,
            }
            trades.append(trade)

            # 8. Print and record position
            if verbose:
                print(contract)
                print(f"  ML: {ml_conf:.2f}  Catalyst: {cat_score:.2f}  "
                      f"IVR: {iv_rank:.0f}%  Combined: {combined:.2f}")
                print(f"  Size: {sizing['contracts']} contract(s)  "
                      f"Risk: ${sizing['premium_risk']:,.0f}  "
                      f"({sizing['premium_risk_pct']:.1f}% portfolio)\n")

            # Track position
            position = OptionsPosition(
                symbol=symbol,
                direction='call',
                strike=contract.strike,
                expiry=contract.expiry,
                dte_at_entry=contract.dte,
                entry_date=datetime.now().isoformat(),
                entry_premium=contract.theoretical_price,
                contracts=sizing['contracts'],
                total_cost=sizing['premium_risk'],
                current_premium=contract.theoretical_price,
                ml_confidence=ml_conf,
                catalyst_score=cat_score,
                iv_rank_entry=iv_rank,
            )
            self._positions.append(position)

        self._save_positions()

        print(f"\n{'─'*60}")
        print(f"  New trades: {len(trades)}  |  Total open: {len(self.open_positions)}")
        print(f"{'─'*60}\n")
        return trades

    # ── Exit logic ────────────────────────────────────────────────────────

    def check_exits(self, verbose: bool = True) -> List[dict]:
        """
        Check all open positions against exit rules.
        Call this from your daily cron alongside run().

        Exit rules:
          1. Option value doubled → take profit
          2. Option lost 50% → stop loss
          3. Day 8 → time exit
          4. < 3 DTE → expire protection
        """
        if not self.open_positions:
            return []

        exits = []
        print(f"\n{'─'*60}")
        print(f"  EXIT CHECK — {len(self.open_positions)} open position(s)")
        print(f"{'─'*60}")

        for pos in self.open_positions:
            # Update current premium from live market
            current_prem = self._get_current_premium(pos)
            if current_prem is None:
                current_prem = pos.current_premium  # use last known if fetch fails

            pos.current_premium = current_prem
            pos.pnl             = (current_prem - pos.entry_premium) * pos.contracts * 100
            pos.pnl_pct         = (current_prem - pos.entry_premium) / pos.entry_premium * 100
            pos.days_held       = (datetime.now() - datetime.fromisoformat(pos.entry_date)).days

            # Current DTE
            exp_date    = datetime.strptime(pos.expiry, "%Y-%m-%d")
            days_to_exp = max((exp_date - datetime.now()).days, 0)

            # Evaluate exit conditions
            exit_reason = None
            if current_prem >= pos.entry_premium * PROFIT_TAKE_MULT:
                exit_reason = f"TAKE_PROFIT ({pos.pnl_pct:+.0f}%)"
                pos.status  = 'closed_profit'
            elif current_prem <= pos.entry_premium * STOP_LOSS_MULT:
                exit_reason = f"STOP_LOSS ({pos.pnl_pct:+.0f}%)"
                pos.status  = 'closed_loss'
            elif pos.days_held >= MAX_HOLD_DAYS:
                exit_reason = f"TIME_EXIT (day {pos.days_held})"
                pos.status  = 'closed_time'
            elif days_to_exp <= MIN_DTE_EXIT:
                exit_reason = f"EXPIRY_PROTECTION ({days_to_exp} DTE)"
                pos.status  = 'closed_time'

            if exit_reason:
                pos.exit_date    = datetime.now().isoformat()
                pos.exit_premium = current_prem
                pos.exit_reason  = exit_reason

                exit_spec = {
                    'symbol':       pos.symbol,
                    'action':       'SELL_TO_CLOSE',
                    'strike':       pos.strike,
                    'expiry':       pos.expiry,
                    'contracts':    pos.contracts,
                    'limit_price':  round(current_prem * 0.98, 2),  # 2% below mid
                    'pnl':          pos.pnl,
                    'pnl_pct':      pos.pnl_pct,
                    'exit_reason':  exit_reason,
                    'days_held':    pos.days_held,
                    'timestamp':    datetime.now().isoformat(),
                }
                exits.append(exit_spec)

                if verbose:
                    icon = "✅" if pos.pnl >= 0 else "❌"
                    print(f"  {icon} EXIT {pos.symbol} ${pos.strike} {pos.expiry}  "
                          f"P&L: ${pos.pnl:+,.2f} ({pos.pnl_pct:+.1f}%)  "
                          f"Reason: {exit_reason}")
            else:
                if verbose:
                    print(f"  📊 HOLD {pos.symbol} ${pos.strike} {pos.expiry}  "
                          f"Day {pos.days_held}  P&L: ${pos.pnl:+,.2f} ({pos.pnl_pct:+.1f}%)  "
                          f"{days_to_exp} DTE")

        self._save_positions()

        if exits:
            print(f"\n  → {len(exits)} exit order(s) to execute")
        print(f"{'─'*60}\n")
        return exits

    def _get_current_premium(self, pos: OptionsPosition) -> Optional[float]:
        """Fetch current mid-price for an open options position."""
        try:
            import yfinance as yf
            ticker  = yf.Ticker(pos.symbol)
            chain   = ticker.option_chain(pos.expiry)
            options = chain.calls if pos.direction == 'call' else chain.puts
            row     = options[options['strike'] == pos.strike]
            if row.empty:
                return None
            bid = float(row['bid'].iloc[0] or 0)
            ask = float(row['ask'].iloc[0] or 0)
            if bid > 0 and ask > 0:
                return round((bid + ask) / 2, 4)
            return float(row['lastPrice'].iloc[0] or 0) or None
        except Exception as e:
            logger.debug(f"Premium fetch failed for {pos.symbol}: {e}")
            return None

    # ── Reporting ─────────────────────────────────────────────────────────

    def print_portfolio(self):
        """Print current portfolio status."""
        open_pos   = self.open_positions
        closed_pos = [p for p in self._positions if p.status != 'open']

        print(f"\n{'═'*60}")
        print(f"  OPTIONS PORTFOLIO SUMMARY")
        print(f"{'═'*60}")
        print(f"  Open:   {len(open_pos)}")
        print(f"  Closed: {len(closed_pos)}")

        if open_pos:
            total_cost  = sum(p.total_cost for p in open_pos)
            total_pnl   = sum(p.pnl for p in open_pos)
            print(f"\n  Open Positions (capital at risk: ${total_cost:,.0f}):")
            for p in open_pos:
                print(f"    {p.symbol:8s} ${p.strike} {p.expiry}  "
                      f"×{p.contracts}  Day {p.days_held}  "
                      f"P&L: ${p.pnl:+,.2f} ({p.pnl_pct:+.1f}%)")
            print(f"    {'─'*50}")
            print(f"    Total unrealised: ${total_pnl:+,.2f}")

        if closed_pos:
            wins   = [p for p in closed_pos if p.pnl > 0]
            losses = [p for p in closed_pos if p.pnl <= 0]
            total  = sum(p.pnl for p in closed_pos)
            wr     = len(wins) / len(closed_pos) * 100 if closed_pos else 0
            print(f"\n  Closed Positions:")
            print(f"    Win rate: {wr:.0f}%  ({len(wins)}W / {len(losses)}L)")
            print(f"    Total realised P&L: ${total:+,.2f}")

        print(f"{'═'*60}\n")

    def inspect_catalyst_db(self):
        """Helper: show what tables/columns exist in catalyst.db."""
        tables = self.catalyst_reader.get_table_names()
        print(f"\nCatalyst DB tables: {tables}")
        if tables:
            try:
                conn   = sqlite3.connect(self.catalyst_reader.db_path)
                cursor = conn.cursor()
                for t in tables:
                    cursor.execute(f"PRAGMA table_info({t})")
                    cols = [r[1] for r in cursor.fetchall()]
                    cursor.execute(f"SELECT COUNT(*) FROM {t}")
                    n = cursor.fetchone()[0]
                    print(f"  {t}: {n} rows | columns: {cols}")
                conn.close()
            except Exception as e:
                print(f"  Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run from cron
# ─────────────────────────────────────────────────────────────────────────────

def run_options_cron(
    catalyst_db:    str = "/home/abhinavsharma1359/macrointel-catalyst/catalyst.db",
    portfolio:      float = DEFAULT_PORTFOLIO,
    ml_cache:       Optional[str] = None,
    check_exits:    bool = True,
    new_entries:    bool = True,
) -> dict:
    """
    Drop-in replacement for your existing 8:30pm cron logic.
    Call this alongside your stock signal generation.

    Returns dict with 'entries' and 'exits' lists.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    trader = OptionsTrader(
        catalyst_db_path=catalyst_db,
        ml_cache_path=ml_cache,
        portfolio_value=portfolio,
    )

    result = {'entries': [], 'exits': []}

    if check_exits:
        result['exits'] = trader.check_exits()

    if new_entries:
        result['entries'] = trader.run()

    trader.print_portfolio()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Self-test / DB inspection
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    db_path = "/home/abhinavsharma1359/macrointel-catalyst/catalyst.db"

    trader = OptionsTrader(
        catalyst_db_path=db_path,
        portfolio_value=50_000,
    )

    # First: inspect what's in the DB so we can tune the SQL
    print("\n=== CATALYST DB INSPECTION ===")
    trader.inspect_catalyst_db()

    # Run (will use ML-only mode if no catalyst overlap)
    print("\n=== RUNNING OPTIONS CRON ===")
    result = run_options_cron(
        catalyst_db=db_path,
        portfolio=50_000,
    )

    print(f"\nEntries: {len(result['entries'])}")
    print(f"Exits:   {len(result['exits'])}")
