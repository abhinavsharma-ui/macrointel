"""
Tradier Sandbox API Client
===========================
Free, reliable options data — replaces yfinance for options chains.

Why Tradier instead of yfinance?
  - yfinance returns incorrect/stale IV on low-OI contracts
  - Tradier sandbox has real-time options chains (delayed 15min but accurate)
  - Tradier has accurate Greeks pre-computed on the server side
  - No rate limits on sandbox

Setup (one-time, free):
  1. Go to https://developer.tradier.com/user/sign_up
  2. Create a free account
  3. Copy your sandbox token from the dashboard
  4. Set it as env variable:
       export TRADIER_TOKEN="your_token_here"
  Or pass it directly: TradierClient(token="your_token_here")

Sandbox vs Production:
  - Sandbox URL: sandbox.tradier.com (free, delayed, for paper trading)
  - Production URL: api.tradier.com (paid, real-time, for live trading)
  Switch by setting: TradierClient(sandbox=False, token=YOUR_LIVE_TOKEN)

Usage:
    from options.tradier_client import TradierClient

    client = TradierClient()  # reads TRADIER_TOKEN from env

    # Get full options chain
    chain = client.get_options_chain("AAPL", "2025-06-20")
    print(chain.head())

    # Get all available expiry dates
    expirations = client.get_expirations("AAPL")

    # Get current stock quote
    quote = client.get_quote("AAPL")
    print(quote['last'])  # current price
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SANDBOX_BASE  = "https://sandbox.tradier.com/v1"
PROD_BASE     = "https://api.tradier.com/v1"

RATE_LIMIT_DELAY = 0.2   # 200ms between requests (well within free tier limits)


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

class TradierClient:
    """
    Lightweight Tradier API wrapper for options data.
    Falls back gracefully to yfinance if Tradier is unavailable.
    """

    def __init__(
        self,
        token:   Optional[str] = None,
        sandbox: bool = True,
    ):
        self.token   = token or os.environ.get("TRADIER_TOKEN", "")
        self.base    = SANDBOX_BASE if sandbox else PROD_BASE
        self.sandbox = sandbox
        self._session = requests.Session()
        self._last_call = 0.0

        if not self.token:
            logger.warning(
                "No TRADIER_TOKEN found. Set env var or pass token= argument.\n"
                "  Get a free token at: https://developer.tradier.com/user/sign_up\n"
                "  Falling back to yfinance for options data."
            )
        else:
            mode = "sandbox" if sandbox else "production"
            logger.info(f"TradierClient initialised ({mode})")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept":        "application/json",
        }

    def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Rate-limited GET with error handling."""
        if not self.token:
            return None

        # rate limit
        elapsed = time.time() - self._last_call
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_call = time.time()

        url = f"{self.base}{endpoint}"
        try:
            resp = self._session.get(url, headers=self._headers(), params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Tradier HTTP error {e.response.status_code} → {url}: {e}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Tradier connection error — are you online?")
        except Exception as e:
            logger.error(f"Tradier request failed: {e}")
        return None

    # ── Quotes ────────────────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Optional[dict]:
        """
        Get current stock quote.
        Returns dict with: symbol, last, bid, ask, volume, change_pct
        """
        data = self._get("/markets/quotes", {"symbols": symbol, "greeks": "false"})
        if not data:
            return None
        try:
            quotes = data.get("quotes", {}).get("quote")
            if isinstance(quotes, list):
                quotes = quotes[0]
            if quotes:
                return {
                    "symbol":     quotes.get("symbol"),
                    "last":       float(quotes.get("last") or 0),
                    "bid":        float(quotes.get("bid") or 0),
                    "ask":        float(quotes.get("ask") or 0),
                    "volume":     int(quotes.get("volume") or 0),
                    "change_pct": float(quotes.get("change_percentage") or 0),
                }
        except Exception as e:
            logger.error(f"Quote parse error for {symbol}: {e}")
        return None

    # ── Options expirations ────────────────────────────────────────────────────

    def get_expirations(self, symbol: str) -> List[str]:
        """
        Returns list of available expiry dates for symbol.
        Format: ["2025-06-20", "2025-06-27", ...]
        """
        data = self._get("/markets/options/expirations", {
            "symbol":         symbol,
            "includeAllRoots": "true",
            "strikes":         "false",
        })
        if not data:
            return []
        try:
            raw = data.get("expirations", {}).get("date", [])
            if isinstance(raw, str):
                raw = [raw]
            return [str(d) for d in raw if d]
        except Exception as e:
            logger.error(f"Expirations parse error for {symbol}: {e}")
            return []

    def get_nearest_expiry(
        self, symbol: str, min_dte: int = 7, target_dte: int = 12
    ) -> Optional[str]:
        """
        Returns the expiry date closest to target_dte that is also ≥ min_dte.
        """
        expirations = self.get_expirations(symbol)
        today = datetime.today().date()
        candidates = []
        for exp_str in expirations:
            try:
                exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp - today).days
                if dte >= min_dte:
                    candidates.append((abs(dte - target_dte), exp_str))
            except ValueError:
                continue
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    # ── Options chain ──────────────────────────────────────────────────────────

    def get_options_chain(
        self,
        symbol:     str,
        expiration: str,           # "YYYY-MM-DD"
        greeks:     bool = True,
    ) -> pd.DataFrame:
        """
        Returns full options chain for one expiration as a DataFrame.

        Columns (calls and puts separate rows):
            option_type, strike, bid, ask, mid, last, volume, open_interest,
            iv, delta, gamma, theta, vega, dte
        """
        data = self._get("/markets/options/chains", {
            "symbol":     symbol,
            "expiration": expiration,
            "greeks":     "true" if greeks else "false",
        })
        if not data:
            return pd.DataFrame()

        try:
            options_raw = data.get("options", {}).get("option", [])
            if not options_raw:
                return pd.DataFrame()
            if isinstance(options_raw, dict):
                options_raw = [options_raw]

            rows = []
            today = datetime.today().date()
            exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            dte = (exp_date - today).days

            for opt in options_raw:
                if not isinstance(opt, dict):
                    continue

                greeks_data = opt.get("greeks") or {}
                bid = float(opt.get("bid") or 0)
                ask = float(opt.get("ask") or 0)
                mid = round((bid + ask) / 2, 4)

                row = {
                    "option_type":   opt.get("option_type", "").lower(),   # call | put
                    "strike":        float(opt.get("strike") or 0),
                    "bid":           bid,
                    "ask":           ask,
                    "mid":           mid,
                    "last":          float(opt.get("last") or 0),
                    "volume":        int(opt.get("volume") or 0),
                    "open_interest": int(opt.get("open_interest") or 0),
                    "iv":            float(greeks_data.get("smv_vol") or
                                          opt.get("greeks", {}).get("smv_vol") or 0),
                    "delta":         float(greeks_data.get("delta") or 0),
                    "gamma":         float(greeks_data.get("gamma") or 0),
                    "theta":         float(greeks_data.get("theta") or 0),
                    "vega":          float(greeks_data.get("vega") or 0),
                    "dte":           dte,
                    "expiration":    expiration,
                    "symbol":        opt.get("underlying"),
                    "description":   opt.get("description", ""),
                }
                rows.append(row)

            df = pd.DataFrame(rows)

            # Split into calls and puts (they're mixed in the response)
            df = df[df["strike"] > 0]

            # remove IV artefacts (IV < 0.01 or > 5.0 = 500%)
            df = df[(df["iv"] > 0.01) | (df["iv"] == 0)]  # keep zeros for now
            df = df[df["iv"] < 5.0]

            logger.debug(f"{symbol} {expiration}: {len(df)} contracts loaded")
            return df

        except Exception as e:
            logger.error(f"Chain parse error for {symbol} {expiration}: {e}")
            return pd.DataFrame()

    def get_calls(self, symbol: str, expiration: str) -> pd.DataFrame:
        """Get only call contracts."""
        chain = self.get_options_chain(symbol, expiration)
        if chain.empty:
            return chain
        return chain[chain["option_type"] == "call"].reset_index(drop=True)

    def get_puts(self, symbol: str, expiration: str) -> pd.DataFrame:
        """Get only put contracts."""
        chain = self.get_options_chain(symbol, expiration)
        if chain.empty:
            return chain
        return chain[chain["option_type"] == "put"].reset_index(drop=True)

    # ── Historical IV (via options chain snapshots) ────────────────────────────

    def get_atm_iv(self, symbol: str) -> Optional[float]:
        """
        Returns the current ATM implied volatility for symbol.
        Uses the nearest weekly expiry ≥ 7 DTE.
        """
        quote = self.get_quote(symbol)
        if not quote:
            return None
        current_price = quote["last"]
        if current_price <= 0:
            return None

        expiry = self.get_nearest_expiry(symbol, min_dte=7, target_dte=14)
        if not expiry:
            return None

        calls = self.get_calls(symbol, expiry)
        if calls.empty:
            return None

        # find ATM strike
        calls["dist"] = (calls["strike"] - current_price).abs()
        atm = calls.nsmallest(3, "dist")
        iv_vals = atm["iv"][(atm["iv"] > 0.01) & (atm["iv"] < 5.0)]
        if iv_vals.empty:
            return None
        return float(iv_vals.median())

    # ── Connectivity check ─────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Returns True if Tradier API is reachable and token is valid."""
        data = self._get("/user/profile")
        return data is not None

    def __repr__(self) -> str:
        mode = "sandbox" if self.sandbox else "production"
        auth = "authenticated" if self.token else "NO TOKEN"
        return f"TradierClient({mode}, {auth})"


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in yfinance replacement helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_options_chain_yf_compat(
    symbol: str,
    client: Optional[TradierClient] = None,
) -> tuple:
    """
    Returns (calls_df, puts_df) in a format compatible with what our
    existing iv_rank.py and strike_selector.py expect.

    Tries Tradier first; falls back to yfinance if unavailable.
    """
    # Try Tradier
    if client and client.token:
        expiry = client.get_nearest_expiry(symbol, min_dte=7, target_dte=14)
        if expiry:
            chain = client.get_options_chain(symbol, expiry)
            if not chain.empty:
                calls = chain[chain["option_type"] == "call"].rename(columns={
                    "iv":           "impliedVolatility",
                    "open_interest":"openInterest",
                    "mid":          "lastPrice",
                })
                puts = chain[chain["option_type"] == "put"].rename(columns={
                    "iv":           "impliedVolatility",
                    "open_interest":"openInterest",
                    "mid":          "lastPrice",
                })
                return calls, puts

    # Fallback: yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        exps   = ticker.options
        if not exps:
            return pd.DataFrame(), pd.DataFrame()
        # pick nearest expiry ≥ 7 DTE
        today = datetime.today().date()
        valid = [e for e in exps if (datetime.strptime(e, "%Y-%m-%d").date() - today).days >= 7]
        if not valid:
            return pd.DataFrame(), pd.DataFrame()
        chain = ticker.option_chain(valid[0])
        return chain.calls, chain.puts
    except Exception as e:
        logger.error(f"yfinance fallback failed for {symbol}: {e}")
        return pd.DataFrame(), pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    token = os.environ.get("TRADIER_TOKEN", "")
    if not token:
        print("\n⚠️  No TRADIER_TOKEN set.")
        print("   1. Sign up free at: https://developer.tradier.com/user/sign_up")
        print("   2. Copy your sandbox token from the dashboard")
        print("   3. Run: export TRADIER_TOKEN='your_token_here'")
        print("   4. Re-run this test\n")
        sys.exit(0)

    client = TradierClient(token=token, sandbox=True)
    print(f"\n{client}")
    print(f"Ping: {'✅ connected' if client.ping() else '❌ failed'}\n")

    for sym in ["SPY", "AAPL", "NVDA"]:
        quote = client.get_quote(sym)
        if quote:
            print(f"{sym:6s}  last=${quote['last']:.2f}  "
                  f"chg={quote['change_pct']:+.2f}%")

            exps = client.get_expirations(sym)
            nearest = client.get_nearest_expiry(sym, min_dte=7, target_dte=12)
            print(f"         expirations: {len(exps)} available  "
                  f"→ nearest ~12 DTE: {nearest}")

            if nearest:
                calls = client.get_calls(sym, nearest)
                if not calls.empty:
                    atm_row = calls.iloc[(calls["strike"] - quote["last"]).abs().argsort()[:1]]
                    r = atm_row.iloc[0]
                    print(f"         ATM call: K={r['strike']:.0f}  "
                          f"IV={r['iv']*100:.1f}%  delta={r['delta']:.2f}  "
                          f"bid={r['bid']:.2f}  ask={r['ask']:.2f}  "
                          f"OI={r['open_interest']:,}")
        print()
