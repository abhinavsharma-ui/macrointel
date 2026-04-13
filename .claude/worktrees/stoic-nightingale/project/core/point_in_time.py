"""
Point-in-Time (PIT) Data Engine
================================
This is the single biggest difference between a student project and
a professional quantitative system.

THE PROBLEM:
  If you train a model in 2024 using historical data, you'll accidentally
  include information that wasn't available at the time:
  - Survivorship bias: companies that went bankrupt are missing
  - Look-ahead bias: adjusted prices use future split data
  - Revision bias: GDP numbers get revised months later

THE SOLUTION:
  Store EVERY data point with two timestamps:
    1. as_of_date   — when the event actually happened in the world
    2. knowledge_date — when WE first knew about it

  When backtesting on 2023-03-15, ONLY show data where knowledge_date <= 2023-03-15

Academic reference: "Quantitative Equity Portfolio Management" (Qian, Hua, Sorensen)
Industry standard: FactSet, Bloomberg, Compustat all use PIT databases.
"""

import sqlite3
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from contextlib import contextmanager

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "pit_database.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────
SCHEMA = """
-- Core PIT price table
CREATE TABLE IF NOT EXISTS prices_pit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    as_of_date      DATE    NOT NULL,   -- The market date
    knowledge_date  DATE    NOT NULL,   -- When we ingested this
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL,
    adj_close       REAL,
    volume          INTEGER,
    source          TEXT    DEFAULT 'yfinance',
    is_active       INTEGER DEFAULT 1,  -- 0 if company later delisted
    UNIQUE(symbol, as_of_date, source)
);

-- Fundamentals (earnings, P/E) with announcement delay
CREATE TABLE IF NOT EXISTS fundamentals_pit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    fiscal_date     DATE    NOT NULL,   -- Quarter end date
    announcement_date DATE NOT NULL,   -- When company announced
    knowledge_date  DATE    NOT NULL,   -- When WE knew (announcement_date + 1 day)
    eps_reported    REAL,
    eps_estimated   REAL,
    surprise_pct    REAL,
    revenue         REAL,
    pe_ratio        REAL,
    source          TEXT    DEFAULT 'alpha_vantage'
);

-- Macro indicators (FRED releases with known publication lag)
CREATE TABLE IF NOT EXISTS macro_pit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id       TEXT    NOT NULL,
    observation_date DATE   NOT NULL,  -- Date of the economic reading
    release_date    DATE    NOT NULL,  -- When FRED published it (may be weeks later)
    knowledge_date  DATE    NOT NULL,  -- = release_date
    value           REAL,
    vintage         TEXT,              -- Revision vintage (e.g. "advance", "revised")
    UNIQUE(series_id, observation_date, vintage)
);

-- Sentiment with ingestion timestamp
CREATE TABLE IF NOT EXISTS sentiment_pit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    as_of_date      DATE    NOT NULL,
    knowledge_date  DATE    NOT NULL,
    sentiment_score REAL,
    article_count   INTEGER,
    source          TEXT
);

-- Universe membership over time (handles survivorship bias)
CREATE TABLE IF NOT EXISTS universe_membership (
    symbol          TEXT    NOT NULL,
    index_name      TEXT    NOT NULL,  -- e.g. "NIFTY50", "SP500"
    added_date      DATE    NOT NULL,
    removed_date    DATE,              -- NULL if still in index
    removal_reason  TEXT,              -- "BANKRUPTCY", "MERGER", "DELISTED", "REBALANCE"
    PRIMARY KEY (symbol, index_name, added_date)
);

-- Audit log of all data ingestions
CREATE TABLE IF NOT EXISTS ingestion_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ingested_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source          TEXT,
    dataset         TEXT,
    rows_added      INTEGER,
    earliest_date   DATE,
    latest_date     DATE,
    notes           TEXT
);

-- Indexes for fast PIT queries
CREATE INDEX IF NOT EXISTS idx_prices_symbol_asof ON prices_pit(symbol, as_of_date);
CREATE INDEX IF NOT EXISTS idx_prices_knowledge ON prices_pit(knowledge_date);
CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol ON fundamentals_pit(symbol, knowledge_date);
CREATE INDEX IF NOT EXISTS idx_macro_series ON macro_pit(series_id, knowledge_date);
CREATE INDEX IF NOT EXISTS idx_universe_dates ON universe_membership(index_name, added_date, removed_date);
"""


# ─────────────────────────────────────────────────────────────
# Database Manager
# ─────────────────────────────────────────────────────────────
class PITDatabase:
    """
    Point-in-Time database manager.
    
    All queries accept an `as_of` parameter (the backtest date).
    Results are filtered to only include data known on that date.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        logger.debug(f"PIT database initialized at {self.db_path}")

    # ── Write operations ──────────────────────────────────────

    def insert_prices(self, df: pd.DataFrame, source: str = "yfinance") -> int:
        """
        Insert price data with PIT timestamps.
        knowledge_date = today (when we're ingesting this data).
        """
        if df.empty:
            return 0

        knowledge_date = date.today().isoformat()
        records = []

        for symbol, group in df.groupby("symbol") if "symbol" in df.columns else [("UNKNOWN", df)]:
            for idx, row in group.iterrows():
                as_of = idx.date() if hasattr(idx, 'date') else idx
                records.append((
                    str(symbol),
                    str(as_of),
                    knowledge_date,
                    float(row.get("open", 0) or 0),
                    float(row.get("high", 0) or 0),
                    float(row.get("low", 0) or 0),
                    float(row.get("close", 0) or 0),
                    float(row.get("adj_close", row.get("close", 0)) or 0),
                    int(row.get("volume", 0) or 0),
                    source,
                    1,
                ))

        with self._conn() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO prices_pit
                (symbol, as_of_date, knowledge_date, open, high, low, close,
                 adj_close, volume, source, is_active)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, records)

        self._log_ingestion("prices_pit", source, len(records))
        return len(records)

    def insert_fundamentals(self, df: pd.DataFrame) -> int:
        """
        Insert earnings/fundamental data with PROPER announcement delay.
        Companies typically announce earnings 3-6 weeks after quarter end.
        We never use data before the announcement date.
        """
        if df.empty:
            return 0

        records = []
        for _, row in df.iterrows():
            announcement = row.get("reported_date") or row.get("announcement_date")
            if pd.isna(announcement):
                # Conservative default: assume 45-day lag from fiscal date
                fiscal = pd.to_datetime(row.get("fiscal_date", date.today()))
                announcement = fiscal + timedelta(days=45)

            announcement = pd.to_datetime(announcement).date()
            knowledge_date = announcement + timedelta(days=1)  # T+1 to be safe

            records.append((
                str(row.get("symbol", "")),
                str(pd.to_datetime(row.get("fiscal_date", date.today())).date()),
                str(announcement),
                str(knowledge_date),
                row.get("reported_eps"),
                row.get("estimated_eps"),
                row.get("surprise_pct"),
                row.get("revenue"),
                row.get("pe_ratio"),
                "alpha_vantage",
            ))

        with self._conn() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO fundamentals_pit
                (symbol, fiscal_date, announcement_date, knowledge_date,
                 eps_reported, eps_estimated, surprise_pct, revenue, pe_ratio, source)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, records)

        self._log_ingestion("fundamentals_pit", "alpha_vantage", len(records))
        return len(records)

    def insert_macro(self, series_id: str, series: pd.Series,
                     release_lag_days: int = 30, vintage: str = "advance") -> int:
        """
        Insert macro time series with FRED publication lag.
        
        FRED Publication Lags (approximate):
        - GDP (advance):        30 days after quarter end
        - CPI:                   2 weeks after reference month
        - NFP:                   1 week after reference month
        - Fed Funds Rate:        Same day (daily series)
        - Industrial Production: ~17 days
        """
        if series.empty:
            return 0

        records = []
        for obs_date, value in series.items():
            if pd.isna(value):
                continue
            obs_date = pd.to_datetime(obs_date).date()
            release_date = obs_date + timedelta(days=release_lag_days)
            records.append((
                series_id,
                str(obs_date),
                str(release_date),
                str(release_date),
                float(value),
                vintage,
            ))

        with self._conn() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO macro_pit
                (series_id, observation_date, release_date, knowledge_date, value, vintage)
                VALUES (?,?,?,?,?,?)
            """, records)

        return len(records)

    def mark_delisted(self, symbol: str, delisted_date: date, reason: str = "DELISTED"):
        """Mark a symbol as no longer active. Critical for survivorship bias elimination."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE prices_pit SET is_active = 0
                WHERE symbol = ? AND as_of_date >= ?
            """, (symbol, str(delisted_date)))

            conn.execute("""
                UPDATE universe_membership
                SET removed_date = ?, removal_reason = ?
                WHERE symbol = ? AND removed_date IS NULL
            """, (str(delisted_date), reason, symbol))

    # ── PIT Read operations ───────────────────────────────────

    def get_prices(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        as_of: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch prices as they would have been known on `as_of` date.
        
        as_of=None means "use everything we have" (for live trading).
        as_of="2022-01-01" means "simulate what we knew on Jan 1, 2022" (backtesting).
        """
        as_of = as_of or date.today().isoformat()
        placeholders = ",".join("?" * len(symbols))

        with self._conn() as conn:
            rows = conn.execute(f"""
                SELECT symbol, as_of_date, open, high, low, close, adj_close, volume
                FROM prices_pit
                WHERE symbol IN ({placeholders})
                  AND as_of_date BETWEEN ? AND ?
                  AND knowledge_date <= ?
                ORDER BY symbol, as_of_date
            """, (*symbols, start_date, end_date, as_of)).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low",
                                          "close", "adj_close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def get_universe_on_date(self, index_name: str, as_of: str) -> List[str]:
        """
        Returns the EXACT index constituents as of a given date.
        Eliminates survivorship bias — companies that later went bankrupt
        ARE included if they were in the index on that date.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT symbol FROM universe_membership
                WHERE index_name = ?
                  AND added_date <= ?
                  AND (removed_date IS NULL OR removed_date > ?)
            """, (index_name, as_of, as_of)).fetchall()

        return [r["symbol"] for r in rows]

    def get_fundamentals(
        self,
        symbol: str,
        as_of: str,
        lookback_quarters: int = 8,
    ) -> pd.DataFrame:
        """Fetch earnings data known as of a given date."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT fiscal_date, announcement_date, eps_reported, eps_estimated,
                       surprise_pct, revenue, pe_ratio
                FROM fundamentals_pit
                WHERE symbol = ? AND knowledge_date <= ?
                ORDER BY fiscal_date DESC
                LIMIT ?
            """, (symbol, as_of, lookback_quarters)).fetchall()

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=[r[0] for r in conn.execute(
            "SELECT * FROM fundamentals_pit LIMIT 0").description
        ] if False else ["fiscal_date", "announcement_date", "eps_reported",
                         "eps_estimated", "surprise_pct", "revenue", "pe_ratio"])

    def get_macro_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        as_of: Optional[str] = None,
    ) -> pd.Series:
        """Fetch macro data as known on as_of date."""
        as_of = as_of or date.today().isoformat()
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT observation_date, value
                FROM macro_pit
                WHERE series_id = ?
                  AND observation_date BETWEEN ? AND ?
                  AND knowledge_date <= ?
                ORDER BY observation_date
            """, (series_id, start_date, end_date, as_of)).fetchall()

        if not rows:
            return pd.Series(dtype=float, name=series_id)

        dates, values = zip(*rows)
        return pd.Series(values, index=pd.to_datetime(list(dates)), name=series_id)

    def _log_ingestion(self, dataset: str, source: str, rows: int):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO ingestion_log (source, dataset, rows_added)
                VALUES (?, ?, ?)
            """, (source, dataset, rows))


# ─────────────────────────────────────────────────────────────
# PIT Feature Builder — wraps FeaturePipeline with PIT awareness
# ─────────────────────────────────────────────────────────────
class PITFeatureBuilder:
    """
    Builds feature matrices that are guaranteed point-in-time safe.
    No future data leakage possible.
    
    Usage in backtesting:
        builder = PITFeatureBuilder(pit_db)
        for backtest_date in date_range:
            features = builder.get_features_as_of(symbol, backtest_date)
            signal = model.predict(features)
    """

    def __init__(self, pit_db: PITDatabase):
        self.db = pit_db

    def get_features_as_of(
        self,
        symbol: str,
        as_of_date: str,
        lookback_days: int = 300,
    ) -> Optional[pd.DataFrame]:
        """
        Returns the feature vector for `symbol` as it would have been computed
        on `as_of_date`, using only data known at that time.
        """
        from core.feature_engineering import TechnicalIndicatorEngine

        end = as_of_date
        start = (pd.Timestamp(as_of_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        price_df = self.db.get_prices([symbol], start, end, as_of=as_of_date)
        if price_df.empty or len(price_df) < 50:
            return None

        engine = TechnicalIndicatorEngine()
        features = engine.compute_all(price_df)
        return features.iloc[[-1]]  # Most recent row only

    def build_backtest_feature_panel(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Build a panel of PIT-safe features across a date range.
        This is what you feed to the backtester — guaranteed no look-ahead.
        """
        all_features = []
        current = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        while current <= end:
            if current.weekday() < 5:  # Trading days only
                as_of_str = current.strftime("%Y-%m-%d")
                feat = self.get_features_as_of(symbol, as_of_str)
                if feat is not None:
                    feat["backtest_date"] = as_of_str
                    all_features.append(feat)
            current += pd.Timedelta(days=1)

        if not all_features:
            return pd.DataFrame()

        panel = pd.concat(all_features).set_index("backtest_date")
        logger.info(f"PIT panel built for {symbol}: {len(panel)} dates, {len(panel.columns)} features")
        return panel
