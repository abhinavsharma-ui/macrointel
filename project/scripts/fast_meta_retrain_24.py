from __future__ import annotations
import json, logging, os, sys, warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore", category=RuntimeWarning)
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv(PROJECT / ".env")
from models import institutional_retraining as ir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s fast_meta %(message)s")
logger = logging.getLogger("fast_meta")

def safe_float(v, default=0.0):
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def market_bucket(symbol: str) -> str:
    s = str(symbol or "").upper()
    if s.endswith(".NS"): return "nse"
    if "USDT" in s or s.endswith("-USD") or s.endswith("USD"): return "crypto"
    return "us"

def matches_scope(symbol: str, scope: str) -> bool:
    scope = str(scope or "us").lower()
    if scope in {"all", "*"}: return True
    return market_bucket(symbol) == scope

def coerce_index(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    idx = None
    if isinstance(frame.index, pd.DatetimeIndex):
        idx = pd.to_datetime(frame.index, errors="coerce", utc=True)
    else:
        for col in ("timestamp", "date", "datetime", "time", "as_of"):
            if col in frame.columns:
                cand = pd.to_datetime(frame[col], errors="coerce", utc=True)
                if float(np.mean(~pd.isna(cand))) > 0.80:
                    idx = cand
                    break
    if idx is not None:
        try:
            idx = idx.dt.tz_convert(None) if hasattr(idx, "dt") else idx.tz_convert(None)
        except Exception:
            try:
                idx = idx.dt.tz_localize(None) if hasattr(idx, "dt") else idx.tz_localize(None)
            except Exception:
                pass
        frame.index = idx
        frame = frame[~pd.isna(frame.index)]
    return frame.sort_index()
def adaptive_horizon_days(frame: pd.DataFrame, idx: int, base: int):
    row = frame.iloc[idx]
    mult = 1.0
    vol_ratio = safe_float(row.get("vol_regime_ratio", np.nan), np.nan)
    if (not np.isnan(vol_ratio) and vol_ratio > 1.5) or safe_float(row.get("vol_regime_stressed", 0.0)) >= 1.0:
        mult *= 0.60
    event_signal = max(
        abs(safe_float(row.get("official_event_signal", 0.0))),
        abs(safe_float(row.get("filing_event_signal", 0.0))),
        abs(safe_float(row.get("earnings_tone_signal", 0.0))),
    )
    if safe_float(row.get("filing_event_hit", 0.0)) > 0 or safe_float(row.get("new_risk_factors", 0.0)) > 0 or event_signal >= 0.60:
        mult = max(mult, 3.0)
    return max(1, int(round(base * mult))), round(float(mult), 3)

def barrier_stats(direction: str, entry: float, future: pd.DataFrame, stop_pct: float, take_pct: float):
    if future is None or future.empty or entry <= 0:
        return {"edge_pct": 0.0, "drawdown_pct": 0.0, "hit": 0, "barrier": "none"}
    highs = pd.to_numeric(future.get("high", future.get("close")), errors="coerce").ffill()
    lows = pd.to_numeric(future.get("low", future.get("close")), errors="coerce").ffill()
    closes = pd.to_numeric(future.get("close"), errors="coerce").ffill()
    if closes.empty:
        return {"edge_pct": 0.0, "drawdown_pct": 0.0, "hit": 0, "barrier": "none"}
    if direction == "sell":
        favorable = ((entry / lows.replace(0, np.nan)) - 1.0) * 100.0
        adverse = ((entry / highs.replace(0, np.nan)) - 1.0) * 100.0
        final_edge = ((entry / max(float(closes.iloc[-1]), 1e-6)) - 1.0) * 100.0
    else:
        favorable = ((highs / entry) - 1.0) * 100.0
        adverse = ((lows / entry) - 1.0) * 100.0
        final_edge = ((float(closes.iloc[-1]) / entry) - 1.0) * 100.0
    favorable = favorable.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    adverse = adverse.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    drawdown = float(abs(min(float(adverse.min()), 0.0)))
    exit_edge = float(final_edge)
    barrier = "time"
    for fav, adv in zip(favorable.to_numpy(dtype=float), adverse.to_numpy(dtype=float)):
        if adv <= -abs(stop_pct):
            exit_edge = -abs(stop_pct); barrier = "stop"; break
        if fav >= abs(take_pct):
            exit_edge = abs(take_pct); barrier = "take"; break
    return {"edge_pct": round(float(exit_edge), 4), "drawdown_pct": round(float(drawdown), 4), "hit": int(exit_edge > 0), "barrier": barrier}

def risk_params(row):
    atr_pct = safe_float(row.get("atr_pct", 0.02), 0.02)
    stop = float(np.clip(atr_pct * 100 * 1.3, 1.2, 6.0))
    take = float(np.clip(stop * 2.2, stop + 0.8, 12.0))
    return {"stop_loss_pct": round(stop, 3), "take_profit_pct": round(take, 3), "risk_reward_ratio": round(take / max(stop, 0.1), 3)}
def build_symbol_records(args):
    symbol, df, scope, min_frame, warmup_rows, horizon_days = args
    try:
        if not matches_scope(symbol, scope):
            return {"symbol": symbol, "records": [], "skipped_reason": "market_scope"}
        ordered = coerce_index(df)
        if ordered.empty:
            return {"symbol": symbol, "records": [], "skipped_reason": "empty_frame"}
        if len(ordered) < min_frame:
            return {"symbol": symbol, "records": [], "skipped_reason": "short_frame"}

        from core.signal_engine_v2 import MultiFactorScorer
        scorer = MultiFactorScorer()
        records: List[Dict] = []
        label_min_edge = float(os.getenv("META_MODEL_LABEL_MIN_EDGE_PCT", "0.18") or 0.18)
        label_min_ratio = float(os.getenv("META_MODEL_LABEL_MIN_EDGE_RATIO", "0.08") or 0.08)

        for idx in range(warmup_rows, len(ordered) - 2):
            row = ordered.iloc[idx]
            score = scorer.score(row)
            direction = str(score.get("direction", "neutral") or "neutral").lower()

            if direction == "neutral":
                synth = 0.45 * safe_float(row.get("alpha_signal", 0.0)) + 0.35 * safe_float(row.get("event_alpha_signal", 0.0)) + 0.20 * safe_float(row.get("momentum_composite", 0.0))
                if synth > 0.10: direction = "buy"
                elif synth < -0.10: direction = "sell"
                else: continue
                score["confidence"] = max(safe_float(score.get("confidence", 0.0)), min(0.58, abs(synth) * 1.75))
                score["conviction_score"] = max(safe_float(score.get("conviction_score", 0.0)), round(min(6.6, abs(synth) * 12.0), 1))

            days, mult = adaptive_horizon_days(ordered, idx, horizon_days)
            future = ordered.iloc[idx + 1 : idx + 1 + days]
            entry = safe_float(row.get("close", 0.0))
            if future.empty or entry <= 0:
                continue

            risk = risk_params(row)
            path = barrier_stats(direction, entry, future, risk["stop_loss_pct"], risk["take_profit_pct"])
            roundtrip_cost_pct = float(os.getenv("META_MODEL_ROUNDTRIP_COST_PCT", "0.22") or 0.22)
            label_max_drawdown = float(os.getenv("META_MODEL_LABEL_MAX_DRAWDOWN_PCT", "3.50") or 3.50)
            net_edge_pct = path["edge_pct"] - roundtrip_cost_pct
            edge_ratio = net_edge_pct / max(path["drawdown_pct"], 0.35)
            take_label = int(
                net_edge_pct > label_min_edge
                and edge_ratio > label_min_ratio
                and path["drawdown_pct"] <= label_max_drawdown
                and path["hit"] == 1
            )

            feature_signal = {
                "signal": direction,
                "confidence": safe_float(score.get("confidence", 0.0)),
                "conviction_score": safe_float(score.get("conviction_score", 0.0)),
                "regime_multiplier": safe_float(score.get("regime_multiplier", 1.0), 1.0),
                "regime": score.get("regime", "unknown"),
                "factor_scores": score.get("factor_scores", {}) or {},
                "risk_parameters": risk,
                "model_agreement": 0.0,
                "xgb_alignment": "",
            }
            vals = ir.MetaFeatureBuilder.build(symbol, feature_signal, row)
            vals.update({
                "symbol": symbol,
                "timestamp": pd.Timestamp(ordered.index[idx]),
                "market_bucket": market_bucket(symbol),
                "direction_label": "long" if direction == "buy" else "short",
                "adaptive_horizon_days": days,
                "adaptive_horizon_multiplier": mult,
                "path_take_label": take_label,
                "take_label": take_label,
                "edge_pct": path["edge_pct"],
                "net_edge_pct": round(float(net_edge_pct), 4),
                "drawdown_pct": path["drawdown_pct"],
                "edge_ratio": round(float(edge_ratio), 4),
                "hit": path["hit"],
                "barrier": path["barrier"],
            })
            records.append(vals)
        return {"symbol": symbol, "records": records, "skipped_reason": "" if records else "no_records"}
    except Exception as exc:
        return {"symbol": symbol, "records": [], "skipped_reason": "error_" + type(exc).__name__, "error": str(exc)[:300]}
def fast_build_training_dataset(self, feature_matrices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    scope = str(getattr(self, "market_scope", os.getenv("INSTITUTIONAL_RETRAIN_MARKET", "us")) or "us").lower()
    min_frame = max(int(os.getenv("META_MODEL_MIN_FRAME_ROWS", "72") or 72), int(getattr(self, "horizon_days", 5)) + 22)
    warmup = max(28, int(os.getenv("META_MODEL_WARMUP_ROWS", "42") or 42))
    horizon = int(getattr(self, "horizon_days", int(os.getenv("META_MODEL_HORIZON_DAYS", "5") or 5)))
    workers = max(1, min(int(os.getenv("META_MODEL_BUILD_WORKERS", "24") or 24), os.cpu_count() or 1))
    items = [(s, f, scope, min_frame, warmup, horizon) for s, f in feature_matrices.items() if matches_scope(s, scope)]
    logger.info("Meta dataset build: using %s workers across %s symbol frames", workers, len(items))

    records: List[Dict] = []
    skips = Counter()
    log_every = max(25, int(os.getenv("RETRAIN_META_LOG_EVERY", "50") or 50))

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(build_symbol_records, item) for item in items]
            for i, fut in enumerate(as_completed(futs), start=1):
                payload = fut.result()
                records.extend(payload.get("records", []))
                reason = payload.get("skipped_reason")
                if reason: skips[str(reason)] += 1
                if i % log_every == 0 or i == len(futs):
                    logger.info("Meta dataset build: %s / %s symbol jobs complete (%s training rows so far)", i, len(futs), len(records))
    else:
        for i, item in enumerate(items, start=1):
            payload = build_symbol_records(item)
            records.extend(payload.get("records", []))
            reason = payload.get("skipped_reason")
            if reason: skips[str(reason)] += 1
            if i % log_every == 0 or i == len(items):
                logger.info("Meta dataset build: %s / %s symbols (%s training rows so far)", i, len(items), len(records))

    logger.info("Meta dataset build skip summary: %s", dict(skips))
    if not records:
        return pd.DataFrame()

    dataset = pd.DataFrame(records)
    ts = pd.to_datetime(dataset["timestamp"], errors="coerce", utc=True)
    dataset["timestamp"] = ts.dt.tz_convert(None)
    dataset = dataset.dropna(subset=["timestamp"])
    dataset = dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    if hasattr(self, "_apply_cross_sectional_context"):
        dataset = self._apply_cross_sectional_context(dataset)

    if os.getenv("META_PRICE_ONLY_RANKING", "0").strip().lower() in {"1", "true", "yes", "on"}:
        group_keys = ["timestamp", "market_bucket", "direction_label"]
        if "market_bucket" not in dataset.columns:
            dataset["market_bucket"] = "us"
        if "direction_label" not in dataset.columns:
            dataset["direction_label"] = "long"
        group_keys = ["timestamp", "market_bucket", "direction_label"]
        if "cross_sectional_candidate_count" not in dataset.columns:
            dataset["cross_sectional_candidate_count"] = dataset.groupby(group_keys)["symbol"].transform("count").astype(float)
        for col in [
            "cross_sectional_selection_rank_pct",
            "cross_sectional_momentum_rank_pct",
            "cross_sectional_conviction_rank_pct",
            "cross_sectional_volatility_rank_pct",
        ]:
            if col not in dataset.columns:
                dataset[col] = 0.5
        score = (
            0.42 * dataset["cross_sectional_selection_rank_pct"].astype(float)
            + 0.30 * dataset["cross_sectional_momentum_rank_pct"].astype(float)
            + 0.18 * dataset["cross_sectional_conviction_rank_pct"].astype(float)
            + 0.10 * dataset["cross_sectional_volatility_rank_pct"].astype(float)
        )
        dataset["cross_sectional_score_rank_pct"] = dataset.assign(__price_only_score=score).groupby(group_keys)["__price_only_score"].rank(pct=True, method="average")
        top_pct = float(os.getenv("META_MODEL_CROSS_SECTIONAL_LABEL_PCT", "0.04") or 0.04)
        top_pct = float(np.clip(top_pct, 0.01, 0.30))
        min_candidates = max(3, int(os.getenv("META_MODEL_CROSS_SECTIONAL_MIN_CANDIDATES", "15") or 15))
        selected = dataset["cross_sectional_score_rank_pct"] >= (1.0 - top_pct)
        small_group = dataset["cross_sectional_candidate_count"].astype(float) < float(min_candidates)
        dataset["take_label"] = (dataset["path_take_label"].astype(int).eq(1) & (selected | small_group)).astype(int)
        logger.info("Meta price-only ranking active: top_pct=%.3f min_candidates=%s positives=%s", top_pct, min_candidates, int(dataset["take_label"].sum()))

    logger.info("Meta dataset ready: %s rows", len(dataset))
    return dataset
def load_features() -> Dict[str, pd.DataFrame]:
    root = Path(os.getenv("XGB_RETRAIN_FEATURE_DIR", "data/features_10yr"))
    if not root.is_absolute():
        root = PROJECT / root
    scope = os.getenv("INSTITUTIONAL_RETRAIN_MARKET", "us").lower()
    files = sorted(root.glob("*.parquet"))
    logger.info("Loading %s parquet files from %s scope=%s", len(files), root, scope)

    out: Dict[str, pd.DataFrame] = {}
    rows = 0
    for i, path in enumerate(files, start=1):
        sym = path.stem
        if not matches_scope(sym, scope):
            continue
        try:
            df = pd.read_parquet(path)
            df = coerce_index(df)
            if not df.empty:
                out[sym] = df
                rows += len(df)
        except Exception as exc:
            logger.warning("feature load skipped %s: %s", path.name, exc)
        if i % 250 == 0:
            logger.info("Feature load progress: %s/%s files, %s symbols, %s rows", i, len(files), len(out), rows)
    logger.info("Feature load complete: %s symbols, %s rows", len(out), rows)
    if not out:
        raise RuntimeError("No feature matrices loaded")
    return out

def make_trainer():
    kwargs = {
        "horizon_days": int(os.getenv("META_MODEL_HORIZON_DAYS", "5") or 5),
        "take_threshold": float(os.getenv("META_MODEL_TAKE_THRESHOLD", "0.60") or 0.60),
        "walk_forward_folds": int(os.getenv("META_MODEL_WALKFORWARD_FOLDS", "3") or 3),
        "min_train_days": int(os.getenv("META_MODEL_MIN_TRAIN_DAYS", "900") or 900),
    }
    scope = os.getenv("INSTITUTIONAL_RETRAIN_MARKET", "us")
    try:
        trainer = ir.MetaModelTrainer(**kwargs, market_scope=scope)
    except TypeError:
        try:
            trainer = ir.MetaModelTrainer(**kwargs)
        except TypeError:
            trainer = ir.MetaModelTrainer()
            for k, v in kwargs.items():
                setattr(trainer, k, v)
        setattr(trainer, "market_scope", scope)
    return trainer

def redirect_artifacts():
    ckpt = PROJECT / "models" / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    cand = ckpt / ("fast_meta_candidate_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    cand.mkdir(parents=True, exist_ok=True)

    for name in [
        "META_CLASSIFIER_PATH", "META_EDGE_PATH", "META_DRAWDOWN_PATH", "META_FEATURES_PATH",
        "META_REPORT_PATH", "META_REPORT_CANDIDATE_PATH",
        "META_DIRECTIONAL_PATH", "META_DIRECTIONAL_CANDIDATE_PATH",
    ]:
        if hasattr(ir, name):
            old = Path(getattr(ir, name))
            setattr(ir, name, cand / old.name)

    (ckpt / "fast_meta_latest.txt").write_text(str(cand), encoding="utf-8")
    logger.info("Candidate artifact dir: %s", cand)
    return cand
def main():
    logger.info("FAST META RETRAIN START pid=%s cpu=%s", os.getpid(), os.cpu_count())
    logger.info("workers=%s folds=%s min_train_days=%s", os.getenv("META_MODEL_BUILD_WORKERS"), os.getenv("META_MODEL_WALKFORWARD_FOLDS"), os.getenv("META_MODEL_MIN_TRAIN_DAYS"))

    ir.MetaModelTrainer.build_training_dataset = fast_build_training_dataset

    cand = redirect_artifacts()
    matrices = load_features()
    trainer = make_trainer()
    report = trainer.train(matrices)

    report_path = cand / "meta_walkforward_report.json"
    if not report_path.exists():
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    summary = report.get("summary") or (report.get("walk_forward") or {}).get("summary") or {}
    logger.info("FAST META RETRAIN DONE status=%s summary=%s", report.get("status"), summary)
    print("FAST_META_DONE")
    print("candidate_dir", cand)
    print("report", report_path)
    print(json.dumps({"status": report.get("status"), "summary": summary}, indent=2, default=str))

if __name__ == "__main__":
    main()
