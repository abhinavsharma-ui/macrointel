"""
Adaptive Ensemble Router — Online Regime-Aware Model Reweighter
================================================================
Tracks per-model directional accuracy within each market regime and
dynamically adjusts ensemble weights so the live signal loop always
uses the best-performing blend for current conditions.

Usage (live loop):
    router = AdaptiveEnsembleRouter.load("router_state.json")

    # At inference time
    weights = router.get_weights(regime)          # {"xgboost": 0.4, ...}
    ensemble_signal = weighted_combine(weights, model_preds)

    # After a trade closes
    router.record_outcome(trade_result)

    # Pre-warm from historical log
    router.backtest_replay("trade_log.csv")
"""

import json
import logging
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

MODEL_NAMES = ("xgboost", "lstm", "transformer")
REGIMES = ("calm", "normal", "stressed", "crisis")
COLD_START_THRESHOLD = 10


class _ModelRegimeTracker:
    """Exponential-decay accuracy tracker for one (model, regime) pair."""

    __slots__ = ("hits", "total", "ema_accuracy", "decay", "predictions")

    def __init__(self, half_life: float = 20.0):
        self.hits: float = 0.0          # decayed hit count
        self.total: float = 0.0         # decayed total count
        self.ema_accuracy: float = 0.5  # running EMA
        self.decay: float = math.log(2) / half_life
        self.predictions: int = 0       # raw count (for cold-start gate)

    def update(self, correct: bool) -> None:
        # Apply decay to existing counts first
        lam = math.exp(-self.decay)
        self.hits *= lam
        self.total *= lam

        self.total += 1.0
        if correct:
            self.hits += 1.0

        self.predictions += 1
        self.ema_accuracy = self.hits / max(self.total, 1e-12)

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "total": self.total,
            "ema_accuracy": self.ema_accuracy,
            "predictions": self.predictions,
        }

    @classmethod
    def from_dict(cls, d: dict, half_life: float) -> "_ModelRegimeTracker":
        t = cls(half_life)
        t.hits = d["hits"]
        t.total = d["total"]
        t.ema_accuracy = d["ema_accuracy"]
        t.predictions = d["predictions"]
        return t


class AdaptiveEnsembleRouter:
    """
    Online adaptive ensemble reweighter.

    Thread-safe: all mutations go through ``self._lock``.
    Persists state to a JSON file after every weight update.

    Parameters
    ----------
    half_life : float
        Exponential decay half-life in number of trades. Default 20.
    state_path : str | Path | None
        Where to persist JSON state. ``None`` disables persistence.
    min_weight : float
        Floor per-model weight (prevents a model from being zeroed out).
    max_weight : float
        Ceiling per-model weight (prevents single-model dominance).
    """

    def __init__(
        self,
        half_life: float = 20.0,
        state_path: Optional[Union[str, Path]] = None,
        min_weight: float = 0.05,
        max_weight: float = 0.80,
    ):
        self.half_life = half_life
        self.state_path = Path(state_path) if state_path else None
        self.min_weight = min_weight
        self.max_weight = max_weight

        self._lock = threading.Lock()

        # trackers[model][regime] -> _ModelRegimeTracker
        self._trackers: Dict[str, Dict[str, _ModelRegimeTracker]] = {
            m: {r: _ModelRegimeTracker(half_life) for r in REGIMES}
            for m in MODEL_NAMES
        }

        # Pending predictions: keyed by trade_id so we can match outcomes.
        # {trade_id: {"regime": str, "predictions": {model: direction}}}
        self._pending: Dict[str, dict] = {}

        # Cache of last computed weights per regime (avoids recompute on
        # every get_weights call when nothing changed).
        self._weight_cache: Dict[str, Dict[str, float]] = {}
        self._cache_dirty = True

        self._update_count = 0

    # ── public API ───────────────────────────────────────────────

    def register_predictions(
        self,
        trade_id: str,
        regime: str,
        model_predictions: Dict[str, str],
    ) -> None:
        """
        Record each model's individual directional prediction at signal time.

        Parameters
        ----------
        trade_id : str
            Unique identifier that will later appear in the trade result.
        regime : str
            Market regime at signal time (calm / normal / stressed / crisis).
        model_predictions : dict
            ``{"xgboost": "buy", "lstm": "sell", "transformer": "buy"}``
            Direction each model predicted independently.
        """
        regime = self._sanitize_regime(regime)
        with self._lock:
            self._pending[trade_id] = {
                "regime": regime,
                "predictions": {
                    m: d for m, d in model_predictions.items() if m in MODEL_NAMES
                },
                "registered_at": datetime.utcnow().isoformat(),
            }

    def record_outcome(self, trade_result: dict) -> Optional[Dict[str, float]]:
        """
        Call after a trade closes. Matches the outcome against each model's
        prediction that was registered at signal time and updates trackers.

        Parameters
        ----------
        trade_result : dict
            Must contain at minimum:
            - ``trade_id`` or ``order_id``: links back to register_predictions
            - ``actual_direction``: the realised direction ("buy"/"sell"/"hold")
              **OR** ``realized_pnl``: positive → buy was correct, negative → sell

        Returns
        -------
        Updated weights for the trade's regime, or None if trade_id not found.
        """
        trade_id = trade_result.get("trade_id") or trade_result.get("order_id", "")
        actual_dir = self._resolve_direction(trade_result)

        with self._lock:
            entry = self._pending.pop(trade_id, None)
            if entry is None:
                logger.debug(f"AdaptiveRouter: no pending predictions for {trade_id}")
                return None

            regime = entry["regime"]
            for model, predicted_dir in entry["predictions"].items():
                correct = predicted_dir == actual_dir
                self._trackers[model][regime].update(correct)

            self._cache_dirty = True
            self._update_count += 1

        # Persist outside the lock to avoid holding it during I/O
        self._persist()

        weights = self.get_weights(regime)
        logger.info(
            f"AdaptiveRouter update #{self._update_count} "
            f"regime={regime} actual={actual_dir} → "
            + " | ".join(f"{m}={w:.3f}" for m, w in weights.items())
        )
        return weights

    def get_weights(self, regime: str) -> Dict[str, float]:
        """
        Return normalised weight dict for the given regime.

        Falls back to equal weights (1/N) if any model has fewer than
        ``COLD_START_THRESHOLD`` predictions in this regime.
        """
        regime = self._sanitize_regime(regime)

        with self._lock:
            if not self._cache_dirty and regime in self._weight_cache:
                return dict(self._weight_cache[regime])

            # Cold-start check
            counts = {
                m: self._trackers[m][regime].predictions for m in MODEL_NAMES
            }
            if any(c < COLD_START_THRESHOLD for c in counts.values()):
                equal = {m: 1.0 / len(MODEL_NAMES) for m in MODEL_NAMES}
                self._weight_cache[regime] = equal
                return dict(equal)

            # Accuracy-proportional weights with exp-decay built in
            raw = {}
            for m in MODEL_NAMES:
                raw[m] = self._trackers[m][regime].ema_accuracy

            weights = self._normalize(raw)
            self._weight_cache[regime] = weights

            # Only clear dirty flag when ALL regimes are refreshed,
            # but caching per-regime is fine — stale entries just get
            # recomputed on next call.
            return dict(weights)

    def get_stats(self) -> Dict:
        """Return a read-only snapshot of all tracker states."""
        with self._lock:
            return {
                m: {
                    r: {
                        **self._trackers[m][r].to_dict(),
                        "weight": self.get_weights(r).get(m, 0),
                    }
                    for r in REGIMES
                }
                for m in MODEL_NAMES
            }

    def backtest_replay(self, trade_log_path: Union[str, Path]) -> Dict[str, Dict[str, float]]:
        """
        Replay a historical trade log to pre-warm weights before going live.

        The trade log must be a JSON file (list of dicts) or CSV with columns:
            trade_id, regime, actual_direction,
            xgboost_prediction, lstm_prediction, transformer_prediction

        Returns final weights per regime after replay.
        """
        path = Path(trade_log_path)
        if not path.exists():
            raise FileNotFoundError(f"Trade log not found: {path}")

        if path.suffix == ".json":
            with open(path, "r") as f:
                records = json.load(f)
        elif path.suffix == ".csv":
            import csv
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                records = list(reader)
        else:
            raise ValueError(f"Unsupported format: {path.suffix} (use .json or .csv)")

        replayed = 0
        for rec in records:
            trade_id = rec.get("trade_id", f"replay_{replayed}")
            regime = rec.get("regime", "normal")

            preds = {}
            for m in MODEL_NAMES:
                key = f"{m}_prediction"
                if key in rec and rec[key]:
                    preds[m] = rec[key]

            if not preds:
                continue

            self.register_predictions(trade_id, regime, preds)

            actual = rec.get("actual_direction")
            if actual is None:
                pnl = float(rec.get("realized_pnl", 0))
                actual = "buy" if pnl > 0 else ("sell" if pnl < 0 else "hold")

            self.record_outcome({
                "trade_id": trade_id,
                "actual_direction": actual,
            })
            replayed += 1

        logger.info(f"AdaptiveRouter: replayed {replayed} trades from {path}")
        result = {r: self.get_weights(r) for r in REGIMES}
        return result

    # ── persistence ──────────────────────────────────────────────

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Serialize full state to JSON."""
        target = Path(path) if path else self.state_path
        if target is None:
            return

        with self._lock:
            state = {
                "half_life": self.half_life,
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
                "update_count": self._update_count,
                "trackers": {
                    m: {r: self._trackers[m][r].to_dict() for r in REGIMES}
                    for m in MODEL_NAMES
                },
                "pending": dict(self._pending),
                "saved_at": datetime.utcnow().isoformat(),
            }

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        tmp.replace(target)  # atomic on most filesystems

    @classmethod
    def load(
        cls,
        state_path: Union[str, Path],
        half_life: float = 20.0,
        min_weight: float = 0.05,
        max_weight: float = 0.80,
    ) -> "AdaptiveEnsembleRouter":
        """
        Load from a persisted JSON file, or return a fresh instance
        if the file doesn't exist.
        """
        path = Path(state_path)
        router = cls(
            half_life=half_life,
            state_path=path,
            min_weight=min_weight,
            max_weight=max_weight,
        )

        if not path.exists():
            logger.info(f"AdaptiveRouter: no state at {path}, starting fresh")
            return router

        try:
            with open(path, "r") as f:
                state = json.load(f)

            hl = state.get("half_life", half_life)
            router.half_life = hl
            router._update_count = state.get("update_count", 0)

            for m in MODEL_NAMES:
                if m in state.get("trackers", {}):
                    for r in REGIMES:
                        if r in state["trackers"][m]:
                            router._trackers[m][r] = _ModelRegimeTracker.from_dict(
                                state["trackers"][m][r], hl
                            )

            router._pending = state.get("pending", {})
            router._cache_dirty = True
            logger.info(
                f"AdaptiveRouter: loaded state from {path} "
                f"({router._update_count} prior updates)"
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"AdaptiveRouter: corrupt state file, starting fresh: {exc}")

        return router

    # ── internals ────────────────────────────────────────────────

    def _normalize(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Clip to [min, max] and renormalize to sum=1."""
        clipped = {m: max(self.min_weight, min(self.max_weight, v)) for m, v in raw.items()}
        total = sum(clipped.values())
        if total < 1e-12:
            return {m: 1.0 / len(MODEL_NAMES) for m in MODEL_NAMES}
        return {m: v / total for m, v in clipped.items()}

    def _persist(self) -> None:
        """Best-effort persist after each update."""
        if self.state_path is None:
            return
        try:
            self.save()
        except OSError as exc:
            logger.warning(f"AdaptiveRouter: persistence failed: {exc}")

    @staticmethod
    def _sanitize_regime(regime: str) -> str:
        regime = regime.lower().strip()
        if regime not in REGIMES:
            logger.warning(f"Unknown regime '{regime}', falling back to 'normal'")
            return "normal"
        return regime

    @staticmethod
    def _resolve_direction(trade_result: dict) -> str:
        """Determine the actual winning direction from a trade result."""
        if "actual_direction" in trade_result:
            return trade_result["actual_direction"].lower().strip()

        pnl = trade_result.get("realized_pnl", 0)
        side = trade_result.get("side", "buy").lower()

        if pnl > 0:
            return side  # the side that was taken was correct
        elif pnl < 0:
            return "sell" if side == "buy" else "buy"
        return "hold"
