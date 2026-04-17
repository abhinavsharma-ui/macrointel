#!/usr/bin/env python3
"""
FinBERT Headline Sentiment
===========================
Runs headlines through ProsusAI/finbert to produce:
    finbert_sentiment = rolling 4h (positive_prob - negative_prob)

Design:
  * Lazy-load the model on first call (~440MB); one instance shared per process.
  * Cache each scored headline by SHA1(hash(title)) so we never reprocess.
  * Hard 3-second timeout per inference batch — on timeout, return 0.0 neutral
    and the caller applies a 0.05 confidence penalty (soft gate).
  * Rolling window: only headlines published within last 4h are aggregated.
  * Both US (Finnhub /news) and India news feeds supported. India feed URL is
    configurable via INDIA_NEWS_FEED_URL env var; expected JSON schema matches
    Finnhub's.

Public API:
  score_symbol(symbol, is_india=False)  -> dict:
      {
        "finbert_sentiment": float in [-1, 1],
        "n_headlines": int,
        "sentiment_available": 0|1,
      }

Standalone test:
    python -m project.pipeline.finbert_sentiment --symbol AAPL
    python -m project.pipeline.finbert_sentiment --symbol RELIANCE --india
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FINBERT_MODEL_NAME = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
FINBERT_INFERENCE_TIMEOUT = float(os.getenv("FINBERT_INFERENCE_TIMEOUT", "3.0"))
FINBERT_WINDOW_HOURS = float(os.getenv("FINBERT_WINDOW_HOURS", "4.0"))
FINBERT_CACHE_MAX = int(os.getenv("FINBERT_CACHE_MAX", "5000"))
FINBERT_BATCH_SIZE = int(os.getenv("FINBERT_BATCH_SIZE", "8"))

NEUTRAL_RESULT = {"finbert_sentiment": 0.0, "n_headlines": 0, "sentiment_available": 0}

_MODEL_LOCK = threading.Lock()
_MODEL_STATE: Dict[str, Any] = {"model": None, "tokenizer": None, "device": None, "failed": False}
_SCORE_CACHE: Dict[str, float] = {}  # hash -> signed score (positive - negative)
_CACHE_LOCK = threading.Lock()
_INFERENCE_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="finbert")


# -----------------------------------------------------------------------------
# Headline fetchers
# -----------------------------------------------------------------------------


async def _fetch_finnhub_headlines(symbol: str, api_key: str, hours: float) -> List[Dict[str, Any]]:
    try:
        import aiohttp
    except ImportError:
        return []
    if not api_key:
        return []
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=max(hours, 24))).date().isoformat()
    end = now.date().isoformat()
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": symbol, "from": start, "to": end, "token": api_key}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("Finnhub news HTTP %s for %s", resp.status, symbol)
                    return []
                data = await resp.json()
    except Exception as exc:  # pragma: no cover
        logger.warning("Finnhub news fetch failed for %s: %s", symbol, exc)
        return []
    return data if isinstance(data, list) else []


async def _fetch_india_headlines(symbol: str, hours: float) -> List[Dict[str, Any]]:
    """
    India news feed. Expected env:
      INDIA_NEWS_FEED_URL - JSON endpoint returning list of {headline, datetime, symbol?}
      INDIA_NEWS_API_KEY  - optional bearer token
    """
    feed_url = os.getenv("INDIA_NEWS_FEED_URL")
    if not feed_url:
        return []
    try:
        import aiohttp
    except ImportError:
        return []
    headers = {}
    if key := os.getenv("INDIA_NEWS_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    params = {"symbol": symbol, "hours": int(hours * 2)}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
            async with session.get(feed_url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    logger.warning("India news HTTP %s for %s", resp.status, symbol)
                    return []
                data = await resp.json()
    except Exception as exc:  # pragma: no cover
        logger.warning("India news fetch failed for %s: %s", symbol, exc)
        return []
    if isinstance(data, dict):
        data = data.get("articles") or data.get("data") or []
    return data if isinstance(data, list) else []


def _normalize_article(article: Dict[str, Any]) -> Optional[Tuple[str, datetime]]:
    """Return (headline, ts) or None."""
    headline = article.get("headline") or article.get("title") or ""
    if not isinstance(headline, str) or not headline.strip():
        return None
    ts_val = article.get("datetime") or article.get("published_at") or article.get("publishedAt")
    if ts_val is None:
        return None
    try:
        if isinstance(ts_val, (int, float)):
            ts = datetime.fromtimestamp(float(ts_val), tz=timezone.utc)
        else:
            s = str(ts_val).replace("Z", "+00:00")
            ts = datetime.fromisoformat(s)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return headline.strip(), ts


# -----------------------------------------------------------------------------
# Model loading & inference
# -----------------------------------------------------------------------------


def _load_finbert() -> bool:
    """Load FinBERT lazily. Thread-safe. Returns True on success."""
    if _MODEL_STATE["failed"]:
        return False
    if _MODEL_STATE["model"] is not None:
        return True
    with _MODEL_LOCK:
        if _MODEL_STATE["model"] is not None:
            return True
        if _MODEL_STATE["failed"]:
            return False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError as exc:
            logger.warning("transformers/torch not installed; FinBERT disabled (%s)", exc)
            _MODEL_STATE["failed"] = True
            return False
        try:
            tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            _MODEL_STATE["tokenizer"] = tokenizer
            _MODEL_STATE["model"] = model
            _MODEL_STATE["device"] = device
            logger.info("FinBERT loaded on %s", device)
            return True
        except Exception as exc:
            logger.warning("FinBERT load failed: %s; sentiment gated to neutral", exc)
            _MODEL_STATE["failed"] = True
            return False


def _headline_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:20]


def _score_batch_blocking(texts: List[str]) -> List[float]:
    """Run FinBERT on a list of texts. Returns positive_prob - negative_prob per text."""
    if not texts:
        return []
    if not _load_finbert():
        return [0.0] * len(texts)
    try:
        import torch
    except ImportError:
        return [0.0] * len(texts)

    tokenizer = _MODEL_STATE["tokenizer"]
    model = _MODEL_STATE["model"]
    device = _MODEL_STATE["device"]

    # FinBERT label order: positive, negative, neutral (per ProsusAI config)
    id2label = model.config.id2label or {0: "positive", 1: "negative", 2: "neutral"}
    pos_idx = next((i for i, lab in id2label.items() if str(lab).lower() == "positive"), 0)
    neg_idx = next((i for i, lab in id2label.items() if str(lab).lower() == "negative"), 1)

    scores: List[float] = []
    for i in range(0, len(texts), FINBERT_BATCH_SIZE):
        batch = texts[i : i + FINBERT_BATCH_SIZE]
        try:
            with torch.no_grad():
                enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                scores.append(float(row[pos_idx] - row[neg_idx]))
        except Exception as exc:
            logger.warning("FinBERT batch inference failed: %s", exc)
            scores.extend([0.0] * len(batch))
    return scores


def _score_with_cache(texts: List[str]) -> Tuple[List[float], int]:
    """Return (scores, num_new_computed). Cached hits reused."""
    out: List[Optional[float]] = [None] * len(texts)
    to_compute_idx: List[int] = []
    to_compute_text: List[str] = []

    with _CACHE_LOCK:
        for i, text in enumerate(texts):
            h = _headline_hash(text)
            cached = _SCORE_CACHE.get(h)
            if cached is not None:
                out[i] = cached
            else:
                to_compute_idx.append(i)
                to_compute_text.append(text)

    if to_compute_text:
        computed = _score_batch_blocking(to_compute_text)
        with _CACHE_LOCK:
            for text, score in zip(to_compute_text, computed):
                _SCORE_CACHE[_headline_hash(text)] = score
            if len(_SCORE_CACHE) > FINBERT_CACHE_MAX:
                # evict oldest ~20% arbitrarily (dict insertion order)
                drop = len(_SCORE_CACHE) - int(FINBERT_CACHE_MAX * 0.8)
                for key in list(_SCORE_CACHE.keys())[:drop]:
                    _SCORE_CACHE.pop(key, None)
        for idx, score in zip(to_compute_idx, computed):
            out[idx] = score

    final = [s if s is not None else 0.0 for s in out]
    return final, len(to_compute_text)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


async def score_symbol(
    symbol: str,
    *,
    is_india: bool = False,
    finnhub_api_key: Optional[str] = None,
    window_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Returns {finbert_sentiment, n_headlines, sentiment_available}.
    NEVER raises. 0.0 on any failure/timeout.
    """
    hours = window_hours if window_hours is not None else FINBERT_WINDOW_HOURS
    if is_india:
        articles = await _fetch_india_headlines(symbol, hours)
    else:
        key = finnhub_api_key or os.getenv("FINNHUB_API_KEY") or (os.getenv("FINNHUB_API_KEYS") or "").split(",")[0]
        articles = await _fetch_finnhub_headlines(symbol, key, hours)

    if not articles:
        return dict(NEUTRAL_RESULT)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    texts: List[str] = []
    for art in articles:
        norm = _normalize_article(art)
        if not norm:
            continue
        headline, ts = norm
        if ts >= cutoff:
            texts.append(headline)

    if not texts:
        return dict(NEUTRAL_RESULT)

    loop = asyncio.get_event_loop()
    try:
        fut = loop.run_in_executor(_INFERENCE_POOL, _score_with_cache, texts)
        scores, _ = await asyncio.wait_for(fut, timeout=FINBERT_INFERENCE_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("FinBERT timeout for %s after %.1fs; neutral", symbol, FINBERT_INFERENCE_TIMEOUT)
        return dict(NEUTRAL_RESULT)
    except Exception as exc:  # pragma: no cover
        logger.warning("FinBERT scoring failure for %s: %s", symbol, exc)
        return dict(NEUTRAL_RESULT)

    if not scores:
        return dict(NEUTRAL_RESULT)

    avg = float(sum(scores) / len(scores))
    return {
        "finbert_sentiment": max(-1.0, min(1.0, avg)),
        "n_headlines": len(scores),
        "sentiment_available": 1,
    }


def score_symbol_sync(symbol: str, **kwargs: Any) -> Dict[str, Any]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return dict(NEUTRAL_RESULT)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(score_symbol(symbol, **kwargs))


# -----------------------------------------------------------------------------
# __main__
# -----------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--india", action="store_true")
    p.add_argument("--window-hours", type=float, default=FINBERT_WINDOW_HOURS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = score_symbol_sync(args.symbol, is_india=args.india, window_hours=args.window_hours)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _cli()
