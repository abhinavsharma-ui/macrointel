"""
Event intelligence digest builder.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pipeline.universe import get_company_name, get_market, get_sector


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except Exception:
        return max(minimum, default)


def _ensure_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    if not value:
        return datetime.now(timezone.utc)
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _json_default(value: Any):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _lane_label(lane: str) -> str:
    return {
        "normal": "Normal Trading",
        "day": "Day Trading",
        "crypto": "Crypto Scalper",
    }.get(str(lane or "").lower(), "Normal Trading")


def _canonical_key(article: Dict[str, Any], symbol: str) -> str:
    url = str(article.get("url") or "").strip().lower()
    title = " ".join(str(article.get("title") or "").lower().split())
    source = str(article.get("source") or "").strip().lower()
    return url or f"{symbol.lower()}::{source}::{title}"


def _article_score(article: Dict[str, Any], generated_at: datetime) -> float:
    weighted = abs(_safe_float(article.get("weighted_compound_score"), 0.0))
    quality = _safe_float(article.get("source_quality_score"), 0.0)
    official = _safe_float(article.get("official_event_hit"), 0.0) + _safe_float(article.get("is_official"), 0.0)
    filing = _safe_float(article.get("filing_event_hit"), 0.0) + _safe_float(article.get("filing_article_count"), 0.0)
    risk = _safe_float(article.get("new_risk_factors"), 0.0)
    press_release = _safe_float(article.get("press_release_count"), 0.0)
    published_at = _ensure_datetime(article.get("publishedAt") or article.get("date"))
    age_hours = max(0.0, (generated_at - published_at).total_seconds() / 3600.0)
    freshness = max(0.0, 2.5 - min(age_hours / 12.0, 2.5))
    return round(
        (weighted * 2.2)
        + (quality * 1.4)
        + (official * 3.6)
        + (filing * 2.4)
        + (risk * 1.5)
        + (press_release * 0.35)
        + freshness,
        4,
    )


def _headline_rows(sentiment_payload: Dict[str, Any], generated_at: datetime) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    headlines_by_symbol = sentiment_payload.get("headlines", {}) if isinstance(sentiment_payload, dict) else {}
    for symbol, articles in (headlines_by_symbol or {}).items():
        for article in articles or []:
            row = dict(article)
            row["symbol"] = symbol
            row["company_name"] = get_company_name(symbol)
            row["market"] = get_market(symbol).upper()
            row["sector"] = get_sector(symbol)
            row["published_at"] = _ensure_datetime(article.get("publishedAt") or article.get("date")).isoformat()
            row["headline_score"] = _article_score(article, generated_at)
            key = _canonical_key(article, symbol)
            current = grouped.get(key)
            if current is None:
                row["symbols"] = [symbol]
                grouped[key] = row
            else:
                current_symbols = set(current.get("symbols", []))
                current_symbols.add(symbol)
                current["symbols"] = sorted(current_symbols)
                if row["headline_score"] > _safe_float(current.get("headline_score"), 0.0):
                    keep_symbols = current["symbols"]
                    grouped[key] = {**row, "symbols": keep_symbols}
    rows = list(grouped.values())
    rows.sort(key=lambda item: (_safe_float(item.get("headline_score"), 0.0), item.get("published_at", "")), reverse=True)
    return rows


def _signal_snapshot(signal_store: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    summary = {
        "normal": {"signals": 0, "trade_ready": 0, "buy": 0, "sell": 0},
        "day": {"signals": 0, "trade_ready": 0, "buy": 0, "sell": 0},
        "crypto": {"signals": 0, "trade_ready": 0, "buy": 0, "sell": 0},
    }
    for signal in (signal_store or {}).values():
        if not isinstance(signal, dict):
            continue
        lane = str(signal.get("lane") or "normal").lower()
        if lane not in summary:
            lane = "normal"
        signal_dir = str(signal.get("signal") or "neutral").lower()
        summary[lane]["signals"] += 1
        if signal.get("trade_eligible") or ((signal.get("meta_decision") or {}).get("take_trade")):
            summary[lane]["trade_ready"] += 1
        if signal_dir == "buy":
            summary[lane]["buy"] += 1
        elif signal_dir == "sell":
            summary[lane]["sell"] += 1
    return summary


def _symbol_heat_rows(
    sentiment_payload: Dict[str, Any],
    signal_store: Optional[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    headlines = sentiment_payload.get("headlines", {}) if isinstance(sentiment_payload, dict) else {}
    providers = sentiment_payload.get("news_provider_by_symbol", {}) if isinstance(sentiment_payload, dict) else {}
    signal_rows = signal_store or {}

    for symbol, articles in (headlines or {}).items():
        records = list(articles or [])
        if not records:
            continue
        weighted_scores = [_safe_float(item.get("weighted_compound_score"), 0.0) for item in records]
        raw_scores = [_safe_float(item.get("compound_score"), 0.0) for item in records]
        official_hits = sum(1 for item in records if _safe_float(item.get("official_event_hit"), 0.0) > 0)
        filing_hits = sum(1 for item in records if _safe_float(item.get("filing_event_hit"), 0.0) > 0)
        risk_flags = sum(1 for item in records if _safe_float(item.get("new_risk_factors"), 0.0) > 0)
        source_quality = sum(_safe_float(item.get("source_quality_score"), 0.0) for item in records) / max(len(records), 1)
        signal = None
        for candidate in signal_rows.values():
            if isinstance(candidate, dict) and str(candidate.get("symbol") or "").upper() == symbol.upper():
                signal = candidate
                break
        lane = str((signal or {}).get("lane") or "normal").lower()
        trade_ready = bool((signal or {}).get("trade_eligible") or (((signal or {}).get("meta_decision") or {}).get("take_trade")))
        take_probability = _safe_float((signal or {}).get("take_probability") or (((signal or {}).get("meta_decision") or {}).get("take_probability")), 0.0)
        conviction = _safe_float((signal or {}).get("conviction_score"), 0.0)
        rows.append(
            {
                "symbol": symbol,
                "company_name": get_company_name(symbol),
                "market": get_market(symbol).upper(),
                "sector": get_sector(symbol),
                "headline_count": len(records),
                "provider": providers.get(symbol, ""),
                "mean_weighted_sentiment": round(sum(weighted_scores) / max(len(weighted_scores), 1), 4),
                "mean_raw_sentiment": round(sum(raw_scores) / max(len(raw_scores), 1), 4),
                "official_hits": official_hits,
                "filing_hits": filing_hits,
                "risk_flags": risk_flags,
                "source_quality_score": round(source_quality, 3),
                "lane": lane,
                "lane_label": _lane_label(lane),
                "trade_ready": trade_ready,
                "take_probability": round(take_probability, 4),
                "conviction_score": round(conviction, 3),
            }
        )

    rows.sort(
        key=lambda item: (
            item["trade_ready"],
            abs(item["mean_weighted_sentiment"]) + (item["official_hits"] * 0.25) + (item["risk_flags"] * 0.2),
            item["headline_count"],
        ),
        reverse=True,
    )
    return rows


def _summary_bullets(
    generated_at: datetime,
    summary: Dict[str, Any],
    top_bullish: List[Dict[str, Any]],
    top_bearish: List[Dict[str, Any]],
    lane_snapshot: Dict[str, Dict[str, Any]],
) -> List[str]:
    bullets: List[str] = []
    covered = int(summary.get("symbols_with_news", 0) or 0)
    total_symbols = int(summary.get("symbols_covered", 0) or 0)
    official = int(summary.get("official_symbols", 0) or 0)
    provider_counts = summary.get("provider_counts", {}) or {}
    provider_text = ", ".join(f"{name}={count}" for name, count in list(provider_counts.items())[:4]) or "no provider mix yet"
    bullets.append(
        f"{covered}/{total_symbols} symbols had fresh news in the last {summary.get('window_days', 3)} days; official catalysts hit {official} symbols ({provider_text})."
    )
    if top_bullish:
        bullish_text = ", ".join(f"{row['symbol']} ({row['mean_weighted_sentiment']:+.2f})" for row in top_bullish[:3])
        bullets.append(f"Most bullish event pressure right now: {bullish_text}.")
    if top_bearish:
        bearish_text = ", ".join(f"{row['symbol']} ({row['mean_weighted_sentiment']:+.2f})" for row in top_bearish[:3])
        bullets.append(f"Most bearish or risk-heavy names right now: {bearish_text}.")
    lane_parts = []
    for lane in ("normal", "day", "crypto"):
        snapshot = lane_snapshot.get(lane, {})
        lane_parts.append(f"{_lane_label(lane)} {snapshot.get('trade_ready', 0)} ready / {snapshot.get('signals', 0)} live")
    bullets.append(f"Lane readiness at {generated_at.strftime('%H:%M UTC')}: " + "; ".join(lane_parts) + ".")
    return bullets


def _markdown_report(payload: Dict[str, Any]) -> str:
    summary = payload.get("summary", {}) or {}
    bullets = payload.get("bullets", []) or []
    headlines = payload.get("top_headlines", []) or []
    symbols = payload.get("symbol_heat", []) or []
    lane_snapshot = payload.get("lane_snapshot", {}) or {}

    lines = [
        f"# Event Intelligence Report",
        "",
        f"- Generated: {payload.get('generated_at', '-')}",
        f"- Universe mode: {payload.get('universe_mode', '-')}",
        f"- Symbols with news: {summary.get('symbols_with_news', 0)}/{summary.get('symbols_covered', 0)}",
        f"- Official catalyst symbols: {summary.get('official_symbols', 0)}",
        "",
        "## Summary",
        "",
    ]
    for bullet in bullets:
        lines.append(f"- {bullet}")
    lines.extend(["", "## Lane Snapshot", ""])
    for lane in ("normal", "day", "crypto"):
        snap = lane_snapshot.get(lane, {}) or {}
        lines.append(
            f"- {_lane_label(lane)}: {snap.get('trade_ready', 0)} trade-ready / {snap.get('signals', 0)} live "
            f"(buy {snap.get('buy', 0)}, sell {snap.get('sell', 0)})"
        )
    lines.extend(["", "## Top Headlines", ""])
    if headlines:
        for item in headlines:
            lines.append(
                f"- **{item.get('symbol', '-') }** {item.get('title', '').strip()} "
                f"({item.get('source', item.get('provider', 'news'))}, score {item.get('headline_score', 0):.2f})"
            )
    else:
        lines.append("- No headlines captured yet.")
    lines.extend(["", "## Symbol Heat", ""])
    if symbols:
        for item in symbols[:12]:
            lines.append(
                f"- **{item.get('symbol', '-') }** {item.get('lane_label', '-')}: "
                f"sentiment {item.get('mean_weighted_sentiment', 0):+.2f}, headlines {item.get('headline_count', 0)}, "
                f"official {item.get('official_hits', 0)}, risk flags {item.get('risk_flags', 0)}, "
                f"trade-ready {item.get('trade_ready', False)}"
            )
    else:
        lines.append("- No symbol heat yet.")
    return "\n".join(lines) + "\n"


def _prune_old_snapshots(directory: Path, retention_days: int) -> None:
    if retention_days <= 0 or not directory.exists():
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for path in directory.glob("snapshot-*.*"):
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < cutoff:
                path.unlink(missing_ok=True)
        except Exception:
            continue


def build_event_intelligence(
    sentiment_payload: Dict[str, Any],
    signal_store: Optional[Dict[str, Dict[str, Any]]] = None,
    output_dir: Optional[str | Path] = None,
    universe_mode: str = "full",
) -> Dict[str, Any]:
    generated_at = datetime.now(timezone.utc)
    max_headlines = _env_int("EVENT_INTEL_MAX_HEADLINES", 18)
    max_symbol_rows = _env_int("EVENT_INTEL_MAX_SYMBOL_ROWS", 18)
    retention_days = _env_int("EVENT_INTEL_RETENTION_DAYS", 7)
    window_days = _env_int("EVENT_INTEL_WINDOW_DAYS", 3)
    out_dir = Path(output_dir or os.getenv("EVENT_INTEL_DIR", "data/event_intel"))
    out_dir.mkdir(parents=True, exist_ok=True)

    top_headlines = _headline_rows(sentiment_payload, generated_at)[:max_headlines]
    symbol_heat = _symbol_heat_rows(sentiment_payload, signal_store)[:max_symbol_rows]
    top_bullish = [row for row in symbol_heat if row.get("mean_weighted_sentiment", 0.0) > 0.05]
    top_bearish = [
        row
        for row in symbol_heat
        if row.get("mean_weighted_sentiment", 0.0) < -0.05 or row.get("risk_flags", 0) > 0 or row.get("official_hits", 0) > 0
    ]
    top_bullish.sort(key=lambda row: (row["mean_weighted_sentiment"], row["headline_count"], row["trade_ready"]), reverse=True)
    top_bearish.sort(
        key=lambda row: (abs(row["mean_weighted_sentiment"]), row["risk_flags"], row["official_hits"], row["headline_count"]),
        reverse=True,
    )

    news_summary = sentiment_payload.get("news_summary", {}) if isinstance(sentiment_payload, dict) else {}
    provider_counts = dict(Counter(news_summary.get("provider_counts", {}) or {}))
    lane_snapshot = _signal_snapshot(signal_store)
    summary = {
        "window_days": window_days,
        "symbols_with_news": int(news_summary.get("symbols_with_news", 0) or 0),
        "symbols_covered": int(sentiment_payload.get("symbols_covered", 0) or 0),
        "official_symbols": int(news_summary.get("official_symbols", 0) or 0),
        "fallback_symbols": int(news_summary.get("fallback_symbols", 0) or 0),
        "provider_counts": provider_counts,
        "headline_count": len(top_headlines),
        "net_sentiment": round(sum(row.get("mean_weighted_sentiment", 0.0) for row in symbol_heat), 4),
        "generated_epoch": int(generated_at.timestamp()),
    }
    bullets = _summary_bullets(generated_at, summary, top_bullish, top_bearish, lane_snapshot)

    payload = {
        "generated_at": generated_at.isoformat(),
        "universe_mode": universe_mode,
        "summary": summary,
        "bullets": bullets,
        "lane_snapshot": lane_snapshot,
        "top_headlines": top_headlines,
        "symbol_heat": symbol_heat,
        "top_bullish": top_bullish[:8],
        "top_bearish": top_bearish[:8],
    }

    stamp = generated_at.strftime("%Y%m%d-%H%M%S")
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    (out_dir / "latest.md").write_text(_markdown_report(payload), encoding="utf-8")
    (out_dir / f"snapshot-{stamp}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    (out_dir / f"snapshot-{stamp}.md").write_text(_markdown_report(payload), encoding="utf-8")
    _prune_old_snapshots(out_dir, retention_days)
    return payload


def load_latest_event_intelligence(output_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    out_dir = Path(output_dir or os.getenv("EVENT_INTEL_DIR", "data/event_intel"))
    latest = out_dir / "latest.json"
    if not latest.exists():
        return {}
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return {}
