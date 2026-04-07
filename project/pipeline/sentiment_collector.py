"""
Sentiment and News Data Pipeline
================================
Free-first multi-source news ingestion with:
  - alias-aware company-name queries
  - Google News RSS fallback
  - official NSE and SEC filings feeds
  - press-release style search queries
  - optional API providers when keys exist
  - source weighting, dedupe, and low-noise logging
"""

import html
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote_plus, urlparse

import pandas as pd
import requests

from pipeline.provider_utils import APIKeyPool, FetchOutcome, looks_rate_limited, parse_api_keys, stable_rotate
from pipeline.universe import get_company_name, get_market, get_sector, get_symbol_aliases

logger = logging.getLogger(__name__)

USER_AGENT = os.getenv(
    "NEWS_USER_AGENT",
    "MacroIntel/1.0 contact: research@example.com",
)
HEADERS = {"User-Agent": USER_AGENT}

GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"
YAHOO_RSS_BASE = "https://feeds.finance.yahoo.com/rss/2.0/headline"
NEWS_API_BASE = "https://newsapi.org/v2/everything"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
FINNHUB_COMPANY_NEWS_BASE = "https://finnhub.io/api/v1/company-news"
GNEWS_BASE = "https://gnews.io/api/v4/search"
SEC_ATOM_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"

NEWS_PROVIDER_ORDER = [
    name.strip().lower()
    for name in os.getenv(
        "NEWS_PROVIDER_ORDER",
        "google_rss,rss,nse_announcements,sec_filings,press_releases,finnhub,alpha_vantage,newsapi,gnews,bse_announcements",
    ).split(",")
    if name.strip()
]
NEWS_MAX_PROVIDER_ATTEMPTS = max(1, int(os.getenv("NEWS_MAX_PROVIDER_ATTEMPTS", "4")))
NEWS_MIN_ARTICLES = max(1, int(os.getenv("NEWS_MIN_ARTICLES", "5")))
NEWS_MAX_ARTICLES = max(NEWS_MIN_ARTICLES, int(os.getenv("NEWS_MAX_ARTICLES", "20")))
NEWS_QUERY_VARIANTS = max(1, int(os.getenv("NEWS_QUERY_VARIANTS", "4")))
NEWS_LOG_PROGRESS_EVERY = max(1, int(os.getenv("NEWS_LOG_PROGRESS_EVERY", "10")))
NEWS_FAILURE_PREVIEW = max(1, int(os.getenv("NEWS_FAILURE_PREVIEW", "6")))
NEWS_OFFICIAL_HIT_TARGET = max(1, int(os.getenv("NEWS_OFFICIAL_HIT_TARGET", "2")))
NEWS_DUPLICATE_WINDOW_HOURS = max(1, int(os.getenv("NEWS_DUPLICATE_WINDOW_HOURS", "36")))

NSE_RSS_FEEDS = {
    "online_announcements": "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml",
    "financial_results": "https://nsearchives.nseindia.com/content/RSS/Financial_Results.xml",
    "board_meetings": "https://nsearchives.nseindia.com/content/RSS/Board_Meetings.xml",
    "corporate_action": "https://nsearchives.nseindia.com/content/RSS/Corporate_action.xml",
    "insider_trading": "https://nsearchives.nseindia.com/content/RSS/Insider_Trading.xml",
    "annual_reports": "https://nsearchives.nseindia.com/content/RSS/Annual_Reports.xml",
}

PRESS_RELEASE_DOMAINS = {
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "accesswire.com",
    "finance.yahoo.com",
}
OFFICIAL_DOMAINS = {
    "sec.gov",
    "www.sec.gov",
    "nseindia.com",
    "www.nseindia.com",
    "nsearchives.nseindia.com",
    "bseindia.com",
    "www.bseindia.com",
}
MAJOR_MEDIA_DOMAINS = {
    "reuters.com",
    "www.reuters.com",
    "bloomberg.com",
    "www.bloomberg.com",
    "cnbc.com",
    "www.cnbc.com",
    "wsj.com",
    "www.wsj.com",
    "ft.com",
    "www.ft.com",
    "marketwatch.com",
    "www.marketwatch.com",
    "finance.yahoo.com",
    "economictimes.indiatimes.com",
    "moneycontrol.com",
    "livemint.com",
    "business-standard.com",
}

RISK_FACTOR_KEYWORDS = (
    "risk factor",
    "material weakness",
    "going concern",
    "impairment",
    "restatement",
    "litigation",
    "investigation",
    "non-compliance",
)

EARNINGS_TONE_KEYWORDS = (
    "earnings call",
    "conference call",
    "prepared remarks",
    "earnings webcast",
    "investor call",
)


def _get_vader():
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk

        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning("NLTK not installed. Run: pip install nltk. Returning neutral scorer.")
        return None


def score_text(text: str, vader=None) -> float:
    if vader is not None:
        return vader.polarity_scores(text)["compound"]

    positive = ["beat", "surge", "rally", "growth", "profit", "record", "upgrade", "bull"]
    negative = ["miss", "crash", "plunge", "loss", "downgrade", "layoff", "bear", "fraud"]
    text_l = text.lower()
    pos = sum(text_l.count(word) for word in positive)
    neg = sum(text_l.count(word) for word in negative)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _normalize_space(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    return _normalize_space(text)


def _safe_domain(url: str) -> str:
    try:
        return urlparse(url or "").netloc.lower()
    except Exception:
        return ""


def _canonicalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        return f"{parsed.netloc.lower()}{path}".lower()
    except Exception:
        return url.lower().strip()


def _normalize_title(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", (value or "").lower())
    return _normalize_space(cleaned)


def _token_set(value: str) -> set:
    tokens = re.findall(r"[a-z0-9]+", (value or "").lower())
    return {token for token in tokens if len(token) > 2}


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 1.0 if not left_tokens and not right_tokens else 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _parse_timestamp(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = pd.to_datetime(value, utc=True)
        if pd.isna(parsed):
            raise ValueError("NaT")
        return parsed.to_pydatetime()
    except Exception:
        try:
            return parsedate_to_datetime(value).astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)


def _article_date(value: str) -> date:
    return _parse_timestamp(value).date()


def _query_region(symbol: str) -> str:
    return "IN" if symbol.endswith(".NS") else "US"


def _alias_queries(symbol: str) -> List[str]:
    aliases = get_symbol_aliases(symbol)
    company_name = get_company_name(symbol)
    sector = get_sector(symbol)
    raw = symbol.replace(".NS", "").replace("^", "").strip()
    variants: List[str] = []

    if symbol.endswith(".NS"):
        variants.extend([
            company_name,
            f"{company_name} ltd",
            f"{company_name} limited",
            raw,
        ])
    else:
        variants.extend([
            company_name,
            f"{company_name} stock",
            raw,
        ])

    if sector and sector != "Other":
        variants.extend([
            f"{company_name} {sector}",
            f"{raw} {sector}",
        ])

    variants.extend(aliases)
    seen = set()
    ordered: List[str] = []
    for item in variants:
        cleaned = _normalize_space(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered[:NEWS_QUERY_VARIANTS]


def _matches_aliases(symbol: str, text: str) -> bool:
    haystack = f" {re.sub(r'[^a-z0-9 ]+', ' ', (text or '').lower())} "
    for alias in get_symbol_aliases(symbol):
        alias_norm = _normalize_space(alias).lower()
        if not alias_norm:
            continue
        if len(alias_norm) <= 4:
            if f" {alias_norm} " in haystack:
                return True
        elif alias_norm in haystack:
            return True
    return False


def _source_flags(article: Dict, provider_name: str) -> Dict[str, float]:
    domain = _safe_domain(article.get("url", ""))
    title = (article.get("title") or "").lower()
    source = (article.get("source") or "").lower()
    is_official = domain in OFFICIAL_DOMAINS or provider_name in {"nse_announcements", "sec_filings", "bse_announcements"}
    is_filing = is_official and any(token in title for token in ["8-k", "10-q", "10-k", "results", "filing", "annual report", "board meeting", "insider"]) 
    is_press_release = domain in PRESS_RELEASE_DOMAINS or provider_name == "press_releases" or "press release" in title

    if is_official:
        quality = 1.00
    elif is_press_release:
        quality = 0.88
    elif domain in MAJOR_MEDIA_DOMAINS or source in {"reuters", "bloomberg", "cnbc", "marketwatch", "yahoo finance rss", "google news"}:
        quality = 0.72
    elif domain:
        quality = 0.52
    else:
        quality = 0.40

    return {
        "source_quality_score": quality,
        "is_official": float(is_official),
        "is_filing": float(is_filing),
        "is_press_release": float(is_press_release),
        "official_event_hit": float(is_official),
        "filing_event_hit": float(is_filing),
    }


def _dedupe_key(article: Dict) -> str:
    canonical_url = _canonicalize_url(article.get("url", ""))
    if canonical_url:
        return canonical_url
    title = _normalize_title(article.get("title", ""))
    published = _parse_timestamp(article.get("publishedAt", "")).strftime("%Y-%m-%d-%H")
    return f"{title}|{published}"


def _article_rank(article: Dict) -> tuple:
    ts = _parse_timestamp(article.get("publishedAt", ""))
    return (
        float(article.get("official_event_hit", 0.0) or 0.0),
        float(article.get("filing_event_hit", 0.0) or 0.0),
        float(article.get("is_press_release", 0.0) or 0.0),
        float(article.get("source_quality_score", 0.0) or 0.0),
        ts.timestamp(),
    )


def _finalize_article(article: Dict, provider_name: str, symbol: str) -> Optional[Dict]:
    title = _normalize_space(article.get("title", ""))
    description = _strip_html(article.get("description", "") or article.get("summary", ""))
    published_at = article.get("publishedAt") or article.get("published_at") or article.get("time_published") or ""
    if not title:
        return None
    requires_alias_match = provider_name not in {"rss", "finnhub", "alpha_vantage", "sec_filings"}
    if requires_alias_match and not _matches_aliases(symbol, f"{title} {description}"):
        return None

    item = {
        "title": title,
        "description": description or title,
        "publishedAt": _parse_timestamp(str(published_at)).isoformat(),
        "source": _normalize_space(article.get("source", provider_name.replace("_", " ").title())),
        "url": article.get("url", "") or article.get("link", ""),
        "provider": provider_name,
    }
    item.update(_source_flags(item, provider_name))
    return item


class NewsAPIFetcher:
    name = "newsapi"
    query_driven = True

    def __init__(self):
        self.keys = APIKeyPool(
            parse_api_keys("NEWS_API_KEYS", "NEWS_API_KEY"),
            default_cooldown=float(os.getenv("NEWS_API_COOLDOWN_SECONDS", "1800")),
        )
        self._last_call_by_key: Dict[str, float] = {}

    def supports(self, symbol: str) -> bool:
        return not symbol.startswith("^")

    def fetch(self, symbol: str, query: str, from_date: date, to_date: date) -> FetchOutcome:
        api_key = self.keys.acquire()
        if not api_key:
            return FetchOutcome(self.name, status="unavailable", error="No NewsAPI key available")

        elapsed = time.time() - self._last_call_by_key.get(api_key, 0.0)
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_call_by_key[api_key] = time.time()

        params = {
            "q": query,
            "from": str(from_date),
            "to": str(to_date),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": NEWS_MAX_ARTICLES,
            "apiKey": api_key,
        }
        try:
            resp = requests.get(NEWS_API_BASE, params=params, timeout=10, headers=HEADERS)
            resp.raise_for_status()
            payload = resp.json()
            articles = payload.get("articles", [])
            if not articles:
                return FetchOutcome(self.name, status="empty", error="NewsAPI returned no articles")
            return FetchOutcome(self.name, data=articles)
        except Exception as exc:
            error = str(exc)
            if looks_rate_limited(error):
                self.keys.cool_down(api_key)
                return FetchOutcome(self.name, status="rate_limited", error=error)
            return FetchOutcome(self.name, status="error", error=error)


class FinnhubNewsFetcher:
    name = "finnhub"
    query_driven = False

    def __init__(self):
        self.keys = APIKeyPool(
            parse_api_keys("FINNHUB_API_KEYS", "FINNHUB_API_KEY"),
            default_cooldown=float(os.getenv("FINNHUB_NEWS_COOLDOWN_SECONDS", "300")),
        )

    def supports(self, symbol: str) -> bool:
        return not symbol.endswith(".NS") and not symbol.startswith("^")

    def fetch(self, symbol: str, query: str, from_date: date, to_date: date) -> FetchOutcome:
        api_key = self.keys.acquire()
        if not api_key:
            return FetchOutcome(self.name, status="unavailable", error="No Finnhub key available")

        params = {
            "symbol": symbol,
            "from": str(from_date),
            "to": str(to_date),
            "token": api_key,
        }
        try:
            resp = requests.get(FINNHUB_COMPANY_NEWS_BASE, params=params, timeout=10, headers=HEADERS)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict):
                error = payload.get("error") or "Finnhub returned no articles"
                if looks_rate_limited(error):
                    self.keys.cool_down(api_key)
                    return FetchOutcome(self.name, status="rate_limited", error=error)
                return FetchOutcome(self.name, status="empty", error=error)
            articles = [
                {
                    "title": item.get("headline", ""),
                    "description": item.get("summary", ""),
                    "publishedAt": datetime.fromtimestamp(item.get("datetime", 0), timezone.utc).isoformat() if item.get("datetime") else "",
                    "source": item.get("source", "Finnhub"),
                    "url": item.get("url", ""),
                }
                for item in payload[:NEWS_MAX_ARTICLES]
            ]
            if not articles:
                return FetchOutcome(self.name, status="empty", error="Finnhub returned no articles")
            return FetchOutcome(self.name, data=articles)
        except Exception as exc:
            error = str(exc)
            if looks_rate_limited(error):
                self.keys.cool_down(api_key)
                return FetchOutcome(self.name, status="rate_limited", error=error)
            return FetchOutcome(self.name, status="error", error=error)


class AlphaVantageNewsFetcher:
    name = "alpha_vantage"
    query_driven = False

    def __init__(self):
        self.keys = APIKeyPool(
            parse_api_keys("ALPHA_VANTAGE_API_KEYS", "ALPHA_VANTAGE_API_KEY"),
            default_cooldown=float(os.getenv("ALPHA_VANTAGE_NEWS_COOLDOWN_SECONDS", "300")),
        )

    def supports(self, symbol: str) -> bool:
        return not symbol.endswith(".NS") and not symbol.startswith("^")

    def fetch(self, symbol: str, query: str, from_date: date, to_date: date) -> FetchOutcome:
        api_key = self.keys.acquire()
        if not api_key:
            return FetchOutcome(self.name, status="unavailable", error="No Alpha Vantage key available")

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "time_from": from_date.strftime("%Y%m%dT0000"),
            "time_to": to_date.strftime("%Y%m%dT2359"),
            "limit": str(NEWS_MAX_ARTICLES),
            "apikey": api_key,
        }
        try:
            resp = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=12, headers=HEADERS)
            resp.raise_for_status()
            payload = resp.json()
            articles = payload.get("feed")
            if not articles:
                error = payload.get("Note") or payload.get("Information") or payload.get("Error Message") or "Alpha Vantage returned no articles"
                if looks_rate_limited(error):
                    self.keys.cool_down(api_key)
                    return FetchOutcome(self.name, status="rate_limited", error=error)
                return FetchOutcome(self.name, status="empty", error=error)
            formatted = [
                {
                    "title": item.get("title", ""),
                    "description": item.get("summary", ""),
                    "publishedAt": item.get("time_published", ""),
                    "source": item.get("source", "Alpha Vantage"),
                    "url": item.get("url", ""),
                }
                for item in articles[:NEWS_MAX_ARTICLES]
            ]
            return FetchOutcome(self.name, data=formatted)
        except Exception as exc:
            error = str(exc)
            if looks_rate_limited(error):
                self.keys.cool_down(api_key)
                return FetchOutcome(self.name, status="rate_limited", error=error)
            return FetchOutcome(self.name, status="error", error=error)


class GNewsFetcher:
    name = "gnews"
    query_driven = True

    def __init__(self):
        self.keys = APIKeyPool(
            parse_api_keys("GNEWS_API_KEYS", "GNEWS_API_KEY"),
            default_cooldown=float(os.getenv("GNEWS_COOLDOWN_SECONDS", "900")),
        )

    def supports(self, symbol: str) -> bool:
        return not symbol.startswith("^")

    def fetch(self, symbol: str, query: str, from_date: date, to_date: date) -> FetchOutcome:
        api_key = self.keys.acquire()
        if not api_key:
            return FetchOutcome(self.name, status="unavailable", error="No GNews key available")

        params = {
            "q": query,
            "lang": "en",
            "from": f"{from_date.isoformat()}T00:00:00Z",
            "to": f"{to_date.isoformat()}T23:59:59Z",
            "max": NEWS_MAX_ARTICLES,
            "apikey": api_key,
        }
        try:
            resp = requests.get(GNEWS_BASE, params=params, timeout=10, headers=HEADERS)
            resp.raise_for_status()
            payload = resp.json()
            articles = payload.get("articles", [])
            if not articles:
                error = payload.get("message") or "GNews returned no articles"
                if looks_rate_limited(error):
                    self.keys.cool_down(api_key)
                    return FetchOutcome(self.name, status="rate_limited", error=error)
                return FetchOutcome(self.name, status="empty", error=error)
            return FetchOutcome(self.name, data=articles)
        except Exception as exc:
            error = str(exc)
            if looks_rate_limited(error):
                self.keys.cool_down(api_key)
                return FetchOutcome(self.name, status="rate_limited", error=error)
            return FetchOutcome(self.name, status="error", error=error)


class RSSFetcher:
    name = "rss"
    query_driven = False

    def supports(self, symbol: str) -> bool:
        return not symbol.startswith("^")

    def fetch(self, symbol: str, query: str, from_date: date, to_date: date) -> FetchOutcome:
        region = _query_region(symbol)
        try:
            resp = requests.get(
                YAHOO_RSS_BASE,
                params={"s": symbol, "region": region, "lang": "en-US"},
                timeout=8,
                headers=HEADERS,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            articles = []
            for item in items[:NEWS_MAX_ARTICLES]:
                title = item.findtext("title", "")
                articles.append(
                    {
                        "title": title,
                        "description": item.findtext("description", "") or title,
                        "publishedAt": item.findtext("pubDate", ""),
                        "source": "Yahoo Finance RSS",
                        "url": item.findtext("link", ""),
                    }
                )
            if not articles:
                return FetchOutcome(self.name, status="empty", error="Yahoo RSS returned no articles")
            return FetchOutcome(self.name, data=articles)
        except Exception as exc:
            return FetchOutcome(self.name, status="error", error=str(exc))


class GoogleNewsRSSFetcher:
    name = "google_rss"
    query_driven = True

    def supports(self, symbol: str) -> bool:
        return not symbol.startswith("^")

    def fetch(self, symbol: str, query: str, from_date: date, to_date: date) -> FetchOutcome:
        region = "IN:en" if symbol.endswith(".NS") else "US:en"
        gl = "IN" if symbol.endswith(".NS") else "US"
        params = {
            "q": query,
            "hl": "en-IN" if symbol.endswith(".NS") else "en-US",
            "gl": gl,
            "ceid": region,
        }
        try:
            resp = requests.get(GOOGLE_NEWS_RSS_BASE, params=params, timeout=10, headers=HEADERS)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            articles = []
            for item in root.findall(".//item")[:NEWS_MAX_ARTICLES]:
                title = item.findtext("title", "")
                source = item.findtext("source", "Google News")
                if not source and " - " in title:
                    title, source = title.rsplit(" - ", 1)
                articles.append(
                    {
                        "title": title,
                        "description": item.findtext("description", "") or title,
                        "publishedAt": item.findtext("pubDate", ""),
                        "source": source or "Google News",
                        "url": item.findtext("link", ""),
                    }
                )
            if not articles:
                return FetchOutcome(self.name, status="empty", error="Google News RSS returned no articles")
            return FetchOutcome(self.name, data=articles)
        except Exception as exc:
            return FetchOutcome(self.name, status="error", error=str(exc))


class CachedRSSFetcher:
    name = "cached_rss"
    query_driven = False
    _cache_ttl_seconds = max(300, int(os.getenv("OFFICIAL_RSS_CACHE_SECONDS", "900")))

    def __init__(self):
        self._cache: Dict[str, tuple] = {}

    def _get_feed_entries(self, feed_name: str, url: str) -> List[Dict]:
        now = time.time()
        cached = self._cache.get(feed_name)
        if cached and (now - cached[0]) < self._cache_ttl_seconds:
            return cached[1]

        resp = requests.get(url, timeout=12, headers=HEADERS)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        entries: List[Dict] = []
        for item in root.findall(".//item"):
            entries.append(
                {
                    "title": item.findtext("title", ""),
                    "description": item.findtext("description", ""),
                    "publishedAt": item.findtext("pubDate", ""),
                    "source": feed_name.replace("_", " ").title(),
                    "url": item.findtext("link", ""),
                }
            )
        self._cache[feed_name] = (now, entries)
        return entries


class NSEAnnouncementsFetcher(CachedRSSFetcher):
    name = "nse_announcements"

    def supports(self, symbol: str) -> bool:
        return symbol.endswith(".NS")

    def fetch(self, symbol: str, query: str, from_date: date, to_date: date) -> FetchOutcome:
        articles: List[Dict] = []
        errors: List[str] = []
        for feed_name, url in NSE_RSS_FEEDS.items():
            try:
                for item in self._get_feed_entries(feed_name, url):
                    published = _article_date(item.get("publishedAt", ""))
                    if published < from_date or published > to_date:
                        continue
                    if not _matches_aliases(symbol, f"{item.get('title', '')} {item.get('description', '')}"):
                        continue
                    enriched = dict(item)
                    enriched["source"] = f"NSE {feed_name.replace('_', ' ').title()}"
                    articles.append(enriched)
            except Exception as exc:
                errors.append(f"{feed_name}: {exc}")

        if articles:
            articles.sort(key=lambda article: _parse_timestamp(article.get("publishedAt", "")), reverse=True)
            return FetchOutcome(self.name, data=articles[:NEWS_MAX_ARTICLES])
        if errors:
            return FetchOutcome(self.name, status="error", error=" | ".join(errors[:3]))
        return FetchOutcome(self.name, status="empty", error="No NSE announcements matched aliases")


class BSEAnnouncementsFetcher(GoogleNewsRSSFetcher):
    name = "bse_announcements"
    query_driven = True

    def supports(self, symbol: str) -> bool:
        return symbol.endswith(".NS")

    def fetch(self, symbol: str, query: str, from_date: date, to_date: date) -> FetchOutcome:
        return super().fetch(symbol, f'site:bseindia.com "{query}"', from_date, to_date)


class SECFilingsFetcher:
    name = "sec_filings"
    query_driven = False

    def supports(self, symbol: str) -> bool:
        return get_market(symbol) == "us" and not symbol.startswith("^")

    def fetch(self, symbol: str, query: str, from_date: date, to_date: date) -> FetchOutcome:
        params = {
            "action": "getcompany",
            "CIK": symbol,
            "owner": "exclude",
            "count": "10",
            "output": "atom",
        }
        try:
            resp = requests.get(SEC_ATOM_BASE, params=params, timeout=12, headers=HEADERS)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = []
            for entry in root.findall("atom:entry", ns):
                title = entry.findtext("atom:title", default="", namespaces=ns)
                summary = entry.findtext("atom:summary", default="", namespaces=ns)
                updated = entry.findtext("atom:updated", default="", namespaces=ns)
                link_el = entry.find("atom:link", ns)
                url = link_el.attrib.get("href", "") if link_el is not None else ""
                published = _article_date(updated)
                if published < from_date or published > to_date:
                    continue
                entries.append(
                    {
                        "title": title,
                        "description": summary,
                        "publishedAt": updated,
                        "source": "SEC EDGAR",
                        "url": url,
                    }
                )
            if not entries:
                return FetchOutcome(self.name, status="empty", error="SEC returned no recent filings")
            return FetchOutcome(self.name, data=entries)
        except Exception as exc:
            return FetchOutcome(self.name, status="error", error=str(exc))


class PressReleaseFetcher(GoogleNewsRSSFetcher):
    name = "press_releases"
    query_driven = True

    def supports(self, symbol: str) -> bool:
        return get_market(symbol) == "us" and not symbol.startswith("^")

    def fetch(self, symbol: str, query: str, from_date: date, to_date: date) -> FetchOutcome:
        pr_query = f'("{query}" "press release") OR ("{query}" "investor relations")'
        return super().fetch(symbol, pr_query, from_date, to_date)


class MultiSourceNewsFetcher:
    def __init__(self):
        registry = {
            "google_rss": GoogleNewsRSSFetcher(),
            "rss": RSSFetcher(),
            "nse_announcements": NSEAnnouncementsFetcher(),
            "bse_announcements": BSEAnnouncementsFetcher(),
            "sec_filings": SECFilingsFetcher(),
            "press_releases": PressReleaseFetcher(),
            "finnhub": FinnhubNewsFetcher(),
            "alpha_vantage": AlphaVantageNewsFetcher(),
            "newsapi": NewsAPIFetcher(),
            "gnews": GNewsFetcher(),
        }
        ordered_names = [name for name in NEWS_PROVIDER_ORDER if name in registry]
        for default_name in ["google_rss", "rss", "nse_announcements", "sec_filings", "press_releases"]:
            if default_name not in ordered_names:
                ordered_names.append(default_name)
        self.providers = [registry[name] for name in ordered_names]
        self._preferred_provider_by_symbol: Dict[str, str] = {}

    def _ordered_providers(self, symbol: str) -> List[object]:
        market = get_market(symbol)
        official_priority = ["google_rss", "rss"]
        if market == "nse":
            official_priority = ["nse_announcements", "google_rss", "rss", "bse_announcements"]
        elif market == "us":
            official_priority = ["sec_filings", "press_releases", "google_rss", "rss"]

        by_name = {provider.name: provider for provider in self.providers}
        ordered = [by_name[name] for name in official_priority if name in by_name]
        remaining = [provider for provider in stable_rotate(self.providers, symbol) if provider.name not in {p.name for p in ordered}]
        preferred = self._preferred_provider_by_symbol.get(symbol)
        if preferred:
            remaining = [p for p in remaining if p.name == preferred] + [p for p in remaining if p.name != preferred]
        return ordered + remaining

    def fetch(self, symbol: str, from_date: date, to_date: date) -> FetchOutcome:
        provider_order = self._ordered_providers(symbol)
        query_variants = _alias_queries(symbol)

        merged: Dict[str, Dict] = {}
        tried: List[str] = []
        errors: List[str] = []
        fallback_hits = 0
        attempts = 0

        for provider in provider_order:
            if not provider.supports(symbol):
                continue
            if attempts >= NEWS_MAX_PROVIDER_ATTEMPTS:
                break

            provider_queries = query_variants if getattr(provider, "query_driven", False) else [query_variants[0]]
            provider_success = False
            for query in provider_queries:
                if attempts >= NEWS_MAX_PROVIDER_ATTEMPTS:
                    break
                attempts += 1
                outcome = provider.fetch(symbol, query, from_date, to_date)
                tried.append(provider.name)
                if outcome.ok:
                    filtered_count = 0
                    for raw_article in outcome.data:
                        prepared = _finalize_article(raw_article, provider.name, symbol)
                        if prepared is None:
                            continue
                        filtered_count += 1
                        key = _dedupe_key(prepared)
                        current = merged.get(key)
                        if current is None or _article_rank(prepared) > _article_rank(current):
                            merged[key] = prepared
                    if filtered_count:
                        self._preferred_provider_by_symbol[symbol] = provider.name
                        provider_success = True
                        if provider.name not in {"google_rss", "rss", "nse_announcements", "sec_filings"}:
                            fallback_hits += 1
                        official_hits = sum(1 for article in merged.values() if article.get("official_event_hit", 0.0) > 0)
                        if len(merged) >= NEWS_MIN_ARTICLES or official_hits >= NEWS_OFFICIAL_HIT_TARGET:
                            articles = sorted(merged.values(), key=_article_rank, reverse=True)[:NEWS_MAX_ARTICLES]
                            return FetchOutcome(
                                provider.name,
                                data=articles,
                                meta={
                                    "providers_tried": tried,
                                    "fallback_hits": fallback_hits,
                                    "official_hits": official_hits,
                                    "query_variants": query_variants,
                                },
                            )
                        if not getattr(provider, "query_driven", False):
                            break
                elif outcome.error:
                    errors.append(f"{provider.name}: {outcome.error}")

            if provider_success and merged and len(merged) >= NEWS_MIN_ARTICLES:
                break

        if merged:
            articles = sorted(merged.values(), key=_article_rank, reverse=True)[:NEWS_MAX_ARTICLES]
            return FetchOutcome(
                tried[-1] if tried else "multi_source",
                data=articles,
                meta={
                    "providers_tried": tried,
                    "fallback_hits": fallback_hits,
                    "official_hits": sum(1 for article in articles if article.get("official_event_hit", 0.0) > 0),
                    "query_variants": query_variants,
                },
            )

        return FetchOutcome(
            "multi_source",
            status="error",
            error=" | ".join(errors[-5:]) or "No provider returned articles",
            meta={"providers_tried": tried, "query_variants": query_variants},
        )


class SentimentPipeline:
    def __init__(self):
        self.news = MultiSourceNewsFetcher()
        self.vader = _get_vader()

    def _score_articles(self, symbol: str, articles: List[Dict]) -> List[Dict]:
        scored = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '') or ''}"
            compound = float(score_text(text, self.vader))
            quality = float(article.get("source_quality_score", 0.0) or 0.0)
            official = float(article.get("is_official", 0.0) or 0.0)
            filing = float(article.get("is_filing", 0.0) or 0.0)
            press_release = float(article.get("is_press_release", 0.0) or 0.0)
            media_weight = max(0.10, 1.0 - official)
            weighted = compound * (0.55 + 0.45 * quality + 0.20 * official + 0.10 * filing)
            try:
                dt = _article_date(article.get("publishedAt", ""))
            except Exception:
                dt = date.today()
            lower_text = text.lower()
            is_earnings_call = float(any(keyword in lower_text for keyword in EARNINGS_TONE_KEYWORDS))
            new_risk_factors = float(any(keyword in lower_text for keyword in RISK_FACTOR_KEYWORDS))

            scored.append(
                {
                    "date": dt,
                    "title": article.get("title", ""),
                    "compound_score": compound,
                    "weighted_compound_score": weighted,
                    "media_sentiment": compound * media_weight,
                    "official_sentiment": compound * official,
                    "filing_sentiment": compound * filing,
                    "filing_change_score": 0.0,
                    "filing_fresh_language_score": 0.0,
                    "new_risk_factors": new_risk_factors,
                    "earnings_tone_signal": compound * is_earnings_call,
                    "earnings_call_count": is_earnings_call,
                    "article_count": 1,
                    "media_article_count": media_weight,
                    "official_article_count": official,
                    "filing_article_count": filing,
                    "press_release_count": press_release,
                    "official_event_hit": float(article.get("official_event_hit", 0.0) or 0.0),
                    "filing_event_hit": float(article.get("filing_event_hit", 0.0) or 0.0),
                    "source_quality_score": quality,
                    "source": article.get("source", "RSS"),
                    "provider": article.get("provider", ""),
                    "url": article.get("url", ""),
                }
            )
        filing_records = [item for item in scored if float(item.get("filing_article_count", 0.0) or 0.0) > 0]
        filing_records.sort(key=lambda item: (item.get("date"), item.get("title", "")))
        previous_filing_text = ""
        for record in filing_records:
            filing_text = f"{record.get('title', '')} {record.get('source', '')}"
            similarity = _jaccard_similarity(filing_text, previous_filing_text) if previous_filing_text else 1.0
            change_score = max(0.0, 1.0 - similarity)
            record["filing_change_score"] = round(change_score, 4)
            record["filing_fresh_language_score"] = round(change_score, 4)
            previous_filing_text = filing_text
        return scored

    def _fetch_symbol(self, symbol: str, from_date: date, to_date: date) -> FetchOutcome:
        outcome = self.news.fetch(symbol, from_date, to_date)
        if outcome.ok:
            outcome.data = self._score_articles(symbol, outcome.data)
        return outcome

    def run(self, symbols: List[str], days_back: int = 3, save: bool = False) -> Dict:
        logger.info(f"SentimentPipeline: {len(symbols)} symbols, last {days_back} days")

        del save
        to_date = date.today()
        from_date = to_date - timedelta(days=days_back)
        all_headlines: Dict[str, List[Dict]] = {}
        provider_by_symbol: Dict[str, str] = {}
        daily_records: List[Dict] = []
        provider_counter: Counter = Counter()
        official_symbols = 0
        fallback_symbols = 0
        symbols_with_articles = 0
        no_news_symbols: List[str] = []

        for idx, symbol in enumerate(symbols, start=1):
            try:
                outcome = self._fetch_symbol(symbol, from_date, to_date)
                articles = outcome.data if outcome.ok else []
                all_headlines[symbol] = articles
                provider_by_symbol[symbol] = outcome.provider
                provider_counter[outcome.provider] += 1

                if articles:
                    symbols_with_articles += 1
                    if any(article.get("official_event_hit", 0.0) > 0 for article in articles):
                        official_symbols += 1
                    if outcome.meta.get("fallback_hits"):
                        fallback_symbols += 1
                else:
                    no_news_symbols.append(symbol)

                for article in articles:
                    daily_records.append(
                        {
                            "symbol": symbol,
                            "date": article["date"],
                            "compound_score": article["compound_score"],
                            "weighted_compound_score": article["weighted_compound_score"],
                            "media_sentiment": article["media_sentiment"],
                            "official_sentiment": article["official_sentiment"],
                            "filing_sentiment": article["filing_sentiment"],
                            "filing_change_score": article["filing_change_score"],
                            "filing_fresh_language_score": article["filing_fresh_language_score"],
                            "new_risk_factors": article["new_risk_factors"],
                            "earnings_tone_signal": article["earnings_tone_signal"],
                            "earnings_call_count": article["earnings_call_count"],
                            "article_count": article["article_count"],
                            "media_article_count": article["media_article_count"],
                            "official_article_count": article["official_article_count"],
                            "filing_article_count": article["filing_article_count"],
                            "press_release_count": article["press_release_count"],
                            "official_event_hit": article["official_event_hit"],
                            "filing_event_hit": article["filing_event_hit"],
                            "source_quality_score": article["source_quality_score"],
                            "title": article["title"],
                            "source": article["source"],
                            "provider": article.get("provider", ""),
                        }
                    )

                if idx % NEWS_LOG_PROGRESS_EVERY == 0 or idx == len(symbols):
                    logger.info(
                        f"Sentiment progress {idx}/{len(symbols)} | with news {symbols_with_articles} | "
                        f"official {official_symbols} | no news {len(no_news_symbols)}"
                    )
            except Exception as exc:
                no_news_symbols.append(symbol)
                logger.error(f"Sentiment error for {symbol}: {exc}")

        if daily_records:
            frame = pd.DataFrame(daily_records)
            frame["date"] = pd.to_datetime(frame["date"])
            daily_agg = (
                frame.groupby(["symbol", "date"])
                .agg(
                    compound_score=("compound_score", "mean"),
                    weighted_compound_score=("weighted_compound_score", "mean"),
                    media_sentiment=("media_sentiment", "mean"),
                    official_sentiment=("official_sentiment", "mean"),
                    filing_sentiment=("filing_sentiment", "mean"),
                    filing_change_score=("filing_change_score", "max"),
                    filing_fresh_language_score=("filing_fresh_language_score", "max"),
                    new_risk_factors=("new_risk_factors", "max"),
                    earnings_tone_signal=("earnings_tone_signal", "mean"),
                    earnings_call_count=("earnings_call_count", "sum"),
                    article_count=("article_count", "sum"),
                    media_article_count=("media_article_count", "sum"),
                    official_article_count=("official_article_count", "sum"),
                    filing_article_count=("filing_article_count", "sum"),
                    press_release_count=("press_release_count", "sum"),
                    official_event_hit=("official_event_hit", "max"),
                    filing_event_hit=("filing_event_hit", "max"),
                    source_quality_score=("source_quality_score", "mean"),
                )
                .reset_index()
                .set_index(["symbol", "date"])
                .sort_index()
            )
        else:
            daily_agg = pd.DataFrame(
                columns=[
                    "compound_score",
                    "weighted_compound_score",
                    "media_sentiment",
                    "official_sentiment",
                    "filing_sentiment",
                    "filing_change_score",
                    "filing_fresh_language_score",
                    "new_risk_factors",
                    "earnings_tone_signal",
                    "earnings_call_count",
                    "article_count",
                    "media_article_count",
                    "official_article_count",
                    "filing_article_count",
                    "press_release_count",
                    "official_event_hit",
                    "filing_event_hit",
                    "source_quality_score",
                ],
                index=pd.MultiIndex.from_tuples([], names=["symbol", "date"]),
            )

        provider_summary = ", ".join(f"{name}={count}" for name, count in provider_counter.most_common())
        preview = ", ".join(no_news_symbols[:NEWS_FAILURE_PREVIEW])
        logger.info(
            "SentimentPipeline complete: "
            f"rows={daily_agg.shape[0]} | symbols_with_news={symbols_with_articles}/{len(symbols)} | "
            f"official_hits={official_symbols} | fallback_symbols={fallback_symbols} | "
            f"no_news={len(no_news_symbols)}"
        )
        if provider_summary:
            logger.info(f"Sentiment providers: {provider_summary}")
        if preview:
            logger.info(f"Sentiment no-news preview: {preview}")

        result = {
            "symbol_sentiment_daily": daily_agg,
            "headlines": all_headlines,
            "news_provider_by_symbol": provider_by_symbol,
            "news_summary": {
                "symbols_with_news": symbols_with_articles,
                "official_symbols": official_symbols,
                "fallback_symbols": fallback_symbols,
                "no_news_symbols": no_news_symbols,
                "provider_counts": dict(provider_counter),
            },
            "run_timestamp": datetime.utcnow().isoformat(),
            "symbols_covered": len(all_headlines),
        }
        return result
