"""
Enhanced Sentiment Features
=============================
Combines VADER (free) + Finnhub sentiment + news sentiment
for comprehensive sentiment scoring.

This module enhances the existing sentiment features with:
- Finnhub social sentiment (Reddit, Twitter)
- News sentiment scores
- Composite sentiment with weighting
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FINNHUB_ENABLED = os.getenv("FINNHUB_API_KEYS") or os.getenv("FINNHUB_API_KEY")


class EnhancedSentimentEngine:
    """
    Combines multiple sentiment sources:
    1. VADER (free, fast, rule-based)
    2. Finnhub social sentiment (free tier)
    3. News sentiment (from pipeline)
    
    Output:
    - compound_score (weighted average)
    - sentiment_zscore (normalized)
    - sentiment_velocity (trend)
    - source_quality_score (reliability)
    """
    
    SOURCE_WEIGHTS = {
        "vader_news": 0.25,
        "vader_filings": 0.20,
        "finnhub_reddit": 0.20,
        "finnhub_twitter": 0.15,
        "finnhub_news": 0.20,
    }
    
    def __init__(self):
        self._vader = None
        self._finnhub_client = None
        self._init_vader()
        self._init_finnhub()
        
    def _init_vader(self):
        """Initialize VADER sentiment analyzer."""
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment initialized")
        except Exception as e:
            logger.warning(f"VADER initialization failed: {e}")
    
    def _init_finnhub(self):
        """Initialize Finnhub client."""
        if FINNHUB_ENABLED:
            try:
                from pipeline.finnhub_client import FinnhubClient
                self._finnhub_client = FinnhubClient()
                logger.info("Finnhub sentiment client initialized")
            except Exception as e:
                logger.warning(f"Finnhub client initialization failed: {e}")
    
    def score_text_vader(self, text: str) -> float:
        """Score text with VADER."""
        if not self._vader:
            return 0.0
        try:
            scores = self._vader.polarity_scores(str(text))
            return float(scores["compound"])
        except Exception as e:
            logger.debug(f"VADER scoring error: {e}")
            return 0.0
    
    def get_finnhub_sentiment(self, symbol: str) -> Dict:
        """Get Finnhub social sentiment for symbol."""
        if not self._finnhub_client:
            return {"reddit": 0.0, "twitter": 0.0, "reddit_vol": 0, "twitter_vol": 0}
        
        try:
            data = self._finnhub_client.get_sentiment(symbol)
            return {
                "reddit": data.get("reddit_sentiment", 0.0),
                "twitter": data.get("twitter_sentiment", 0.0),
                "reddit_vol": data.get("reddit_mention_count", 0),
                "twitter_vol": data.get("twitter_mention_count", 0),
            }
        except Exception as e:
            logger.debug(f"Finnhub sentiment error for {symbol}: {e}")
            return {"reddit": 0.0, "twitter": 0.0, "reddit_vol": 0, "twitter_vol": 0}
    
    def get_finnhub_news_sentiment(self, symbol: str, limit: int = 10) -> float:
        """Get aggregated sentiment from Finnhub news."""
        if not self._finnhub_client:
            return 0.0
        
        try:
            news = self._finnhub_client.get_news(symbol, category="general")
            if not news:
                return 0.0
            
            sentiments = []
            for item in news[:limit]:
                text = item.get("headline", "") + " " + item.get("summary", "")
                sentiment = self.score_text_vader(text)
                sentiments.append(sentiment)
            
            return np.mean(sentiments) if sentiments else 0.0
        except Exception as e:
            logger.debug(f"Finnhub news sentiment error: {e}")
            return 0.0
    
    def compute_composite_sentiment(
        self,
        news_texts: List[str],
        filing_texts: List[str],
        symbol: str,
    ) -> Dict:
        """
        Compute composite sentiment from all sources.
        
        Returns:
            {
                "compound_score": 0.65,
                "sentiment_zscore": 1.2,
                "sentiment_velocity": 0.3,
                "reddit_sentiment": 0.55,
                "twitter_sentiment": 0.42,
                "source_quality_score": 0.85,
                "news_count": 15,
                "filing_count": 3
            }
        """
        news_sentiment = 0.0
        if news_texts:
            news_scores = [self.score_text_vader(text) for text in news_texts]
            news_sentiment = np.mean(news_scores)
        
        filing_sentiment = 0.0
        if filing_texts:
            filing_scores = [self.score_text_vader(text) for text in filing_texts]
            filing_sentiment = np.mean(filing_scores)
        
        finnhub_social = self.get_finnhub_sentiment(symbol)
        finnhub_news = self.get_finnhub_news_sentiment(symbol)
        
        weighted_score = (
            self.SOURCE_WEIGHTS["vader_news"] * news_sentiment +
            self.SOURCE_WEIGHTS["vader_filings"] * filing_sentiment +
            self.SOURCE_WEIGHTS["finnhub_reddit"] * finnhub_social.get("reddit", 0.0) +
            self.SOURCE_WEIGHTS["finnhub_twitter"] * finnhub_social.get("twitter", 0.0) +
            self.SOURCE_WEIGHTS["finnhub_news"] * finnhub_news
        )
        
        source_count = sum([
            len(news_texts) > 0,
            len(filing_texts) > 0,
            finnhub_social.get("reddit_vol", 0) > 0,
            finnhub_social.get("twitter_vol", 0) > 0,
            finnhub_news != 0.0,
        ])
        quality_score = min(1.0, source_count / 3.0)
        
        return {
            "compound_score": round(float(weighted_score), 4),
            "sentiment_zscore": round(float(weighted_score / max(abs(weighted_score) + 0.01, 0.01)), 4),
            "sentiment_velocity": 0.0,
            "reddit_sentiment": round(float(finnhub_social.get("reddit", 0.0)), 4),
            "twitter_sentiment": round(float(finnhub_social.get("twitter", 0.0)), 4),
            "reddit_volume": finnhub_social.get("reddit_vol", 0),
            "twitter_volume": finnhub_social.get("twitter_vol", 0),
            "source_quality_score": round(float(quality_score), 4),
            "news_count": len(news_texts),
            "filing_count": len(filing_texts),
            "news_sentiment": round(float(news_sentiment), 4),
            "filing_sentiment": round(float(filing_sentiment), 4),
            "finnhub_news_sentiment": round(float(finnhub_news), 4),
        }
    
    def update_sentiment_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        news_texts_col: Optional[str] = None,
        filing_texts_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Update DataFrame with enhanced sentiment features.
        """
        df = df.copy()
        
        news_texts = []
        filing_texts = []
        
        if news_texts_col and news_texts_col in df.columns:
            news_texts = df[news_texts_col].dropna().astype(str).tolist()
        
        if filing_texts_col and filing_texts_col in df.columns:
            filing_texts = df[filing_texts_col].dropna().astype(str).tolist()
        
        sentiment = self.compute_composite_sentiment(news_texts, filing_texts, symbol)
        
        df["compound_score"] = sentiment["compound_score"]
        df["sentiment_zscore"] = sentiment["sentiment_zscore"]
        df["sentiment_velocity"] = sentiment["sentiment_velocity"]
        df["source_quality_score"] = sentiment["source_quality_score"]
        df["reddit_sentiment"] = sentiment["reddit_sentiment"]
        df["twitter_sentiment"] = sentiment["twitter_sentiment"]
        df["reddit_volume"] = sentiment["reddit_volume"]
        df["twitter_volume"] = sentiment["twitter_volume"]
        
        return df


def get_sentiment_engine() -> EnhancedSentimentEngine:
    """Factory function to get sentiment engine."""
    return EnhancedSentimentEngine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = get_sentiment_engine()
    
    test_news = [
        "AAPL reports record quarterly earnings, beats estimates",
        "Strong iPhone sales drive revenue growth",
        "Analysts upgrade to buy rating",
    ]
    
    test_filings = [
        "SEC filing shows increased buyback authorization",
    ]
    
    result = engine.compute_composite_sentiment(test_news, test_filings, "AAPL")
    print(f"Composite Sentiment: {result}")
    
    print("\nEnhanced sentiment engine initialized successfully.")