"""
NVIDIA NIM Client
=================
Connects to NVIDIA API Catalog for AI inference.
Uses MiniMax-M2.5 model (specialized for agentic tasks).

Documentation: https://docs.api.nvidia.com/nim/
Model: https://build.nvidia.com/MiniMaxAI/minimax-m2.5

Features:
- Sentiment analysis from news/earnings
- Trade thesis validation
- Event classification
- Risk assessment
- Caching for cost optimization (24h TTL)
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", os.getenv("NVAPI_KEY", ""))
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NIMClient:
    """
    Client for NVIDIA NIM (Neural Inference Microservice) API.
    
    Usage:
        client = NIMClient()
        
        # Sentiment analysis
        sentiment = client.analyze_sentiment("AAPL reports strong earnings")
        
        # Trade thesis validation
        validation = client.validate_thesis(
            symbol="AAPL",
            thesis="Strong earnings beat, momentum bullish",
            features={"rsi": 45, "macd": "bullish", "trend": "up"}
        )
        
        # Event classification
        event_type = client.classify_event("AAPL announces $100B buyback")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "minimaxai/minimax-m2.5",
        cache_ttl: int = 86400,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.api_key = api_key or NVIDIA_API_KEY
        self.model = model
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.timeout = timeout
        self._cache: Dict[str, Any] = {}
        self._cache_times: Dict[str, datetime] = {}
        
        if not self.api_key:
            logger.warning("NVIDIA API key not set. Get key from https://build.nvidia.com/")
            self._enabled = False
        else:
            self._enabled = True
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        cache_time = self._cache_times.get(key)
        if cache_time is None:
            return False
        return (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl
    
    def _get_cached(self, key: str) -> Optional[Any]:
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None
    
    def _set_cache(self, key: str, value: Any):
        self._cache[key] = value
        self._cache_times[key] = datetime.utcnow()
    
    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> Optional[str]:
        """Make chat completion request to NVIDIA NIM."""
        if not self._enabled:
            return None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{NVIDIA_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                logger.warning(f"NIM API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        return None
    
    def analyze_sentiment(self, text: str, cache: bool = True) -> Dict:
        """
        Analyze sentiment of text (news headline, earnings call, etc.)
        
        Returns:
            {
                "sentiment": "bullish|bearish|neutral",
                "confidence": 0.85,
                "key_points": ["positive point 1", "positive point 2"],
                "risk_factors": ["risk 1", "risk 2"]
            }
        """
        if cache:
            cache_key = f"sentiment:{text[:100]}"
            cached = self._get_cached(cache_key)
            if cached:
                return cached
        
        prompt = f"""Analyze the sentiment of this financial text. 
Respond in JSON format with:
- sentiment: "bullish", "bearish", or "neutral"
- confidence: 0.0-1.0 score
- key_points: array of 2-3 key bullish or bearish points
- risk_factors: array of potential risks mentioned

Text: {text}

JSON:"""

        result = self._chat_completion([
            {"role": "system", "content": "You are a financial analyst. Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ])
        
        if not result:
            return {"sentiment": "neutral", "confidence": 0.0, "error": "API failed"}
        
        try:
            parsed = json.loads(result.strip("```json").strip("```"))
            if cache:
                self._set_cache(cache_key, parsed)
            return parsed
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse NIM sentiment response: {result[:200]}")
            return {"sentiment": "neutral", "confidence": 0.0, "raw": result}
    
    def validate_thesis(
        self,
        symbol: str,
        thesis: str,
        features: Dict,
        cache: bool = True,
    ) -> Dict:
        """
        Validate a trade thesis using LLM reasoning.
        
        Args:
            symbol: Stock symbol
            thesis: Trade thesis statement
            features: Dict of technical features (RSI, MACD, trend, etc.)
        
        Returns:
            {
                "valid": true/false,
                "reasoning": "explanation",
                "confidence": 0.75,
                "suggested_direction": "buy/sell/hold",
                "risk_assessment": "low/medium/high"
            }
        """
        if cache:
            cache_key = f"thesis:{symbol}:{thesis[:50]}"
            cached = self._get_cached(cache_key)
            if cached:
                return cached
        
        feature_str = json.dumps(features, indent=2)
        
        prompt = f"""Validate this trade thesis for {symbol}.

Thesis: {thesis}

Technical Features:
{feature_str}

Respond in JSON format with:
- valid: boolean (is thesis supported by data?)
- reasoning: string (explain your decision)
- confidence: 0.0-1.0
- suggested_direction: "buy", "sell", or "hold"
- risk_assessment: "low", "medium", or "high"

JSON:"""

        result = self._chat_completion([
            {"role": "system", "content": "You are an expert trading analyst. Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ], temperature=0.2, max_tokens=400)
        
        if not result:
            return {"valid": False, "confidence": 0.0, "error": "API failed"}
        
        try:
            parsed = json.loads(result.strip("```json").strip("```"))
            if cache:
                self._set_cache(cache_key, parsed)
            return parsed
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse NIM thesis response: {result[:200]}")
            return {"valid": False, "confidence": 0.0, "raw": result}
    
    def classify_event(
        self,
        event_text: str,
        cache: bool = True,
    ) -> Dict:
        """
        Classify financial event type.
        
        Returns:
            {
                "event_type": "earnings|buyback|dividend|merger|regulatory|earnings_guidance|other",
                "impact": "positive|negative|mixed",
                "confidence": 0.80,
                "description": "brief description"
            }
        """
        if cache:
            cache_key = f"event:{event_text[:80]}"
            cached = self._get_cached(cache_key)
            if cached:
                return cached
        
        prompt = f"""Classify this financial event.

Event: {event_text}

Respond in JSON format with:
- event_type: one of "earnings", "buyback", "dividend", "merger", "regulatory", "earnings_guidance", "product", "legal", "other"
- impact: "positive", "negative", or "mixed"
- confidence: 0.0-1.0
- description: 1-2 sentence description

JSON:"""

        result = self._chat_completion([
            {"role": "system", "content": "You are a financial event classifier. Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ], temperature=0.2, max_tokens=300)
        
        if not result:
            return {"event_type": "other", "confidence": 0.0, "error": "API failed"}
        
        try:
            parsed = json.loads(result.strip("```json").strip("```"))
            if cache:
                self._set_cache(cache_key, parsed)
            return parsed
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse NIM event response: {result[:200]}")
            return {"event_type": "other", "confidence": 0.0, "raw": result}
    
    def assess_risk(
        self,
        symbol: str,
        position_type: str,
        features: Dict,
        market_conditions: Optional[Dict] = None,
    ) -> Dict:
        """
        Assess risk of a potential trade.
        
        Returns:
            {
                "risk_score": 0.65,
                "risk_level": "medium",
                "factors": ["factor1", "factor2"],
                "mitigations": ["mitigation1"],
                "recommendation": "proceed_with_caution"
            }
        """
        cache_key = f"risk:{symbol}:{position_type}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        feature_str = json.dumps(features, indent=2)
        market_str = json.dumps(market_conditions or {}, indent=2)
        
        prompt = f"""Assess the risk of this potential trade.

Symbol: {symbol}
Position Type: {position_type}

Technical Features:
{feature_str}

Market Conditions:
{market_str}

Respond in JSON format with:
- risk_score: 0.0-1.0 (higher = more risky)
- risk_level: "low", "medium", "high", or "very_high"
- factors: array of risk factors
- mitigations: array of suggested risk mitigations
- recommendation: "proceed", "proceed_with_caution", "avoid", or "wait"

JSON:"""

        result = self._chat_completion([
            {"role": "system", "content": "You are a risk assessment expert. Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ], temperature=0.2, max_tokens=400)
        
        if not result:
            return {"risk_score": 0.5, "risk_level": "medium", "error": "API failed"}
        
        try:
            parsed = json.loads(result.strip("```json").strip("```"))
            self._set_cache(cache_key, parsed)
            return parsed
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse NIM risk response: {result[:200]}")
            return {"risk_score": 0.5, "risk_level": "medium", "raw": result}
    
    def generate_trade_reasoning(
        self,
        signal: Dict,
        features: Dict,
    ) -> str:
        """
        Generate human-readable trade reasoning.
        
        Returns:
            "Buy AAPL - Strong momentum with RSI oversold at 32..."
        """
        prompt = f"""Generate a brief trading rationale for this signal.

Signal: {json.dumps(signal, indent=2)}

Features: {json.dumps(features, indent=2)}

Generate a 2-3 sentence trading rationale in this format:
"{{direction}} {{symbol}} - {{key_reason}}"

Direction should be BUY or SELL.
Example: "BUY AAPL - RSI oversold at 32 with bullish MACD crossover suggesting momentum reversal."

Output:"""

        result = self._chat_completion([
            {"role": "system", "content": "You are a trading assistant. Generate clear, concise reasoning."},
            {"role": "user", "content": prompt},
        ], temperature=0.3, max_tokens=150)
        
        return result or "Signal generated - manual review recommended"
    
    def batch_analyze(self, texts: List[str], task: str = "sentiment") -> List[Dict]:
        """
        Batch process multiple texts.
        
        Note: NIM doesn't support true batching, so we process sequentially
        with caching to optimize.
        """
        results = []
        for text in texts:
            if task == "sentiment":
                result = self.analyze_sentiment(text)
            elif task == "event":
                result = self.classify_event(text)
            else:
                result = {"error": f"Unknown task: {task}"}
            results.append(result)
        return results
    
    def clear_cache(self):
        """Clear all cached responses."""
        self._cache.clear()
        self._cache_times.clear()
        logger.info("NIM cache cleared")


def get_nim_client() -> NIMClient:
    """Factory function to get NIM client."""
    return NIMClient()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = NIMClient()
    
    if client._enabled:
        print("Testing NVIDIA NIM API...")
        
        sentiment = client.analyze_sentiment("AAPL beats earnings expectations by 15%")
        print(f"\nSentiment: {sentiment}")
        
        thesis_validation = client.validate_thesis(
            symbol="AAPL",
            thesis="Strong earnings beat, expect continued momentum",
            features={"rsi": 45, "macd": "bullish", "trend": "up", "volume": "high"}
        )
        print(f"\nThesis Validation: {thesis_validation}")
        
        event = client.classify_event("AAPL announces $100B stock buyback program")
        print(f"\nEvent Classification: {event}")
        
    else:
        print("NVIDIA NIM not configured. Add NVIDIA_API_KEY to .env")