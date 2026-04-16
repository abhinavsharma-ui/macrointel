"""
LLM Thesis Validator
====================
Validates trade thesis using NVIDIA NIM LLM.
Adds quality check layer before trade execution.

Usage:
    validator = ThesisValidator()
    result = validator.validate(
        symbol="AAPL",
        direction="buy",
        thesis="RSI oversold at 32, MACD bullish crossover, strong earnings",
        features={"rsi": 32, "macd": "bullish", "earnings": "beat"}
    )
"""

import logging
import os
from typing import Dict, Optional
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", os.getenv("NVAPI_KEY", ""))


class ThesisValidator:
    """
    Uses LLM to validate trade thesis quality.
    
    Filters out:
    - Weak thesis (no clear rationale)
    - Contradictory signals (e.g., bear news + buy signal)
    - Low confidence setups
    """
    
    def __init__(self, cache_ttl: int = 3600):
        self.cache_ttl = cache_ttl
        self._enabled = bool(NVIDIA_API_KEY and NVIDIA_API_KEY != "YOUR_NVIDIA_API_KEY_HERE")
        self._cache: Dict[str, Dict] = {}
        self._cache_times: Dict[str, datetime] = {}
        
        if self._enabled:
            try:
                from pipeline.nim_client import NIMClient
                self._nim_client = NIMClient(cache_ttl=cache_ttl)
                logger.info("ThesisValidator: NIM client initialized")
            except Exception as e:
                logger.warning(f"ThesisValidator: NIM init failed: {e}")
                self._enabled = False
        else:
            logger.warning("ThesisValidator: NVIDIA API not configured")
    
    def _get_cache_key(self, symbol: str, direction: str, thesis: str) -> str:
        return f"{symbol}:{direction}:{thesis[:50]}"
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        cache_time = self._cache_times.get(key)
        if cache_time is None:
            return False
        return (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl
    
    def validate(
        self,
        symbol: str,
        direction: str,
        thesis: str,
        features: Dict,
    ) -> Dict:
        """
        Validate trade thesis.
        
        Returns:
            {
                "valid": true/false,
                "reasoning": "explanation",
                "confidence": 0.75,
                "risk_level": "low/medium/high",
                "warnings": ["warning1"]
            }
        """
        cache_key = self._get_cache_key(symbol, direction, thesis)
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        if not self._enabled:
            return {
                "valid": True,
                "reasoning": "LLM validation disabled",
                "confidence": 0.5,
                "risk_level": "medium",
                "warnings": ["LLM validation not available"],
                "source": "fallback",
            }
        
        try:
            feature_summary = self._format_features(features)
            
            prompt = f"""Validate this trade thesis for {symbol}.

Direction: {direction.upper()}
Thesis: {thesis}

Technical Features:
{feature_summary}

Respond in JSON format with:
- valid: boolean (is thesis supported by features?)
- reasoning: 1-2 sentence explanation
- confidence: 0.0-1.0 score
- risk_level: "low", "medium", or "high"
- warnings: array of concerns (if any)

JSON:"""

            result = self._nim_client._chat_completion([
                {"role": "system", "content": "You are an expert trading analyst. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ], temperature=0.2, max_tokens=400)
            
            if not result:
                return self._fallback_result()
            
            import json
            parsed = json.loads(result.strip("```json").strip("```"))
            parsed["source"] = "llm"
            
            self._cache[cache_key] = parsed
            self._cache_times[cache_key] = datetime.utcnow()
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Thesis validation failed for {symbol}: {e}")
            return self._fallback_result()
    
    def _format_features(self, features: Dict) -> str:
        """Format features for LLM prompt."""
        lines = []
        for k, v in features.items():
            lines.append(f"  - {k}: {v}")
        return "\n".join(lines) if lines else "  (none)"
    
    def _fallback_result(self) -> Dict:
        """Fallback when LLM is unavailable."""
        return {
            "valid": True,
            "reasoning": "Validation unavailable, defaulting to take",
            "confidence": 0.4,
            "risk_level": "medium",
            "warnings": ["LLM validation failed"],
            "source": "fallback",
        }
    
    def batch_validate(
        self,
        signals: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        Validate multiple trade signals.
        
        Args:
            signals: Dict of {symbol: signal_data}
        
        Returns:
            Dict of {symbol: validation_result}
        """
        results = {}
        for symbol, signal in signals.items():
            if signal.get("signal") == "neutral":
                results[symbol] = {
                    "valid": False,
                    "reasoning": "Neutral signal, no validation needed",
                    "confidence": 0.0,
                    "risk_level": "low",
                    "warnings": [],
                    "source": "neutral",
                }
                continue
            
            result = self.validate(
                symbol=symbol,
                direction=signal.get("signal", "hold"),
                thesis=signal.get("reasoning", ""),
                features=signal.get("factor_scores", {}),
            )
            results[symbol] = result
        
        return results
    
    def clear_cache(self):
        """Clear validation cache."""
        self._cache.clear()
        self._cache_times.clear()
        logger.info("ThesisValidator cache cleared")


def get_thesis_validator() -> ThesisValidator:
    """Factory function to get thesis validator."""
    return ThesisValidator()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    validator = get_thesis_validator()
    
    test_signal = {
        "signal": "buy",
        "reasoning": "RSI oversold at 32, MACD bullish crossover, strong earnings beat",
        "factor_scores": {
            "rsi_14": 32,
            "macd_hist": 0.5,
            "momentum": 0.8,
            "sentiment": 0.6,
        },
    }
    
    result = validator.validate(
        symbol="AAPL",
        direction="buy",
        thesis=test_signal["reasoning"],
        features=test_signal["factor_scores"],
    )
    
    print(f"Validation Result: {result}")