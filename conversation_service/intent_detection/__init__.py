"""
Intent Detection Engine - Système hybride de détection d'intention
Architecture 3 niveaux : Pattern Cache (L0) → TinyBERT (L1) → DeepSeek (L2)
"""

from .engine import IntentDetectionEngine
from .models import IntentResult, IntentLevel, IntentConfidence
from .cache_manager import IntentCacheManager
from .pattern_matcher import PatternMatcher
from .lightweight_classifier import LightweightClassifier
from .llm_fallback import LLMFallback

__version__ = "1.0.0"
__all__ = [
    "IntentDetectionEngine",
    "IntentResult", 
    "IntentLevel",
    "IntentConfidence",
    "IntentCacheManager",
    "PatternMatcher",
    "LightweightClassifier", 
    "LLMFallback"
]
