"""
🎯 Conversation Service - Architecture Minimaliste TinyBERT

Service ultra-simplifié pour détection intentions financières
avec un seul endpoint et mesure précise de latence.

Architecture: Input → TinyBERT → Intent → Response
Performance: < 50ms, > 70% précision

Version: 1.0.0
"""

from .main import app
from .intent_detector import TinyBERTDetector
from .models import IntentRequest, IntentResponse, HealthResponse

__version__ = "1.0.0"
__all__ = ["app", "TinyBERTDetector", "IntentRequest", "IntentResponse", "HealthResponse"]