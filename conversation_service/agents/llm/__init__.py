"""
Agents LLM - Phase 4 Components
Architecture v2.0

Ce module contient les agents IA utilisant les LLMs :
- LLMProviderManager : Abstraction multi-provider avec fallback
- IntentClassifier : Classification autonome few-shot
- ResponseGenerator : Streaming + insights automatiques
"""

from .llm_provider import (
    LLMProviderManager,
    BaseLLMProvider,
    DeepSeekProvider,
    OpenAIProvider,
    LocalProvider,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderType,
    ModelCapability
)

from .intent_classifier import (
    IntentClassifier,
    ClassificationRequest,
    ClassificationResult,
    ExtractedEntity,
    IntentConfidence
)

from .response_generator import (
    ResponseGenerator,
    ResponseGenerationRequest,
    ResponseGenerationResult,
    GeneratedInsight,
    ResponseType,
    InsightType
)

__all__ = [
    # Provider abstraction
    "LLMProviderManager",
    "BaseLLMProvider", 
    "DeepSeekProvider",
    "OpenAIProvider",
    "LocalProvider",
    "LLMRequest",
    "LLMResponse",
    "ProviderConfig",
    "ProviderType",
    "ModelCapability",
    
    # Intent classification
    "IntentClassifier",
    "ClassificationRequest",
    "ClassificationResult", 
    "ExtractedEntity",
    "IntentConfidence",
    
    # Response generation
    "ResponseGenerator",
    "ResponseGenerationRequest",
    "ResponseGenerationResult",
    "GeneratedInsight",
    "ResponseType",
    "InsightType"
]