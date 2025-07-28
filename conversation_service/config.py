"""
⚙️ Configuration Centralisée - Service Détection d'Intention

Configuration optimisée pour performance et coûts avec DeepSeek fallback.
Tous les paramètres tunables centralisés ici.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeepSeekConfig:
    """Configuration client DeepSeek pour fallback LLM"""
    api_key: str = "sk-6923dd2c9f674a10b78665f3e01f9193"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout: float = 6.0
    max_retries: int = 2
    temperature: float = 0.05
    max_tokens: int = 100
    
    # Coût par token (estimation)
    cost_per_1k_tokens: float = 0.00014


@dataclass(frozen=True)  
class PerformanceConfig:
    """Configuration cibles de performance"""
    target_latency_ms: float = 50.0
    target_accuracy: float = 0.85
    cache_max_size: int = 200
    cache_ttl_seconds: int = 3600
    
    # Seuils confiance pour décisions
    high_confidence_threshold: float = 0.85
    deepseek_threshold: float = 0.3
    cache_threshold: float = 0.6


@dataclass(frozen=True)
class RuleEngineConfig:
    """Configuration moteur de règles intelligent"""
    enable_rules: bool = True
    enable_cache: bool = True
    enable_entity_extraction: bool = True
    
    # Boost confiance par type de match
    exact_match_boost: float = 0.9
    partial_match_boost: float = 0.7
    multi_pattern_boost: float = 1.2
    
    # Seuil minimum pour considérer un match
    min_match_threshold: float = 0.1


@dataclass(frozen=True)
class ServiceConfig:
    """Configuration générale du service"""
    service_name: str = "conversation-service"
    version: str = "2.0.0"
    description: str = "Service détection intention ultra-optimisé"
    
    # API Configuration
    cors_origins: list = field(default_factory=lambda: ["*"])
    enable_docs: bool = True
    log_level: str = "INFO"
    
    # Feature flags
    enable_deepseek_fallback: bool = True
    enable_batch_processing: bool = True
    enable_metrics_collection: bool = True


@dataclass(frozen=True)
class IntentPatternsConfig:
    """Configuration patterns par intention"""
    
    # Intentions supportées avec métadonnées
    supported_intents: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "ACCOUNT_BALANCE": {
            "priority": 1,
            "confidence_base": 0.9,
            "requires_entities": False,
            "common_entities": ["account_type", "month"]
        },
        "SEARCH_BY_CATEGORY": {
            "priority": 2, 
            "confidence_base": 0.85,
            "requires_entities": True,
            "common_entities": ["category", "period", "month"]
        },
        "BUDGET_ANALYSIS": {
            "priority": 3,
            "confidence_base": 0.8,
            "requires_entities": False,
            "common_entities": ["amount", "period", "month"]
        },
        "TRANSFER": {
            "priority": 1,
            "confidence_base": 0.9,
            "requires_entities": True,
            "common_entities": ["amount", "recipient"]
        },
        "SEARCH_BY_DATE": {
            "priority": 4,
            "confidence_base": 0.75,
            "requires_entities": True,
            "common_entities": ["month", "period", "date"]
        },
        "CARD_MANAGEMENT": {
            "priority": 1,
            "confidence_base": 0.95,
            "requires_entities": False,
            "common_entities": ["card_type", "amount"]
        },
        "GREETING": {
            "priority": 5,
            "confidence_base": 0.95,
            "requires_entities": False,
            "common_entities": []
        },
        "HELP": {
            "priority": 5,
            "confidence_base": 0.8,
            "requires_entities": False,
            "common_entities": []
        },
        "GOODBYE": {
            "priority": 5,
            "confidence_base": 0.95,
            "requires_entities": False,
            "common_entities": []
        },
        "UNKNOWN": {
            "priority": 10,
            "confidence_base": 0.0,
            "requires_entities": False,
            "common_entities": []
        }
    })


class Config:
    """Configuration principale - Point d'accès unique"""
    
    def __init__(self):
        self.deepseek = DeepSeekConfig()
        self.performance = PerformanceConfig()
        self.rule_engine = RuleEngineConfig()
        self.service = ServiceConfig()
        self.intent_patterns = IntentPatternsConfig()
        
        # Override avec variables d'environnement si disponibles
        self._load_env_overrides()
    
    def _load_env_overrides(self):
        """Charge les overrides depuis variables d'environnement"""
        
        # DeepSeek config overrides
        if api_key := os.getenv("DEEPSEEK_API_KEY"):
            object.__setattr__(self.deepseek, 'api_key', api_key)
        
        if base_url := os.getenv("DEEPSEEK_BASE_URL"):
            object.__setattr__(self.deepseek, 'base_url', base_url)
            
        if model := os.getenv("DEEPSEEK_MODEL"):
            object.__setattr__(self.deepseek, 'model', model)
        
        # Performance config overrides
        if target_latency := os.getenv("TARGET_LATENCY_MS"):
            try:
                object.__setattr__(self.performance, 'target_latency_ms', float(target_latency))
            except ValueError:
                pass
        
        if target_accuracy := os.getenv("TARGET_ACCURACY"):
            try:
                object.__setattr__(self.performance, 'target_accuracy', float(target_accuracy))
            except ValueError:
                pass
        
        # Service config overrides
        if log_level := os.getenv("LOG_LEVEL"):
            object.__setattr__(self.service, 'log_level', log_level)
        
        if enable_deepseek := os.getenv("ENABLE_DEEPSEEK_FALLBACK"):
            if enable_deepseek.lower() in ('false', '0', 'no'):
                object.__setattr__(self.service, 'enable_deepseek_fallback', False)
    
    def get_intent_search_code(self, intent: str) -> str:
        """Mapping intention vers code search service"""
        intent_to_search_code = {
            "ACCOUNT_BALANCE": "ACCOUNT_BALANCE",
            "SEARCH_BY_CATEGORY": "SEARCH_BY_CATEGORY", 
            "BUDGET_ANALYSIS": "BUDGET_ANALYSIS",
            "TRANSFER": "TRANSFER",
            "SEARCH_BY_DATE": "SEARCH_BY_DATE",
            "CARD_MANAGEMENT": "CARD_MANAGEMENT",
            "GREETING": "GREETING",
            "HELP": "HELP",
            "GOODBYE": "GOODBYE",
            "UNKNOWN": "UNKNOWN"
        }
        return intent_to_search_code.get(intent, "UNKNOWN")
    
    def is_high_confidence(self, confidence: float) -> bool:
        """Vérifie si confiance est élevée"""
        return confidence >= self.performance.high_confidence_threshold
    
    def should_use_deepseek(self, confidence: float) -> bool:
        """Détermine si fallback DeepSeek nécessaire"""
        return (
            self.service.enable_deepseek_fallback and 
            confidence < self.performance.deepseek_threshold
        )
    
    def should_cache_result(self, confidence: float) -> bool:
        """Détermine si résultat doit être mis en cache"""
        return (
            self.rule_engine.enable_cache and
            confidence >= self.performance.cache_threshold
        )
    
    def get_intent_metadata(self, intent: str) -> Dict[str, Any]:
        """Récupère métadonnées d'une intention"""
        return self.intent_patterns.supported_intents.get(
            intent, 
            self.intent_patterns.supported_intents["UNKNOWN"]
        )


# Instance globale de configuration
config = Config()


# Fonctions utilitaires d'accès rapide
def get_deepseek_config() -> DeepSeekConfig:
    """Accès rapide config DeepSeek"""
    return config.deepseek


def get_performance_config() -> PerformanceConfig:
    """Accès rapide config performance"""
    return config.performance


def get_supported_intents() -> Dict[str, Dict[str, Any]]:
    """Accès rapide intentions supportées"""
    return config.intent_patterns.supported_intents


def is_intent_supported(intent: str) -> bool:
    """Vérifie si intention supportée"""
    return intent in config.intent_patterns.supported_intents


# Configuration par défaut exportée
__all__ = [
    "Config",
    "config", 
    "DeepSeekConfig",
    "PerformanceConfig",
    "RuleEngineConfig", 
    "ServiceConfig",
    "IntentPatternsConfig",
    "get_deepseek_config",
    "get_performance_config", 
    "get_supported_intents",
    "is_intent_supported"
]