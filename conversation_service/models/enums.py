"""
📝 Énumérations - Types et constantes du service

Définit tous les types énumérés utilisés dans le service de détection d'intention.
"""

from enum import Enum, IntEnum
from typing import Dict, List


class IntentType(str, Enum):
    """Types d'intentions financières supportées"""
    
    # Intentions financières principales
    ACCOUNT_BALANCE = "ACCOUNT_BALANCE"
    SEARCH_BY_CATEGORY = "SEARCH_BY_CATEGORY"
    BUDGET_ANALYSIS = "BUDGET_ANALYSIS"
    TRANSFER = "TRANSFER"
    SEARCH_BY_DATE = "SEARCH_BY_DATE"
    CARD_MANAGEMENT = "CARD_MANAGEMENT"
    
    # Intentions conversationnelles
    GREETING = "GREETING"
    HELP = "HELP"
    GOODBYE = "GOODBYE"
    
    # Intention par défaut
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def get_financial_intents(cls) -> List[str]:
        """Retourne uniquement les intentions financières"""
        return [
            cls.ACCOUNT_BALANCE,
            cls.SEARCH_BY_CATEGORY,
            cls.BUDGET_ANALYSIS,
            cls.TRANSFER,
            cls.SEARCH_BY_DATE,
            cls.CARD_MANAGEMENT
        ]
    
    @classmethod
    def get_conversational_intents(cls) -> List[str]:
        """Retourne uniquement les intentions conversationnelles"""
        return [cls.GREETING, cls.HELP, cls.GOODBYE]
    
    @classmethod
    def is_financial(cls, intent: str) -> bool:
        """Vérifie si intention est financière"""
        return intent in cls.get_financial_intents()


class EntityType(str, Enum):
    """Types d'entités extraites"""
    
    # Entités financières
    AMOUNT = "amount"
    ACCOUNT_TYPE = "account_type"
    CATEGORY = "category"
    MERCHANT = "merchant"
    
    # Entités temporelles
    DATE = "date"
    MONTH = "month"
    PERIOD = "period"
    
    # Entités de transaction
    RECIPIENT = "recipient"
    CARD_TYPE = "card_type"
    
    # Entités contextuelles
    CURRENCY = "currency"
    LOCATION = "location"


class ConfidenceLevel(str, Enum):
    """Niveaux de confiance pour classification"""
    
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.5  
    MEDIUM = "medium"          # 0.5 - 0.7
    HIGH = "high"              # 0.7 - 0.9
    VERY_HIGH = "very_high"    # 0.9 - 1.0
    
    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Convertit score numérique en niveau confiance"""
        if score < 0.2:
            return cls.VERY_LOW
        elif score < 0.5:
            return cls.LOW
        elif score < 0.7:
            return cls.MEDIUM
        elif score < 0.9:
            return cls.HIGH
        else:
            return cls.VERY_HIGH
    
    def to_score_range(self) -> tuple[float, float]:
        """Retourne intervalle scores pour ce niveau"""
        ranges = {
            self.VERY_LOW: (0.0, 0.2),
            self.LOW: (0.2, 0.5),
            self.MEDIUM: (0.5, 0.7),
            self.HIGH: (0.7, 0.9),
            self.VERY_HIGH: (0.9, 1.0)
        }
        return ranges[self]


class DetectionMethod(str, Enum):
    """Méthodes de détection d'intention"""
    
    RULES = "rules"                    # Moteur de règles heuristiques
    ML_CLASSIFIER = "ml_classifier"    # Classificateur ML (TinyBERT)
    LLM_FALLBACK = "llm_fallback"     # Fallback LLM (DeepSeek)
    HYBRID = "hybrid"                  # Combinaison méthodes
    CACHE = "cache"                    # Résultat du cache
    
    # Méthodes composées
    RULES_VS_ML = "rules_vs_ml"
    RULES_VS_LLM = "rules_vs_llm"
    ML_VS_LLM = "ml_vs_llm"
    RULES_VS_DEEPSEEK = "rules_vs_deepseek"
    
    @classmethod
    def get_primary_methods(cls) -> List[str]:
        """Retourne méthodes de détection primaires"""
        return [cls.RULES, cls.ML_CLASSIFIER, cls.LLM_FALLBACK]
    
    @classmethod
    def is_composite_method(cls, method: str) -> bool:
        """Vérifie si méthode est composée de plusieurs"""
        composite_methods = [
            cls.HYBRID, cls.RULES_VS_ML, cls.RULES_VS_LLM, 
            cls.ML_VS_LLM, cls.RULES_VS_DEEPSEEK
        ]
        return method in composite_methods


class CacheStrategy(str, Enum):
    """Stratégies de mise en cache"""
    
    DISABLED = "disabled"              # Pas de cache
    CONSERVATIVE = "conservative"      # Cache seulement haute confiance
    AGGRESSIVE = "aggressive"          # Cache même confiance moyenne
    SMART = "smart"                    # Cache adaptatif selon patterns
    
    def get_min_confidence(self) -> float:
        """Retourne confiance minimale pour cache selon stratégie"""
        thresholds = {
            self.DISABLED: float('inf'),
            self.CONSERVATIVE: 0.8,
            self.AGGRESSIVE: 0.5,
            self.SMART: 0.6
        }
        return thresholds[self]


class ProcessingStatus(str, Enum):
    """Statuts de traitement d'une requête"""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    TIMEOUT = "timeout"


class MetricType(str, Enum):
    """Types de métriques collectées"""
    
    # Métriques de performance
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    CONFIDENCE = "confidence"
    
    # Métriques de coût
    API_COST = "api_cost"
    CACHE_HIT_RATE = "cache_hit_rate"
    METHOD_DISTRIBUTION = "method_distribution"
    
    # Métriques métier
    INTENT_DISTRIBUTION = "intent_distribution"
    ENTITY_EXTRACTION_RATE = "entity_extraction_rate"
    FALLBACK_RATE = "fallback_rate"


class ErrorSeverity(IntEnum):
    """Niveaux de sévérité des erreurs"""
    
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    
    def __str__(self) -> str:
        names = {
            1: "INFO",
            2: "WARNING", 
            3: "ERROR",
            4: "CRITICAL"
        }
        return names[self.value]


# Mappings utilitaires
INTENT_TO_PRIORITY: Dict[str, int] = {
    IntentType.CARD_MANAGEMENT: 1,      # Urgence sécurité
    IntentType.TRANSFER: 1,             # Transaction critique
    IntentType.ACCOUNT_BALANCE: 2,      # Information essentielle
    IntentType.SEARCH_BY_CATEGORY: 3,   # Analyse courante
    IntentType.BUDGET_ANALYSIS: 3,      # Analyse courante
    IntentType.SEARCH_BY_DATE: 4,       # Recherche historique
    IntentType.GREETING: 5,             # Interaction sociale
    IntentType.HELP: 5,                 # Support
    IntentType.GOODBYE: 5,              # Interaction sociale
    IntentType.UNKNOWN: 10              # Priorité minimale
}

ENTITY_TO_VALIDATION_PATTERN: Dict[EntityType, str] = {
    EntityType.AMOUNT: r"^\d+(?:[,\.]\d+)?$",
    EntityType.DATE: r"^\d{1,2}/\d{1,2}/\d{4}$",
    EntityType.MONTH: r"^(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)$",
    EntityType.CARD_TYPE: r"^(visa|mastercard|cb|carte\s+bleue)$",
    EntityType.ACCOUNT_TYPE: r"^(compte\s+courant|livret\s+a|épargne|livret)$",
    EntityType.CATEGORY: r"^(restaurant|courses|transport|shopping|loisirs|santé|alimentation)$"
}

METHOD_TO_EXPECTED_LATENCY: Dict[DetectionMethod, float] = {
    DetectionMethod.CACHE: 1.0,           # Ultra-rapide
    DetectionMethod.RULES: 5.0,           # Très rapide
    DetectionMethod.ML_CLASSIFIER: 50.0,  # Rapide
    DetectionMethod.LLM_FALLBACK: 2000.0, # Lent mais précis
    DetectionMethod.HYBRID: 100.0         # Variable
}

CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "cache_eligible": 0.6,
    "high_confidence": 0.85,
    "deepseek_trigger": 0.3,
    "min_acceptable": 0.1
}


# Exports publics
__all__ = [
    # Énumérations principales
    "IntentType",
    "EntityType", 
    "ConfidenceLevel",
    "DetectionMethod",
    "CacheStrategy",
    "ProcessingStatus",
    "MetricType",
    "ErrorSeverity",
    
    # Mappings utilitaires
    "INTENT_TO_PRIORITY",
    "ENTITY_TO_VALIDATION_PATTERN",
    "METHOD_TO_EXPECTED_LATENCY",
    "CONFIDENCE_THRESHOLDS"
]