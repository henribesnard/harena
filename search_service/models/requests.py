"""
📥 Modèles Requêtes Search Service - API Endpoints
=================================================

Modèles Pydantic pour les requêtes API du Search Service spécialisé en recherche
lexicale pure Elasticsearch. Ces modèles définissent les interfaces FastAPI endpoints.

Responsabilités:
- Modèles requêtes endpoints API
- Validation paramètres entrée
- Sérialisation données requêtes
- Documentation API automatique
- Validation sécurité et limites
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from pydantic import BaseModel, Field, validator
from enum import Enum

from .service_contracts import (
    QueryType, IntentType, FilterOperator, AggregationType,
    SearchServiceQuery, SearchFilter, TextSearchQuery
)


# =============================================================================
# 🔍 REQUÊTES RECHERCHE LEXICALE
# =============================================================================

class SimpleLexicalSearchRequest(BaseModel):
    """Requête recherche lexicale simple."""
    query: str = Field(..., min_length=1, max_length=1000, description="Texte à rechercher")
    user_id: int = Field(..., ge=1, description="ID utilisateur (obligatoire)")
    fields: Optional[List[str]] = Field(
        default=["searchable_text", "primary_description", "merchant_name"],
        description="Champs de recherche"
    )
    limit: int = Field(default=20, ge=1, le=100, description="Nombre de résultats")
    offset: int = Field(default=0, ge=0, le=10000, description="Décalage pagination")
    
    @validator('query')
    def validate_search_query(cls, v):
        """Validation requête recherche."""
        if not v.strip():
            raise ValueError("Search query cannot be empty")
        return v.strip()
    
    @validator('fields')
    def validate_search_fields(cls, v):
        """Validation champs recherche."""
        allowed_fields = {
            "searchable_text", "primary_description", "merchant_name", 
            "category_name", "operation_type"
        }
        
        for field in v:
            if field not in allowed_fields:
                raise ValueError(f"Field '{field}' not allowed for search")
        
        return v

class CategorySearchRequest(BaseModel):
    """Requête recherche par catégorie."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    category: str = Field(..., min_length=1, max_length=100, description="Nom catégorie")
    date_from: Optional[date] = Field(None, description="Date début période")
    date_to: Optional[date] = Field(None, description="Date fin période")
    limit: int = Field(default=20, ge=1, le=100, description="Nombre de résultats")
    include_stats: bool = Field(default=False, description="Inclure statistiques")
    
    @validator('category')
    def validate_category_name(cls, v):
        """Validation nom catégorie."""
        return v.strip().lower()
    
    @validator('date_to')
    def validate_date_range(cls, v, values):
        """Validation cohérence plage dates."""
        date_from = values.get('date_from')
        if date_from and v and v < date_from:
            raise ValueError("date_to must be after date_from")
        return v

class MerchantSearchRequest(BaseModel):
    """Requête recherche par marchand."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    merchant: str = Field(..., min_length=1, max_length=200, description="Nom marchand")
    exact_match: bool = Field(default=False, description="Correspondance exacte")
    date_from: Optional[date] = Field(None, description="Date début")
    date_to: Optional[date] = Field(None, description="Date fin")
    limit: int = Field(default=20, ge=1, le=100, description="Nombre de résultats")
    
    @validator('merchant')
    def validate_merchant_name(cls, v):
        """Validation nom marchand."""
        return v.strip()

class AmountRangeSearchRequest(BaseModel):
    """Requête recherche par plage montant."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    amount_min: Optional[float] = Field(None, description="Montant minimum")
    amount_max: Optional[float] = Field(None, description="Montant maximum")
    amount_type: str = Field(default="absolute", description="Type montant (signed/absolute)")
    currency: Optional[str] = Field(None, max_length=3, description="Code devise")
    date_from: Optional[date] = Field(None, description="Date début")
    date_to: Optional[date] = Field(None, description="Date fin")
    limit: int = Field(default=20, ge=1, le=100, description="Nombre de résultats")
    
    @validator('amount_type')
    def validate_amount_type(cls, v):
        """Validation type montant."""
        allowed_types = {"signed", "absolute"}
        if v not in allowed_types:
            raise ValueError(f"amount_type must be one of {allowed_types}")
        return v
    
    @validator('amount_max')
    def validate_amount_range(cls, v, values):
        """Validation plage montants."""
        amount_min = values.get('amount_min')
        if amount_min is not None and v is not None and v < amount_min:
            raise ValueError("amount_max must be greater than amount_min")
        return v
    
    @validator('currency')
    def validate_currency_code(cls, v):
        """Validation code devise."""
        if v:
            v = v.upper()
            # Codes devises courants
            allowed_currencies = {"EUR", "USD", "GBP", "CHF", "CAD", "JPY"}
            if v not in allowed_currencies:
                raise ValueError(f"Unsupported currency: {v}")
        return v


# =============================================================================
# 📊 REQUÊTES AGRÉGATION ET ANALYSE
# =============================================================================

class CategoryAnalysisRequest(BaseModel):
    """Requête analyse par catégorie."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    date_from: Optional[date] = Field(None, description="Date début analyse")
    date_to: Optional[date] = Field(None, description="Date fin analyse")
    top_categories: int = Field(default=10, ge=1, le=50, description="Nombre top catégories")
    include_subcategories: bool = Field(default=False, description="Inclure sous-catégories")
    metrics: List[str] = Field(
        default=["count", "sum", "avg"],
        description="Métriques à calculer"
    )
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """Validation métriques."""
        allowed_metrics = {"count", "sum", "avg", "min", "max", "std"}
        for metric in v:
            if metric not in allowed_metrics:
                raise ValueError(f"Metric '{metric}' not supported")
        return v

class MerchantAnalysisRequest(BaseModel):
    """Requête analyse par marchand."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    date_from: Optional[date] = Field(None, description="Date début")
    date_to: Optional[date] = Field(None, description="Date fin")
    top_merchants: int = Field(default=10, ge=1, le=50, description="Nombre top marchands")
    category_filter: Optional[str] = Field(None, description="Filtrer par catégorie")
    min_transactions: int = Field(default=1, ge=1, description="Minimum transactions")
    
    @validator('category_filter')
    def validate_category_filter(cls, v):
        """Validation filtre catégorie."""
        if v:
            return v.strip().lower()
        return v

class TemporalAnalysisRequest(BaseModel):
    """Requête analyse temporelle."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    date_from: date = Field(..., description="Date début analyse")
    date_to: date = Field(..., description="Date fin analyse")
    granularity: str = Field(default="month", description="Granularité temporelle")
    category_filter: Optional[str] = Field(None, description="Filtrer par catégorie")
    merchant_filter: Optional[str] = Field(None, description="Filtrer par marchand")
    include_trends: bool = Field(default=True, description="Inclure analyse tendances")
    
    @validator('granularity')
    def validate_granularity(cls, v):
        """Validation granularité."""
        allowed_granularities = {"day", "week", "month", "quarter", "year"}
        if v not in allowed_granularities:
            raise ValueError(f"Granularity must be one of {allowed_granularities}")
        return v
    
    @validator('date_to')
    def validate_temporal_range(cls, v, values):
        """Validation plage temporelle."""
        date_from = values.get('date_from')
        if date_from and v < date_from:
            raise ValueError("date_to must be after date_from")
        
        # Limiter période analyse
        if date_from and (v - date_from).days > 730:  # 2 ans max
            raise ValueError("Analysis period cannot exceed 2 years")
        
        return v

class SpendingPatternRequest(BaseModel):
    """Requête analyse patterns de dépenses."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    pattern_type: str = Field(..., description="Type de pattern à analyser")
    lookback_days: int = Field(default=90, ge=7, le=365, description="Période d'analyse (jours)")
    min_amount: Optional[float] = Field(None, ge=0, description="Montant minimum")
    categories: Optional[List[str]] = Field(None, description="Catégories à inclure")
    
    @validator('pattern_type')
    def validate_pattern_type(cls, v):
        """Validation type pattern."""
        allowed_patterns = {
            "weekly", "monthly", "seasonal", "recurring", 
            "anomaly", "trend", "frequency"
        }
        if v not in allowed_patterns:
            raise ValueError(f"Pattern type must be one of {allowed_patterns}")
        return v


# =============================================================================
# 🔧 REQUÊTES UTILITAIRES ET VALIDATION
# =============================================================================

class ValidateQueryRequest(BaseModel):
    """Requête validation de requête Elasticsearch."""
    query: Dict[str, Any] = Field(..., description="Requête Elasticsearch à valider")
    index: str = Field(default="harena_transactions", description="Index cible")
    explain: bool = Field(default=False, description="Inclure explication")
    
    @validator('query')
    def validate_elasticsearch_query(cls, v):
        """Validation structure requête Elasticsearch."""
        if not isinstance(v, dict):
            raise ValueError("Query must be a dictionary")
        
        # Vérifications basiques structure ES
        if 'query' not in v and 'aggs' not in v:
            raise ValueError("Query must contain 'query' or 'aggs' section")
        
        return v

class HealthCheckRequest(BaseModel):
    """Requête health check."""
    detailed: bool = Field(default=False, description="Check détaillé")
    include_metrics: bool = Field(default=False, description="Inclure métriques")
    check_elasticsearch: bool = Field(default=True, description="Vérifier Elasticsearch")
    check_redis: bool = Field(default=True, description="Vérifier Redis")

class MetricsRequest(BaseModel):
    """Requête métriques système."""
    time_range: str = Field(default="1h", description="Période métriques")
    detailed: bool = Field(default=False, description="Métriques détaillées")
    include_performance: bool = Field(default=True, description="Inclure performance")
    include_errors: bool = Field(default=True, description="Inclure erreurs")
    
    @validator('time_range')
    def validate_time_range(cls, v):
        """Validation période métriques."""
        allowed_ranges = {"5m", "15m", "1h", "6h", "24h", "7d", "30d"}
        if v not in allowed_ranges:
            raise ValueError(f"Time range must be one of {allowed_ranges}")
        return v


# =============================================================================
# 🤝 REQUÊTES CONTRATS (Interface avec Conversation Service)
# =============================================================================

class ContractSearchRequest(BaseModel):
    """Requête utilisant contrat SearchServiceQuery."""
    contract: SearchServiceQuery = Field(..., description="Contrat requête standardisé")
    validate_contract: bool = Field(default=True, description="Valider contrat")
    debug_mode: bool = Field(default=False, description="Mode debug")
    
    class Config:
        """Configuration Pydantic."""
        schema_extra = {
            "example": {
                "contract": {
                    "query_metadata": {
                        "user_id": 34,
                        "intent_type": "SEARCH_BY_CATEGORY",
                        "confidence": 0.95,
                        "agent_name": "query_generator_agent"
                    },
                    "search_parameters": {
                        "query_type": "filtered_search",
                        "fields": ["searchable_text", "primary_description"],
                        "limit": 20
                    },
                    "filters": {
                        "required": [
                            {"field": "user_id", "operator": "eq", "value": 34},
                            {"field": "category_name", "operator": "eq", "value": "restaurant"}
                        ]
                    }
                },
                "validate_contract": True,
                "debug_mode": False
            }
        }


# =============================================================================
# 📋 REQUÊTES BATCH ET BULK
# =============================================================================

class BatchSearchRequest(BaseModel):
    """Requête recherche batch."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    queries: List[Dict[str, Any]] = Field(..., min_items=1, max_items=10, description="Liste requêtes")
    parallel_execution: bool = Field(default=True, description="Exécution parallèle")
    fail_fast: bool = Field(default=False, description="Arrêter au premier échec")
    
    @validator('queries')
    def validate_batch_queries(cls, v):
        """Validation requêtes batch."""
        if len(v) > 10:
            raise ValueError("Maximum 10 queries per batch")
        
        for i, query in enumerate(v):
            if not isinstance(query, dict):
                raise ValueError(f"Query {i} must be a dictionary")
        
        return v

class BulkValidationRequest(BaseModel):
    """Requête validation bulk de requêtes."""
    queries: List[SearchServiceQuery] = Field(..., min_items=1, max_items=50, description="Requêtes à valider")
    strict_mode: bool = Field(default=True, description="Validation stricte")
    return_details: bool = Field(default=False, description="Retourner détails validation")
    
    @validator('queries')
    def validate_bulk_queries(cls, v):
        """Validation requêtes bulk."""
        if len(v) > 50:
            raise ValueError("Maximum 50 queries for bulk validation")
        return v


# =============================================================================
# 🎯 REQUÊTES SPÉCIALISÉES FINANCIÈRES
# =============================================================================

class RecurringTransactionRequest(BaseModel):
    """Requête détection transactions récurrentes."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    lookback_months: int = Field(default=6, ge=1, le=24, description="Période analyse (mois)")
    min_occurrences: int = Field(default=3, ge=2, le=50, description="Occurrences minimum")
    amount_tolerance: float = Field(default=0.05, ge=0, le=0.5, description="Tolérance montant (%)")
    date_tolerance_days: int = Field(default=3, ge=0, le=15, description="Tolérance date (jours)")
    
    @validator('amount_tolerance')
    def validate_amount_tolerance(cls, v):
        """Validation tolérance montant."""
        if not 0 <= v <= 0.5:
            raise ValueError("Amount tolerance must be between 0 and 0.5 (50%)")
        return v

class SuspiciousActivityRequest(BaseModel):
    """Requête détection activité suspecte."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    detection_rules: List[str] = Field(..., description="Règles de détection")
    sensitivity: str = Field(default="medium", description="Sensibilité détection")
    lookback_days: int = Field(default=30, ge=1, le=90, description="Période analyse")
    
    @validator('sensitivity')
    def validate_sensitivity(cls, v):
        """Validation sensibilité."""
        allowed_levels = {"low", "medium", "high"}
        if v not in allowed_levels:
            raise ValueError(f"Sensitivity must be one of {allowed_levels}")
        return v
    
    @validator('detection_rules')
    def validate_detection_rules(cls, v):
        """Validation règles détection."""
        allowed_rules = {
            "unusual_amount", "unusual_merchant", "unusual_location",
            "unusual_time", "unusual_frequency", "unusual_category"
        }
        
        for rule in v:
            if rule not in allowed_rules:
                raise ValueError(f"Unknown detection rule: {rule}")
        
        return v

class BudgetAnalysisRequest(BaseModel):
    """Requête analyse budget."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    budget_period: str = Field(..., description="Période budget")
    categories: Optional[List[str]] = Field(None, description="Catégories à analyser")
    include_projections: bool = Field(default=True, description="Inclure projections")
    compare_previous: bool = Field(default=True, description="Comparer période précédente")
    
    @validator('budget_period')
    def validate_budget_period(cls, v):
        """Validation période budget."""
        allowed_periods = {"current_month", "current_quarter", "current_year", "custom"}
        if v not in allowed_periods:
            raise ValueError(f"Budget period must be one of {allowed_periods}")
        return v


# =============================================================================
# 📋 CLASSE FACTORY REQUÊTES
# =============================================================================

class RequestFactory:
    """Factory pour créer requêtes selon le type."""
    
    @staticmethod
    def create_simple_search(user_id: int, query: str, **kwargs) -> SimpleLexicalSearchRequest:
        """Créer requête recherche simple."""
        return SimpleLexicalSearchRequest(
            user_id=user_id,
            query=query,
            **kwargs
        )
    
    @staticmethod
    def create_category_search(user_id: int, category: str, **kwargs) -> CategorySearchRequest:
        """Créer requête recherche catégorie."""
        return CategorySearchRequest(
            user_id=user_id,
            category=category,
            **kwargs
        )
    
    @staticmethod
    def create_merchant_search(user_id: int, merchant: str, **kwargs) -> MerchantSearchRequest:
        """Créer requête recherche marchand."""
        return MerchantSearchRequest(
            user_id=user_id,
            merchant=merchant,
            **kwargs
        )
    
    @staticmethod
    def create_temporal_analysis(user_id: int, date_from: date, date_to: date, **kwargs) -> TemporalAnalysisRequest:
        """Créer requête analyse temporelle."""
        return TemporalAnalysisRequest(
            user_id=user_id,
            date_from=date_from,
            date_to=date_to,
            **kwargs
        )
    
    @staticmethod
    def from_contract(contract: SearchServiceQuery, **kwargs) -> ContractSearchRequest:
        """Créer requête depuis contrat."""
        return ContractSearchRequest(
            contract=contract,
            **kwargs
        )


# =============================================================================
# 📋 EXPORTS
# =============================================================================

__all__ = [
    # Requêtes recherche
    "SimpleLexicalSearchRequest", "CategorySearchRequest", "MerchantSearchRequest", "AmountRangeSearchRequest",
    # Requêtes analyse
    "CategoryAnalysisRequest", "MerchantAnalysisRequest", "TemporalAnalysisRequest", "SpendingPatternRequest",
    # Requêtes utilitaires
    "ValidateQueryRequest", "HealthCheckRequest", "MetricsRequest",
    # Requêtes contrats
    "ContractSearchRequest",
    # Requêtes batch
    "BatchSearchRequest", "BulkValidationRequest",
    # Requêtes spécialisées
    "RecurringTransactionRequest", "SuspiciousActivityRequest", "BudgetAnalysisRequest",
    # Factory
    "RequestFactory",
]