"""
Templates de requêtes Elasticsearch par intention financière
Bibliothèque de templates optimisés pour le Search Service
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import logging

from search_service.models.elasticsearch_queries import (
    ESSearchQuery, FinancialTransactionQueryBuilder
)
from search_service.models.service_contracts import (
    SearchServiceQuery, FilterOperator, AggregationRequest
)


logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Types d'intentions financières supportés"""
    # === RECHERCHE SIMPLE ===
    SEARCH_BY_CATEGORY = "search_by_category"
    SEARCH_BY_MERCHANT = "search_by_merchant"
    SEARCH_BY_AMOUNT = "search_by_amount"
    SEARCH_BY_DATE = "search_by_date"
    
    # === RECHERCHE TEXTUELLE ===
    TEXT_SEARCH = "text_search"
    TEXT_SEARCH_WITH_CATEGORY = "text_search_with_category"
    TEXT_SEARCH_WITH_MERCHANT = "text_search_with_merchant"
    TEXT_SEARCH_WITH_DATE = "text_search_with_date"
    
    # === ANALYSE ET COMPTAGE ===
    COUNT_OPERATIONS = "count_operations"
    COUNT_OPERATIONS_BY_CATEGORY = "count_operations_by_category"
    COUNT_OPERATIONS_BY_AMOUNT = "count_operations_by_amount"
    COUNT_OPERATIONS_BY_DATE = "count_operations_by_date"
    
    # === ANALYSE TEMPORELLE ===
    TEMPORAL_SPENDING_ANALYSIS = "temporal_spending_analysis"
    MONTHLY_BREAKDOWN = "monthly_breakdown"
    WEEKLY_PATTERN_ANALYSIS = "weekly_pattern_analysis"
    DAILY_SPENDING_TREND = "daily_spending_trend"
    
    # === ANALYSE CATÉGORIELLE ===
    CATEGORY_SPENDING_ANALYSIS = "category_spending_analysis"
    TOP_MERCHANTS_ANALYSIS = "top_merchants_analysis"
    EXPENSE_DISTRIBUTION = "expense_distribution"
    
    # === COMPARAISONS ===
    PERIOD_COMPARISON = "period_comparison"
    CATEGORY_COMPARISON = "category_comparison"
    MERCHANT_COMPARISON = "merchant_comparison"
    
    # === RECHERCHE AVANCÉE ===
    COMPLEX_FILTER_SEARCH = "complex_filter_search"
    RANGE_ANALYSIS = "range_analysis"
    PATTERN_DETECTION = "pattern_detection"


class TemplateParameterType(str, Enum):
    """Types de paramètres de template"""
    USER_ID = "user_id"
    CATEGORY = "category"
    MERCHANT = "merchant"
    AMOUNT = "amount"
    DATE_RANGE = "date_range"
    TEXT_QUERY = "text_query"
    LIMIT = "limit"
    OFFSET = "offset"
    AGGREGATION_FIELD = "aggregation_field"
    GROUP_BY_FIELD = "group_by_field"


class TemplateParameter(BaseModel):
    """Paramètre de template avec validation"""
    name: str = Field(..., description="Nom du paramètre")
    type: TemplateParameterType = Field(..., description="Type du paramètre")
    required: bool = Field(default=True, description="Paramètre obligatoire")
    default_value: Optional[Any] = Field(default=None, description="Valeur par défaut")
    validation_rule: Optional[str] = Field(default=None, description="Règle de validation")


class QueryTemplate(BaseModel):
    """Template de requête avec métadonnées"""
    intent_type: IntentType = Field(..., description="Type d'intention")
    template_name: str = Field(..., description="Nom du template")
    description: str = Field(..., description="Description du template")
    parameters: List[TemplateParameter] = Field(..., description="Paramètres du template")
    performance_category: str = Field(..., description="Catégorie de performance")
    estimated_time_ms: int = Field(..., description="Temps d'exécution estimé en ms")
    cache_duration_minutes: int = Field(default=30, description="Durée de cache en minutes")
    version: str = Field(default="1.0", description="Version du template")


class QueryTemplateEngine:
    """Moteur de génération de templates de requêtes Elasticsearch"""
    
    def __init__(self):
        self.templates: Dict[IntentType, QueryTemplate] = self._initialize_templates()
        self.performance_cache = {}
        
    def _initialize_templates(self) -> Dict[IntentType, QueryTemplate]:
        """Initialise tous les templates disponibles"""
        templates = {}
        
        # === TEMPLATES RECHERCHE SIMPLE ===
        templates[IntentType.SEARCH_BY_CATEGORY] = QueryTemplate(
            intent_type=IntentType.SEARCH_BY_CATEGORY,
            template_name="category_search",
            description="Recherche de transactions par catégorie",
            parameters=[
                TemplateParameter(name="user_id", type=TemplateParameterType.USER_ID),
                TemplateParameter(name="category", type=TemplateParameterType.CATEGORY),
                TemplateParameter(name="limit", type=TemplateParameterType.LIMIT, required=False, default_value=20)
            ],
            performance_category="fast",
            estimated_time_ms=15
        )
        
        templates[IntentType.SEARCH_BY_MERCHANT] = QueryTemplate(
            intent_type=IntentType.SEARCH_BY_MERCHANT,
            template_name="merchant_search",
            description="Recherche de transactions par marchand",
            parameters=[
                TemplateParameter(name="user_id", type=TemplateParameterType.USER_ID),
                TemplateParameter(name="merchant", type=TemplateParameterType.MERCHANT),
                TemplateParameter(name="limit", type=TemplateParameterType.LIMIT, required=False, default_value=20)
            ],
            performance_category="fast",
            estimated_time_ms=18
        )
        
        # === TEMPLATES RECHERCHE TEXTUELLE ===
        templates[IntentType.TEXT_SEARCH] = QueryTemplate(
            intent_type=IntentType.TEXT_SEARCH,
            template_name="text_search",
            description="Recherche textuelle libre sur les descriptions",
            parameters=[
                TemplateParameter(name="user_id", type=TemplateParameterType.USER_ID),
                TemplateParameter(name="text_query", type=TemplateParameterType.TEXT_QUERY),
                TemplateParameter(name="limit", type=TemplateParameterType.LIMIT, required=False, default_value=20)
            ],
            performance_category="medium",
            estimated_time_ms=45
        )
        
        templates[IntentType.TEXT_SEARCH_WITH_CATEGORY] = QueryTemplate(
            intent_type=IntentType.TEXT_SEARCH_WITH_CATEGORY,
            template_name="text_search_with_category",
            description="Recherche textuelle filtrée par catégorie",
            parameters=[
                TemplateParameter(name="user_id", type=TemplateParameterType.USER_ID),
                TemplateParameter(name="text_query", type=TemplateParameterType.TEXT_QUERY),
                TemplateParameter(name="category", type=TemplateParameterType.CATEGORY),
                TemplateParameter(name="limit", type=TemplateParameterType.LIMIT, required=False, default_value=20)
            ],
            performance_category="medium",
            estimated_time_ms=55
        )
        
        # === TEMPLATES ANALYSE ET COMPTAGE ===
        templates[IntentType.COUNT_OPERATIONS] = QueryTemplate(
            intent_type=IntentType.COUNT_OPERATIONS,
            template_name="count_operations",
            description="Comptage total des opérations",
            parameters=[
                TemplateParameter(name="user_id", type=TemplateParameterType.USER_ID),
                TemplateParameter(name="date_range", type=TemplateParameterType.DATE_RANGE, required=False)
            ],
            performance_category="fast",
            estimated_time_ms=12
        )
        
        templates[IntentType.COUNT_OPERATIONS_BY_AMOUNT] = QueryTemplate(
            intent_type=IntentType.COUNT_OPERATIONS_BY_AMOUNT,
            template_name="count_operations_by_amount",
            description="Comptage des opérations par plage de montant",
            parameters=[
                TemplateParameter(name="user_id", type=TemplateParameterType.USER_ID),
                TemplateParameter(name="amount", type=TemplateParameterType.AMOUNT)
            ],
            performance_category="fast",
            estimated_time_ms=20
        )
        
        # === TEMPLATES ANALYSE TEMPORELLE ===
        templates[IntentType.TEMPORAL_SPENDING_ANALYSIS] = QueryTemplate(
            intent_type=IntentType.TEMPORAL_SPENDING_ANALYSIS,
            template_name="temporal_spending_analysis",
            description="Analyse des dépenses sur une période",
            parameters=[
                TemplateParameter(name="user_id", type=TemplateParameterType.USER_ID),
                TemplateParameter(name="date_range", type=TemplateParameterType.DATE_RANGE),
                TemplateParameter(name="aggregation_field", type=TemplateParameterType.AGGREGATION_FIELD, required=False, default_value="amount_abs")
            ],
            performance_category="slow",
            estimated_time_ms=85
        )
        
        templates[IntentType.MONTHLY_BREAKDOWN] = QueryTemplate(
            intent_type=IntentType.MONTHLY_BREAKDOWN,
            template_name="monthly_breakdown",
            description="Répartition mensuelle des dépenses",
            parameters=[
                TemplateParameter(name="user_id", type=TemplateParameterType.USER_ID),
                TemplateParameter(name="date_range", type=TemplateParameterType.DATE_RANGE, required=False)
            ],
            performance_category="medium",
            estimated_time_ms=65
        )
        
        # === TEMPLATES ANALYSE CATÉGORIELLE ===
        templates[IntentType.CATEGORY_SPENDING_ANALYSIS] = QueryTemplate(
            intent_type=IntentType.CATEGORY_SPENDING_ANALYSIS,
            template_name="category_spending_analysis",
            description="Analyse des dépenses par catégorie",
            parameters=[
                TemplateParameter(name="user_id", type=TemplateParameterType.USER_ID),
                TemplateParameter(name="date_range", type=TemplateParameterType.DATE_RANGE, required=False)
            ],
            performance_category="medium",
            estimated_time_ms=70
        )
        
        templates[IntentType.TOP_MERCHANTS_ANALYSIS] = QueryTemplate(
            intent_type=IntentType.TOP_MERCHANTS_ANALYSIS,
            template_name="top_merchants_analysis",
            description="Analyse des top marchands par dépenses",
            parameters=[
                TemplateParameter(name="user_id", type=TemplateParameterType.USER_ID),
                TemplateParameter(name="limit", type=TemplateParameterType.LIMIT, required=False, default_value=10)
            ],
            performance_category="medium",
            estimated_time_ms=60
        )
        
        return templates
    
    def get_template(self, intent_type: IntentType) -> Optional[QueryTemplate]:
        """Récupère un template par type d'intention"""
        return self.templates.get(intent_type)
    
    def list_templates(self) -> List[QueryTemplate]:
        """Liste tous les templates disponibles"""
        return list(self.templates.values())
    
    def get_templates_by_performance(self, category: str) -> List[QueryTemplate]:
        """Récupère les templates par catégorie de performance"""
        return [t for t in self.templates.values() if t.performance_category == category]


class FinancialQueryTemplateBuilder:
    """Builder spécialisé pour la construction de requêtes financières à partir de templates"""
    
    def __init__(self):
        self.template_engine = QueryTemplateEngine()
        
    def build_from_contract(self, contract: SearchServiceQuery) -> ESSearchQuery:
        """Construit une requête Elasticsearch à partir d'un contrat"""
        intent_type = IntentType(contract.query_metadata.intent_type)
        template = self.template_engine.get_template(intent_type)
        
        if not template:
            logger.warning(f"Template non trouvé pour l'intention: {intent_type}")
            return self._build_fallback_query(contract)
        
        return self._build_from_template(contract, template)
    
    def _build_from_template(self, contract: SearchServiceQuery, template: QueryTemplate) -> ESSearchQuery:
        """Construit une requête à partir d'un template spécifique"""
        builder = FinancialTransactionQueryBuilder()
        
        # Extraction des paramètres du contrat
        user_id = contract.query_metadata.user_id
        limit = contract.search_parameters.limit
        offset = contract.search_parameters.offset
        
        # Ajout obligatoire du filtre utilisateur
        builder.add_user_filter(user_id)
        
        # Construction selon le type d'intention
        if template.intent_type == IntentType.SEARCH_BY_CATEGORY:
            return self._build_category_search(builder, contract, limit, offset)
            
        elif template.intent_type == IntentType.SEARCH_BY_MERCHANT:
            return self._build_merchant_search(builder, contract, limit, offset)
            
        elif template.intent_type == IntentType.TEXT_SEARCH:
            return self._build_text_search(builder, contract, limit, offset)
            
        elif template.intent_type == IntentType.TEXT_SEARCH_WITH_CATEGORY:
            return self._build_text_search_with_category(builder, contract, limit, offset)
            
        elif template.intent_type == IntentType.COUNT_OPERATIONS:
            return self._build_count_operations(builder, contract, limit, offset)
            
        elif template.intent_type == IntentType.COUNT_OPERATIONS_BY_AMOUNT:
            return self._build_count_operations_by_amount(builder, contract, limit, offset)
            
        elif template.intent_type == IntentType.TEMPORAL_SPENDING_ANALYSIS:
            return self._build_temporal_spending_analysis(builder, contract, limit, offset)
            
        elif template.intent_type == IntentType.MONTHLY_BREAKDOWN:
            return self._build_monthly_breakdown(builder, contract, limit, offset)
            
        elif template.intent_type == IntentType.CATEGORY_SPENDING_ANALYSIS:
            return self._build_category_spending_analysis(builder, contract, limit, offset)
            
        elif template.intent_type == IntentType.TOP_MERCHANTS_ANALYSIS:
            return self._build_top_merchants_analysis(builder, contract, limit, offset)
            
        else:
            logger.warning(f"Template non implémenté pour: {template.intent_type}")
            return self._build_fallback_query(contract)
    
    def _build_category_search(self, builder: FinancialTransactionQueryBuilder, 
                              contract: SearchServiceQuery, limit: int, offset: int) -> ESSearchQuery:
        """Construit une recherche par catégorie"""
        # Récupération de la catégorie depuis les filtres
        category_filter = next((f for f in contract.filters.required if f.field == "category_name"), None)
        if category_filter:
            builder.add_category_filter(category_filter.value)
        
        # Configuration standard
        return (builder
                .set_pagination(limit, offset)
                .set_sort_by_relevance_and_date()
                .add_highlighting()
                .build())
    
    def _build_merchant_search(self, builder: FinancialTransactionQueryBuilder,
                              contract: SearchServiceQuery, limit: int, offset: int) -> ESSearchQuery:
        """Construit une recherche par marchand"""
        # Récupération du marchand depuis les filtres
        merchant_filter = next((f for f in contract.filters.required if f.field == "merchant_name"), None)
        if merchant_filter:
            builder.add_merchant_filter(merchant_filter.value)
        
        return (builder
                .set_pagination(limit, offset)
                .set_sort_by_relevance_and_date()
                .add_highlighting()
                .build())
    
    def _build_text_search(self, builder: FinancialTransactionQueryBuilder,
                          contract: SearchServiceQuery, limit: int, offset: int) -> ESSearchQuery:
        """Construit une recherche textuelle"""
        if contract.text_search:
            builder.add_text_search(
                contract.text_search.query,
                fields=contract.text_search.fields,
                fuzziness=contract.text_search.fuzziness
            )
        
        return (builder
                .set_pagination(limit, offset)
                .set_sort_by_relevance_and_date()
                .add_highlighting()
                .build())
    
    def _build_text_search_with_category(self, builder: FinancialTransactionQueryBuilder,
                                        contract: SearchServiceQuery, limit: int, offset: int) -> ESSearchQuery:
        """Construit une recherche textuelle avec filtre catégorie"""
        # Ajout du filtre catégorie
        category_filter = next((f for f in contract.filters.required if f.field == "category_name"), None)
        if category_filter:
            builder.add_category_filter(category_filter.value)
        
        # Ajout de la recherche textuelle
        if contract.text_search:
            builder.add_text_search(
                contract.text_search.query,
                fields=contract.text_search.fields,
                fuzziness=contract.text_search.fuzziness
            )
        
        return (builder
                .set_pagination(limit, offset)
                .set_sort_by_relevance_and_date()
                .add_highlighting()
                .build())
    
    def _build_count_operations(self, builder: FinancialTransactionQueryBuilder,
                               contract: SearchServiceQuery, limit: int, offset: int) -> ESSearchQuery:
        """Construit un comptage d'opérations"""
        # Ajout des filtres de date si présents
        self._add_date_filters(builder, contract)
        
        # Configuration pour comptage uniquement
        query = builder.build()
        query.size = 0  # Pas besoin des documents, juste le count
        
        return query
    
    def _build_count_operations_by_amount(self, builder: FinancialTransactionQueryBuilder,
                                         contract: SearchServiceQuery, limit: int, offset: int) -> ESSearchQuery:
        """Construit un comptage d'opérations par montant"""
        # Ajout des filtres de montant
        amount_filters = [f for f in contract.filters.ranges if f.field in ["amount", "amount_abs"]]
        for amount_filter in amount_filters:
            if amount_filter.operator == FilterOperator.GT:
                builder.add_amount_range_filter(min_amount=amount_filter.value)
            elif amount_filter.operator == FilterOperator.LT:
                builder.add_amount_range_filter(max_amount=amount_filter.value)
            elif amount_filter.operator == FilterOperator.BETWEEN:
                min_val, max_val = amount_filter.value
                builder.add_amount_range_filter(min_amount=min_val, max_amount=max_val)
        
        query = builder.build()
        query.size = 0  # Comptage uniquement
        
        return query
    
    def _build_temporal_spending_analysis(self, builder: FinancialTransactionQueryBuilder,
                                         contract: SearchServiceQuery, limit: int, offset: int) -> ESSearchQuery:
        """Construit une analyse temporelle des dépenses"""
        # Ajout des filtres de date
        self._add_date_filters(builder, contract)
        
        # Configuration pour agrégation temporelle
        query = builder.build()
        query.size = 0  # Pas de documents individuels
        
        # Ajout d'agrégations temporelles
        if contract.aggregations and contract.aggregations.enabled:
            self._add_temporal_aggregations(query, contract.aggregations)
        
        return query
    
    def _build_monthly_breakdown(self, builder: FinancialTransactionQueryBuilder,
                                contract: SearchServiceQuery, limit: int, offset: int) -> ESSearchQuery:
        """Construit une répartition mensuelle"""
        # Ajout des filtres de date
        self._add_date_filters(builder, contract)
        
        query = builder.build()
        query.size = 0
        
        # Agrégation par mois
        self._add_monthly_aggregation(query)
        
        return query
    
    def _build_category_spending_analysis(self, builder: FinancialTransactionQueryBuilder,
                                         contract: SearchServiceQuery, limit: int, offset: int) -> ESSearchQuery:
        """Construit une analyse des dépenses par catégorie"""
        # Ajout des filtres de date
        self._add_date_filters(builder, contract)
        
        query = builder.build()
        query.size = 0
        
        # Agrégation par catégorie
        self._add_category_aggregation(query)
        
        return query
    
    def _build_top_merchants_analysis(self, builder: FinancialTransactionQueryBuilder,
                                     contract: SearchServiceQuery, limit: int, offset: int) -> ESSearchQuery:
        """Construit une analyse des top marchands"""
        # Ajout des filtres de date
        self._add_date_filters(builder, contract)
        
        query = builder.build()
        query.size = 0
        
        # Agrégation par marchand avec somme des montants
        self._add_merchants_aggregation(query, limit)
        
        return query
    
    def _add_date_filters(self, builder: FinancialTransactionQueryBuilder, contract: SearchServiceQuery):
        """Ajoute les filtres de date à partir du contrat"""
        date_filters = [f for f in contract.filters.ranges if f.field == "date"]
        for date_filter in date_filters:
            if date_filter.operator == FilterOperator.BETWEEN:
                start_date, end_date = date_filter.value
                builder.add_date_range_filter(start_date, end_date)
            elif date_filter.operator == FilterOperator.GTE:
                builder.add_date_range_filter(start_date=date_filter.value)
            elif date_filter.operator == FilterOperator.LTE:
                builder.add_date_range_filter(end_date=date_filter.value)
    
    def _add_temporal_aggregations(self, query: ESSearchQuery, agg_config: AggregationRequest):
        """Ajoute des agrégations temporelles à la requête"""
        # Note: Cette méthode devrait utiliser les classes d'agrégation d'elasticsearch_queries
        # Pour l'instant, on ajoute directement au dict (à améliorer avec les modèles ES)
        if not query.aggs:
            query.aggs = {}
        
        query.aggs["spending_over_time"] = {
            "date_histogram": {
                "field": "date",
                "interval": agg_config.date_histogram_interval or "month"
            },
            "aggs": {
                "total_amount": {
                    "sum": {"field": "amount_abs"}
                },
                "avg_amount": {
                    "avg": {"field": "amount_abs"}
                },
                "transaction_count": {
                    "value_count": {"field": "transaction_id"}
                }
            }
        }
    
    def _add_monthly_aggregation(self, query: ESSearchQuery):
        """Ajoute une agrégation mensuelle"""
        if not query.aggs:
            query.aggs = {}
        
        query.aggs["monthly_breakdown"] = {
            "terms": {
                "field": "month_year",
                "size": 12
            },
            "aggs": {
                "total_spending": {
                    "sum": {"field": "amount_abs"}
                },
                "transaction_count": {
                    "value_count": {"field": "transaction_id"}
                }
            }
        }
    
    def _add_category_aggregation(self, query: ESSearchQuery):
        """Ajoute une agrégation par catégorie"""
        if not query.aggs:
            query.aggs = {}
        
        query.aggs["spending_by_category"] = {
            "terms": {
                "field": "category_name.keyword",
                "size": 20
            },
            "aggs": {
                "total_spending": {
                    "sum": {"field": "amount_abs"}
                },
                "avg_spending": {
                    "avg": {"field": "amount_abs"}
                },
                "transaction_count": {
                    "value_count": {"field": "transaction_id"}
                }
            }
        }
    
    def _add_merchants_aggregation(self, query: ESSearchQuery, limit: int):
        """Ajoute une agrégation par marchand"""
        if not query.aggs:
            query.aggs = {}
        
        query.aggs["top_merchants"] = {
            "terms": {
                "field": "merchant_name.keyword",
                "size": limit,
                "order": {"total_spending": "desc"}
            },
            "aggs": {
                "total_spending": {
                    "sum": {"field": "amount_abs"}
                },
                "transaction_count": {
                    "value_count": {"field": "transaction_id"}
                }
            }
        }
    
    def _build_fallback_query(self, contract: SearchServiceQuery) -> ESSearchQuery:
        """Construit une requête de fallback générique"""
        logger.info("Construction d'une requête de fallback générique")
        
        builder = FinancialTransactionQueryBuilder()
        builder.add_user_filter(contract.query_metadata.user_id)
        
        # Ajout des filtres du contrat
        for filter_obj in contract.filters.required:
            if filter_obj.field != "user_id":  # user_id déjà ajouté
                if filter_obj.field == "category_name":
                    builder.add_category_filter(filter_obj.value)
                elif filter_obj.field == "merchant_name":
                    builder.add_merchant_filter(filter_obj.value)
        
        # Ajout de la recherche textuelle si présente
        if contract.text_search:
            builder.add_text_search(
                contract.text_search.query,
                fields=contract.text_search.fields
            )
        
        return (builder
                .set_pagination(contract.search_parameters.limit, contract.search_parameters.offset)
                .set_sort_by_relevance_and_date()
                .add_highlighting()
                .build())


class QueryTemplateValidator:
    """Validateur pour les templates de requêtes"""
    
    @staticmethod
    def validate_template_parameters(template: QueryTemplate, parameters: Dict[str, Any]) -> bool:
        """Valide que tous les paramètres requis sont présents"""
        required_params = [p.name for p in template.parameters if p.required]
        
        for param_name in required_params:
            if param_name not in parameters:
                logger.error(f"Paramètre requis manquant: {param_name} pour le template {template.template_name}")
                return False
        
        return True
    
    @staticmethod
    def validate_query_contract(contract: SearchServiceQuery) -> bool:
        """Valide qu'un contrat de requête est correct"""
        # Vérification de l'user_id
        if contract.query_metadata.user_id <= 0:
            logger.error("user_id invalide dans le contrat")
            return False
        
        # Vérification que le filtre user_id est présent
        user_filter_exists = any(
            f.field == "user_id" for f in contract.filters.required
        )
        if not user_filter_exists:
            logger.error("Filtre user_id obligatoire manquant")
            return False
        
        # Vérification des limites
        if contract.search_parameters.limit > 1000:
            logger.error("Limite trop élevée (max 1000)")
            return False
        
        return True


# === INSTANCE GLOBALE ===
financial_template_builder = FinancialQueryTemplateBuilder()
template_validator = QueryTemplateValidator()


# === FONCTIONS D'UTILITÉ ===

def build_query_from_intent(contract: SearchServiceQuery) -> ESSearchQuery:
    """
    Fonction utilitaire principale pour construire une requête à partir d'un contrat
    """
    if not template_validator.validate_query_contract(contract):
        raise ValueError("Contrat de requête invalide")
    
    return financial_template_builder.build_from_contract(contract)


def get_available_templates() -> List[QueryTemplate]:
    """
    Retourne la liste de tous les templates disponibles
    """
    engine = QueryTemplateEngine()
    return engine.list_templates()


def get_template_by_intent(intent_type: str) -> Optional[QueryTemplate]:
    """
    Récupère un template par type d'intention
    """
    try:
        intent = IntentType(intent_type)
        engine = QueryTemplateEngine()
        return engine.get_template(intent)
    except ValueError:
        logger.error(f"Type d'intention inconnu: {intent_type}")
        return None


def get_performance_estimate(intent_type: str) -> int:
    """
    Retourne l'estimation de temps d'exécution pour un type d'intention
    """
    template = get_template_by_intent(intent_type)
    return template.estimated_time_ms if template else 100  # Défaut: 100ms


def validate_template_compatibility(contract: SearchServiceQuery) -> bool:
    """
    Valide qu'un contrat est compatible avec les templates disponibles
    """
    template = get_template_by_intent(contract.query_metadata.intent_type)
    if not template:
        return False
    
    # Extraction des paramètres du contrat
    contract_params = _extract_contract_parameters(contract)
    
    return template_validator.validate_template_parameters(template, contract_params)


def _extract_contract_parameters(contract: SearchServiceQuery) -> Dict[str, Any]:
    """
    Extrait les paramètres d'un contrat pour la validation
    """
    params = {
        "user_id": contract.query_metadata.user_id,
        "limit": contract.search_parameters.limit,
        "offset": contract.search_parameters.offset
    }
    
    # Extraction des filtres
    for filter_obj in contract.filters.required:
        if filter_obj.field == "category_name":
            params["category"] = filter_obj.value
        elif filter_obj.field == "merchant_name":
            params["merchant"] = filter_obj.value
    
    # Extraction des plages de dates
    date_filters = [f for f in contract.filters.ranges if f.field == "date"]
    if date_filters:
        params["date_range"] = date_filters[0].value
    
    # Extraction de la recherche textuelle
    if contract.text_search:
        params["text_query"] = contract.text_search.query
    
    return params


class TemplatePerformanceOptimizer:
    """
    Optimiseur de performance pour les templates de requêtes
    """
    
    def __init__(self):
        self.performance_history = {}
        self.optimization_rules = self._initialize_optimization_rules()
    
    def _initialize_optimization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les règles d'optimisation par type de template"""
        return {
            "fast": {
                "max_timeout_ms": 100,
                "cache_duration_minutes": 60,
                "prefer_term_queries": True
            },
            "medium": {
                "max_timeout_ms": 500,
                "cache_duration_minutes": 30,
                "enable_highlighting": True
            },
            "slow": {
                "max_timeout_ms": 2000,
                "cache_duration_minutes": 15,
                "enable_aggregation_cache": True
            }
        }
    
    def optimize_query(self, query: ESSearchQuery, template: QueryTemplate) -> ESSearchQuery:
        """
        Optimise une requête selon sa catégorie de performance
        """
        rules = self.optimization_rules.get(template.performance_category, {})
        
        # Application du timeout optimal
        if "max_timeout_ms" in rules:
            query.timeout = f"{rules['max_timeout_ms']}ms"
        
        # Optimisation selon la catégorie
        if template.performance_category == "fast":
            query = self._optimize_for_speed(query)
        elif template.performance_category == "medium":
            query = self._optimize_for_balance(query)
        elif template.performance_category == "slow":
            query = self._optimize_for_completeness(query)
        
        return query
    
    def _optimize_for_speed(self, query: ESSearchQuery) -> ESSearchQuery:
        """Optimisations pour les requêtes rapides"""
        # Désactiver le highlighting si pas nécessaire
        if hasattr(query, 'highlight') and query.highlight:
            query.highlight = None
        
        # Limiter la taille des résultats
        if query.size and query.size > 50:
            query.size = 50
        
        return query
    
    def _optimize_for_balance(self, query: ESSearchQuery) -> ESSearchQuery:
        """Optimisations pour les requêtes équilibrées"""
        # Configurer le highlighting optimal
        if hasattr(query, 'highlight') and query.highlight:
            query.highlight["fragment_size"] = 100
            query.highlight["number_of_fragments"] = 1
        
        return query
    
    def _optimize_for_completeness(self, query: ESSearchQuery) -> ESSearchQuery:
        """Optimisations pour les requêtes complexes"""
        # Activer la request cache pour les agrégations lourdes
        if hasattr(query, 'aggs') and query.aggs:
            # Note: L'activation du cache se ferait au niveau du client ES
            pass
        
        return query
    
    def record_performance(self, template_name: str, execution_time_ms: int):
        """Enregistre les performances d'exécution"""
        if template_name not in self.performance_history:
            self.performance_history[template_name] = []
        
        self.performance_history[template_name].append(execution_time_ms)
        
        # Garder seulement les 100 dernières mesures
        if len(self.performance_history[template_name]) > 100:
            self.performance_history[template_name] = self.performance_history[template_name][-100:]
    
    def get_average_performance(self, template_name: str) -> Optional[float]:
        """Retourne la performance moyenne d'un template"""
        if template_name not in self.performance_history:
            return None
        
        times = self.performance_history[template_name]
        return sum(times) / len(times) if times else None


class TemplateVersionManager:
    """
    Gestionnaire de versions pour les templates
    """
    
    def __init__(self):
        self.template_versions = {}
        self.migration_rules = {}
    
    def register_template_version(self, template: QueryTemplate):
        """Enregistre une version de template"""
        key = f"{template.intent_type}:{template.version}"
        self.template_versions[key] = template
    
    def get_template_version(self, intent_type: IntentType, version: str) -> Optional[QueryTemplate]:
        """Récupère une version spécifique d'un template"""
        key = f"{intent_type}:{version}"
        return self.template_versions.get(key)
    
    def get_latest_template(self, intent_type: IntentType) -> Optional[QueryTemplate]:
        """Récupère la dernière version d'un template"""
        # Recherche de la version la plus récente
        versions = [k for k in self.template_versions.keys() if k.startswith(f"{intent_type}:")]
        if not versions:
            return None
        
        # Tri par version (assume format x.y)
        latest_key = max(versions, key=lambda x: tuple(map(int, x.split(':')[1].split('.'))))
        return self.template_versions[latest_key]
    
    def migrate_contract(self, contract: SearchServiceQuery, target_version: str) -> SearchServiceQuery:
        """Migre un contrat vers une version de template cible"""
        # Cette méthode pourrait implémenter des règles de migration
        # Pour l'instant, retourne le contrat tel quel
        return contract


# === INSTANCES GLOBALES DES OPTIMISEURS ===
performance_optimizer = TemplatePerformanceOptimizer()
version_manager = TemplateVersionManager()


# === FONCTION PRINCIPALE D'ORCHESTRATION ===

def process_financial_query(contract: SearchServiceQuery) -> ESSearchQuery:
    """
    Fonction principale pour traiter une requête financière
    Combine template, validation, optimisation et construction
    """
    try:
        # 1. Validation du contrat
        if not validate_template_compatibility(contract):
            raise ValueError("Contrat incompatible avec les templates disponibles")
        
        # 2. Construction de la requête depuis le template
        es_query = build_query_from_intent(contract)
        
        # 3. Récupération du template pour optimisation
        template = get_template_by_intent(contract.query_metadata.intent_type)
        if template:
            # 4. Optimisation selon la catégorie de performance
            es_query = performance_optimizer.optimize_query(es_query, template)
        
        logger.info(f"Requête construite avec succès pour l'intention: {contract.query_metadata.intent_type}")
        return es_query
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
        # Fallback vers une requête générique
        return financial_template_builder._build_fallback_query(contract)


# === MÉTRIQUES ET MONITORING ===

class TemplateMetrics:
    """
    Collecteur de métriques pour les templates
    """
    
    def __init__(self):
        self.usage_stats = {}
        self.error_stats = {}
        self.performance_stats = {}
    
    def record_template_usage(self, intent_type: str):
        """Enregistre l'utilisation d'un template"""
        if intent_type not in self.usage_stats:
            self.usage_stats[intent_type] = 0
        self.usage_stats[intent_type] += 1
    
    def record_template_error(self, intent_type: str, error_type: str):
        """Enregistre une erreur de template"""
        key = f"{intent_type}:{error_type}"
        if key not in self.error_stats:
            self.error_stats[key] = 0
        self.error_stats[key] += 1
    
    def record_template_performance(self, intent_type: str, execution_time_ms: int):
        """Enregistre la performance d'un template"""
        if intent_type not in self.performance_stats:
            self.performance_stats[intent_type] = []
        
        self.performance_stats[intent_type].append(execution_time_ms)
        
        # Garder seulement les 1000 dernières mesures
        if len(self.performance_stats[intent_type]) > 1000:
            self.performance_stats[intent_type] = self.performance_stats[intent_type][-1000:]
    
    def get_most_used_templates(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Retourne les templates les plus utilisés"""
        sorted_usage = sorted(self.usage_stats.items(), key=lambda x: x[1], reverse=True)
        return sorted_usage[:limit]
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Retourne un résumé des performances par template"""
        summary = {}
        
        for intent_type, times in self.performance_stats.items():
            if times:
                summary[intent_type] = {
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "count": len(times)
                }
        
        return summary


# === INSTANCE GLOBALE DES MÉTRIQUES ===
template_metrics = TemplateMetrics()


# === HOOKS POUR INTÉGRATION ===

def before_query_execution(contract: SearchServiceQuery) -> None:
    """Hook appelé avant l'exécution d'une requête"""
    template_metrics.record_template_usage(contract.query_metadata.intent_type)


def after_query_execution(contract: SearchServiceQuery, execution_time_ms: int, success: bool) -> None:
    """Hook appelé après l'exécution d'une requête"""
    intent_type = contract.query_metadata.intent_type
    
    if success:
        template_metrics.record_template_performance(intent_type, execution_time_ms)
        performance_optimizer.record_performance(f"{intent_type}_template", execution_time_ms)
    else:
        template_metrics.record_template_error(intent_type, "execution_failed")


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # Classes principales
    "QueryTemplateEngine",
    "FinancialQueryTemplateBuilder", 
    "QueryTemplateValidator",
    "TemplatePerformanceOptimizer",
    "TemplateVersionManager",
    "TemplateMetrics",
    
    # Enums
    "IntentType",
    "TemplateParameterType",
    
    # Modèles
    "QueryTemplate",
    "TemplateParameter",
    
    # Fonctions utilitaires
    "build_query_from_intent",
    "get_available_templates",
    "get_template_by_intent", 
    "get_performance_estimate",
    "validate_template_compatibility",
    "process_financial_query",
    
    # Hooks
    "before_query_execution",
    "after_query_execution",
    
    # Instances globales
    "financial_template_builder",
    "template_validator",
    "performance_optimizer",
    "version_manager",
    "template_metrics"
]