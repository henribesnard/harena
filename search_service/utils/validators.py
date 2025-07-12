"""
Validateurs pour les requêtes du Search Service
Validation des contrats, requêtes Elasticsearch et paramètres
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, date
from decimal import Decimal, InvalidOperation

from models import (
    SearchServiceQuery,
    SearchServiceResponse,
    InternalSearchRequest,
    FilterSet,
    ValidatedFilter,
    FIELD_CONFIGURATIONS,
    QueryType,
    FilterOperator
)
from config import (
    settings,
    SUPPORTED_INTENT_TYPES,
    SUPPORTED_FILTER_OPERATORS,
    INDEXED_FIELDS
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception pour les erreurs de validation"""
    
    def __init__(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        self.message = message
        self.field = field
        self.code = code
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'erreur en dictionnaire pour API"""
        return {
            "error": "validation_error",
            "message": self.message,
            "field": self.field,
            "code": self.code
        }


class SecurityValidator:
    """Validateur pour les aspects sécurité des requêtes"""
    
    @staticmethod
    def validate_user_isolation(query: SearchServiceQuery) -> bool:
        """
        Valide que l'isolation utilisateur est respectée
        
        CRITIQUE: Chaque requête DOIT contenir un filtre user_id
        pour éviter les fuites de données entre utilisateurs
        """
        # Vérifier métadonnées
        if not query.query_metadata.user_id or query.query_metadata.user_id <= 0:
            raise ValidationError(
                "user_id must be positive in query metadata",
                field="query_metadata.user_id",
                code="INVALID_USER_ID"
            )
        
        # Vérifier filtres obligatoires
        user_filters = [
            f for f in query.filters.required 
            if f.field == "user_id"
        ]
        
        if not user_filters:
            raise ValidationError(
                "user_id filter is mandatory in required filters for security",
                field="filters.required",
                code="MISSING_USER_FILTER"
            )
        
        # Vérifier cohérence user_id entre métadonnées et filtres
        filter_user_id = user_filters[0].value
        metadata_user_id = query.query_metadata.user_id
        
        if filter_user_id != metadata_user_id:
            raise ValidationError(
                f"user_id mismatch: metadata={metadata_user_id}, filter={filter_user_id}",
                field="user_id",
                code="USER_ID_MISMATCH"
            )
        
        # Vérifier que user_id est en premier (optimisation)
        if query.filters.required[0].field != "user_id":
            logger.warning("user_id filter should be first for optimal performance")
        
        return True
    
    @staticmethod
    def validate_rate_limits(user_id: int, query_complexity: str) -> bool:
        """Valide que les limites de taux sont respectées"""
        # TODO: Implémenter rate limiting basé sur Redis
        # Pour l'instant, validation basique sur les limites
        
        if query_complexity == "complex":
            # Limites plus strictes pour requêtes complexes
            max_results = 50
        else:
            max_results = settings.max_results_per_query
        
        # Note: Cette validation sera étendue avec Redis rate limiting
        logger.debug(f"Rate limit check for user {user_id}: complexity={query_complexity}")
        return True
    
    @staticmethod
    def sanitize_text_input(text: str) -> str:
        """Nettoie et sécurise les entrées textuelles"""
        if not text:
            return ""
        
        # Supprimer caractères de contrôle
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Limiter la longueur
        max_length = settings.max_text_search_length
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text input truncated to {max_length} characters")
        
        # Échapper caractères dangereux pour Elasticsearch
        dangerous_chars = ['\\', '/', '"', "'", '<', '>', '&']
        for char in dangerous_chars:
            text = text.replace(char, ' ')
        
        return text.strip()


class ContractValidator:
    """Validateur pour les contrats SearchServiceQuery/Response"""
    
    @staticmethod
    def validate_search_query(query: SearchServiceQuery) -> Tuple[bool, List[str]]:
        """
        Valide complètement un contrat SearchServiceQuery
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # 1. Validation sécurité obligatoire
            SecurityValidator.validate_user_isolation(query)
            
            # 2. Validation métadonnées
            errors.extend(ContractValidator._validate_metadata(query.query_metadata))
            
            # 3. Validation paramètres de recherche
            errors.extend(ContractValidator._validate_search_parameters(query.search_parameters))
            
            # 4. Validation filtres
            errors.extend(ContractValidator._validate_filters(query.filters))
            
            # 5. Validation recherche textuelle si présente
            if query.text_search:
                errors.extend(ContractValidator._validate_text_search(query.text_search))
            
            # 6. Validation agrégations si présentes
            if query.aggregations:
                errors.extend(ContractValidator._validate_aggregations(query.aggregations))
            
            # 7. Validation options
            errors.extend(ContractValidator._validate_options(query.options))
            
            # 8. Validation cohérence globale
            errors.extend(ContractValidator._validate_query_consistency(query))
            
        except ValidationError as e:
            errors.append(e.message)
        except Exception as e:
            errors.append(f"Unexpected validation error: {e}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_metadata(metadata) -> List[str]:
        """Valide les métadonnées de requête"""
        errors = []
        
        # Validation intention
        if metadata.intent_type not in SUPPORTED_INTENT_TYPES:
            errors.append(f"Unsupported intent type: {metadata.intent_type}")
        
        # Validation confiance
        if not 0.0 <= metadata.confidence <= 1.0:
            errors.append(f"Confidence must be between 0.0 and 1.0, got {metadata.confidence}")
        
        # Validation agent
        if not metadata.agent_name or len(metadata.agent_name.strip()) == 0:
            errors.append("agent_name cannot be empty")
        
        # Validation timestamp
        if metadata.timestamp > datetime.utcnow():
            errors.append("timestamp cannot be in the future")
        
        return errors
    
    @staticmethod
    def _validate_search_parameters(params) -> List[str]:
        """Valide les paramètres de recherche"""
        errors = []
        
        # Validation type de requête
        try:
            QueryType(params.query_type)
        except ValueError:
            errors.append(f"Invalid query type: {params.query_type}")
        
        # Validation limites
        if params.limit < 1 or params.limit > settings.max_results_per_query:
            errors.append(f"limit must be between 1 and {settings.max_results_per_query}")
        
        if params.offset < 0 or params.offset > settings.max_pagination_offset:
            errors.append(f"offset must be between 0 and {settings.max_pagination_offset}")
        
        # Validation timeout
        if params.timeout_ms < 100 or params.timeout_ms > settings.max_query_timeout_ms:
            errors.append(f"timeout_ms must be between 100 and {settings.max_query_timeout_ms}")
        
        # Validation champs retournés
        if params.fields:
            invalid_fields = [
                field for field in params.fields 
                if field not in INDEXED_FIELDS
            ]
            if invalid_fields:
                errors.append(f"Invalid fields requested: {invalid_fields}")
        
        return errors
    
    @staticmethod
    def _validate_filters(filters) -> List[str]:
        """Valide les filtres de recherche"""
        errors = []
        
        # Validation filtres obligatoires
        for filter_obj in filters.required:
            filter_errors = ContractValidator._validate_single_filter(filter_obj)
            errors.extend(filter_errors)
        
        # Validation filtres optionnels
        for filter_obj in filters.optional:
            filter_errors = ContractValidator._validate_single_filter(filter_obj)
            errors.extend(filter_errors)
        
        # Validation filtres de plage
        for filter_obj in filters.ranges:
            filter_errors = ContractValidator._validate_single_filter(filter_obj)
            errors.extend(filter_errors)
        
        # Validation recherche textuelle dans filtres
        if filters.text_search:
            if not isinstance(filters.text_search, dict):
                errors.append("text_search in filters must be a dictionary")
            elif "query" not in filters.text_search:
                errors.append("text_search must contain 'query' field")
        
        return errors
    
    @staticmethod
    def _validate_single_filter(filter_obj) -> List[str]:
        """Valide un filtre individuel"""
        errors = []
        
        # Validation champ
        if filter_obj.field not in INDEXED_FIELDS:
            errors.append(f"Unknown filter field: {filter_obj.field}")
            return errors  # Pas la peine de continuer si le champ n'existe pas
        
        # Validation opérateur
        if filter_obj.operator not in SUPPORTED_FILTER_OPERATORS:
            errors.append(f"Unsupported filter operator: {filter_obj.operator}")
        
        # Validation valeur selon l'opérateur
        try:
            FilterValidator.validate_filter_value(
                filter_obj.field, 
                filter_obj.operator, 
                filter_obj.value
            )
        except ValidationError as e:
            errors.append(e.message)
        
        return errors
    
    @staticmethod
    def _validate_text_search(text_search) -> List[str]:
        """Valide la configuration de recherche textuelle"""
        errors = []
        
        # Validation requête
        if not text_search.query or len(text_search.query.strip()) == 0:
            errors.append("text_search query cannot be empty")
        
        if len(text_search.query) > settings.max_text_search_length:
            errors.append(f"text_search query too long (max {settings.max_text_search_length})")
        
        # Validation champs
        if not text_search.fields:
            errors.append("text_search fields cannot be empty")
        else:
            invalid_fields = [
                field for field in text_search.fields 
                if field not in INDEXED_FIELDS
            ]
            if invalid_fields:
                errors.append(f"Invalid text_search fields: {invalid_fields}")
        
        # Validation boost
        if text_search.boost:
            for field, boost_value in text_search.boost.items():
                if not isinstance(boost_value, (int, float)) or boost_value <= 0:
                    errors.append(f"Invalid boost value for field {field}: {boost_value}")
        
        # Validation fuzziness
        if text_search.fuzziness:
            valid_fuzziness = ["AUTO", "0", "1", "2"]
            if text_search.fuzziness not in valid_fuzziness:
                errors.append(f"Invalid fuzziness: {text_search.fuzziness}")
        
        return errors
    
    @staticmethod
    def _validate_aggregations(aggregations) -> List[str]:
        """Valide la configuration d'agrégations"""
        errors = []
        
        # Validation types d'agrégation
        from config import SUPPORTED_AGGREGATION_TYPES
        invalid_types = [
            agg_type for agg_type in aggregations.types 
            if agg_type not in SUPPORTED_AGGREGATION_TYPES
        ]
        if invalid_types:
            errors.append(f"Unsupported aggregation types: {invalid_types}")
        
        # Validation champs de groupement
        if aggregations.group_by:
            invalid_fields = [
                field for field in aggregations.group_by 
                if field not in INDEXED_FIELDS
            ]
            if invalid_fields:
                errors.append(f"Invalid group_by fields: {invalid_fields}")
        
        # Validation métriques
        if aggregations.metrics:
            invalid_metrics = [
                metric for metric in aggregations.metrics 
                if metric not in INDEXED_FIELDS
            ]
            if invalid_metrics:
                errors.append(f"Invalid aggregation metrics: {invalid_metrics}")
        
        # Validation taille terms
        if aggregations.terms_size > settings.max_aggregation_buckets:
            errors.append(f"terms_size too large (max {settings.max_aggregation_buckets})")
        
        return errors
    
    @staticmethod
    def _validate_options(options) -> List[str]:
        """Valide les options de recherche"""
        errors = []
        
        # Validation score minimum
        if options.min_score is not None:
            if not isinstance(options.min_score, (int, float)) or options.min_score < 0:
                errors.append("min_score must be a positive number")
        
        # Validation préférence
        if options.preference:
            valid_preferences = ["_local", "_primary", "_replica", "_only_local"]
            if not any(options.preference.startswith(pref) for pref in valid_preferences):
                if not re.match(r'^[a-zA-Z0-9_-]+$', options.preference):
                    errors.append("Invalid preference format")
        
        return errors
    
    @staticmethod
    def _validate_query_consistency(query: SearchServiceQuery) -> List[str]:
        """Valide la cohérence globale de la requête"""
        errors = []
        
        # Vérifier cohérence type de requête et configuration
        query_type = query.search_parameters.query_type
        
        if query_type in [QueryType.TEXT_SEARCH, QueryType.TEXT_SEARCH_WITH_FILTER]:
            if not query.text_search:
                errors.append(f"text_search required for query_type {query_type}")
        
        if query_type in [QueryType.FILTERED_AGGREGATION, QueryType.TEMPORAL_AGGREGATION]:
            if not query.aggregations or not query.aggregations.enabled:
                errors.append(f"aggregations required for query_type {query_type}")
        
        # Vérifier cohérence intention et configuration
        intent = query.query_metadata.intent_type
        
        if intent == "TEXT_SEARCH" and not query.text_search:
            errors.append("TEXT_SEARCH intent requires text_search configuration")
        
        if intent.startswith("AGGREGATE") and not query.aggregations:
            errors.append(f"{intent} intent requires aggregations configuration")
        
        return errors
    
    @staticmethod
    def validate_search_response(response: SearchServiceResponse) -> Tuple[bool, List[str]]:
        """Valide un contrat SearchServiceResponse"""
        errors = []
        
        try:
            # Validation métadonnées de réponse
            if response.response_metadata.execution_time_ms < 0:
                errors.append("execution_time_ms cannot be negative")
            
            if response.response_metadata.total_hits < 0:
                errors.append("total_hits cannot be negative")
            
            if response.response_metadata.returned_hits < 0:
                errors.append("returned_hits cannot be negative")
            
            # Cohérence entre résultats et métadonnées
            actual_results = len(response.results)
            if actual_results != response.response_metadata.returned_hits:
                errors.append(f"Results count mismatch: actual={actual_results}, metadata={response.response_metadata.returned_hits}")
            
            # Validation agrégations si présentes
            if response.aggregations:
                if response.aggregations.transaction_count < 0:
                    errors.append("aggregations.transaction_count cannot be negative")
            
            # Validation métriques de performance
            if response.performance.elasticsearch_took < 0:
                errors.append("elasticsearch_took cannot be negative")
            
            # Validation enrichissement contextuel
            if not 0.0 <= response.context_enrichment.result_quality_score <= 1.0:
                errors.append("result_quality_score must be between 0.0 and 1.0")
            
        except Exception as e:
            errors.append(f"Response validation error: {e}")
        
        return len(errors) == 0, errors


class FilterValidator:
    """Validateur spécialisé pour les filtres"""
    
    @staticmethod
    def validate_filter_value(field: str, operator: str, value: Any) -> bool:
        """
        Valide une valeur de filtre selon le champ et l'opérateur
        
        Args:
            field: Nom du champ
            operator: Opérateur de filtrage
            value: Valeur à valider
            
        Returns:
            True si valide
            
        Raises:
            ValidationError: Si validation échoue
        """
        if field not in FIELD_CONFIGURATIONS:
            raise ValidationError(f"Unknown field: {field}")
        
        field_config = FIELD_CONFIGURATIONS[field]
        
        # Validation selon le type de champ
        if field_config.field_type.value == "integer":
            FilterValidator._validate_integer_value(field, operator, value)
        elif field_config.field_type.value == "float":
            FilterValidator._validate_float_value(field, operator, value)
        elif field_config.field_type.value == "date":
            FilterValidator._validate_date_value(field, operator, value)
        elif field_config.field_type.value == "keyword":
            FilterValidator._validate_keyword_value(field, operator, value)
        elif field_config.field_type.value == "text":
            FilterValidator._validate_text_value(field, operator, value)
        elif field_config.field_type.value == "boolean":
            FilterValidator._validate_boolean_value(field, operator, value)
        
        # Validation des valeurs autorisées
        if field_config.allowed_values and value not in field_config.allowed_values:
            raise ValidationError(
                f"Value '{value}' not allowed for field '{field}'. "
                f"Allowed values: {field_config.allowed_values}"
            )
        
        # Validation pattern si défini
        if field_config.validation_pattern and isinstance(value, str):
            if not re.match(field_config.validation_pattern, value):
                raise ValidationError(
                    f"Value '{value}' does not match required pattern for field '{field}'"
                )
        
        return True
    
    @staticmethod
    def _validate_integer_value(field: str, operator: str, value: Any):
        """Valide une valeur entière"""
        if operator in ["in", "not_in"]:
            if not isinstance(value, list):
                raise ValidationError(f"Operator {operator} requires a list for field {field}")
            for item in value:
                if not isinstance(item, int):
                    raise ValidationError(f"All values must be integers for field {field}")
        elif operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise ValidationError(f"between operator requires list of 2 integers for field {field}")
            if not all(isinstance(v, int) for v in value):
                raise ValidationError(f"between values must be integers for field {field}")
            if value[0] > value[1]:
                raise ValidationError(f"between range invalid: {value[0]} > {value[1]} for field {field}")
        elif operator == "exists":
            if not isinstance(value, bool):
                raise ValidationError(f"exists operator requires boolean for field {field}")
        else:
            if not isinstance(value, int):
                raise ValidationError(f"Value must be integer for field {field}")
    
    @staticmethod
    def _validate_float_value(field: str, operator: str, value: Any):
        """Valide une valeur décimale"""
        if operator in ["in", "not_in"]:
            if not isinstance(value, list):
                raise ValidationError(f"Operator {operator} requires a list for field {field}")
            for item in value:
                if not isinstance(item, (int, float)):
                    raise ValidationError(f"All values must be numbers for field {field}")
        elif operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise ValidationError(f"between operator requires list of 2 numbers for field {field}")
            if not all(isinstance(v, (int, float)) for v in value):
                raise ValidationError(f"between values must be numbers for field {field}")
            if value[0] > value[1]:
                raise ValidationError(f"between range invalid: {value[0]} > {value[1]} for field {field}")
        else:
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Value must be number for field {field}")
            
            # Validation spéciale pour les montants
            if field in ["amount", "amount_abs"]:
                if abs(value) > 1000000:  # 1M limite
                    raise ValidationError(f"Amount too large for field {field}: {value}")
    
    @staticmethod
    def _validate_date_value(field: str, operator: str, value: Any):
        """Valide une valeur de date"""
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        
        if operator in ["in", "not_in"]:
            if not isinstance(value, list):
                raise ValidationError(f"Operator {operator} requires a list for field {field}")
            for item in value:
                if not isinstance(item, str) or not re.match(date_pattern, item):
                    raise ValidationError(f"Invalid date format for field {field}: {item}")
        elif operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise ValidationError(f"between operator requires list of 2 dates for field {field}")
            for date_str in value:
                if not isinstance(date_str, str) or not re.match(date_pattern, date_str):
                    raise ValidationError(f"Invalid date format for field {field}: {date_str}")
            if value[0] > value[1]:
                raise ValidationError(f"Date range invalid: {value[0]} > {value[1]} for field {field}")
        else:
            if not isinstance(value, str) or not re.match(date_pattern, value):
                raise ValidationError(f"Invalid date format for field {field}: {value}")
    
    @staticmethod
    def _validate_keyword_value(field: str, operator: str, value: Any):
        """Valide une valeur keyword"""
        if operator in ["in", "not_in"]:
            if not isinstance(value, list):
                raise ValidationError(f"Operator {operator} requires a list for field {field}")
            for item in value:
                if not isinstance(item, str):
                    raise ValidationError(f"All values must be strings for field {field}")
        elif operator == "exists":
            if not isinstance(value, bool):
                raise ValidationError(f"exists operator requires boolean for field {field}")
        else:
            if not isinstance(value, str):
                raise ValidationError(f"Value must be string for field {field}")
            
            # Validation longueur
            if len(value) > 500:
                raise ValidationError(f"String too long for field {field}: {len(value)} characters")
    
    @staticmethod
    def _validate_text_value(field: str, operator: str, value: Any):
        """Valide une valeur textuelle"""
        if operator == "match":
            if not isinstance(value, str):
                raise ValidationError(f"match operator requires string for field {field}")
            if len(value.strip()) == 0:
                raise ValidationError(f"match query cannot be empty for field {field}")
        else:
            FilterValidator._validate_keyword_value(field, operator, value)
    
    @staticmethod
    def _validate_boolean_value(field: str, operator: str, value: Any):
        """Valide une valeur booléenne"""
        if operator != "eq":
            raise ValidationError(f"Boolean field {field} only supports 'eq' operator")
        
        if not isinstance(value, bool):
            raise ValidationError(f"Value must be boolean for field {field}")


class ElasticsearchQueryValidator:
    """Validateur pour les requêtes Elasticsearch brutes"""
    
    @staticmethod
    def validate_query_body(body: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Valide une requête Elasticsearch complète
        
        Args:
            body: Corps de la requête Elasticsearch
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, errors)
        """
        errors = []
        
        if not isinstance(body, dict):
            return False, ["Query body must be a dictionary"]
        
        # Validation structure de base
        if "query" not in body:
            errors.append("Query body must contain 'query' field")
        else:
            query_errors = ElasticsearchQueryValidator._validate_query_clause(body["query"])
            errors.extend(query_errors)
        
        # Validation paramètres
        if "size" in body:
            if not isinstance(body["size"], int) or body["size"] < 0:
                errors.append("size must be a non-negative integer")
            elif body["size"] > settings.max_results_per_query:
                errors.append(f"size too large (max {settings.max_results_per_query})")
        
        if "from" in body:
            if not isinstance(body["from"], int) or body["from"] < 0:
                errors.append("from must be a non-negative integer")
            elif body["from"] > settings.max_pagination_offset:
                errors.append(f"from too large (max {settings.max_pagination_offset})")
        
        # Validation agrégations
        if "aggs" in body:
            agg_errors = ElasticsearchQueryValidator._validate_aggregations(body["aggs"])
            errors.extend(agg_errors)
        
        # Validation tri
        if "sort" in body:
            sort_errors = ElasticsearchQueryValidator._validate_sort(body["sort"])
            errors.extend(sort_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_query_clause(query: Dict[str, Any]) -> List[str]:
        """Valide une clause query Elasticsearch"""
        errors = []
        
        if not isinstance(query, dict):
            return ["Query clause must be a dictionary"]
        
        # Types de requêtes supportés
        supported_query_types = [
            "bool", "match", "match_all", "term", "terms", "range",
            "wildcard", "prefix", "multi_match", "simple_query_string",
            "match_phrase", "exists"
        ]
        
        query_keys = list(query.keys())
        if len(query_keys) != 1:
            errors.append("Query clause must contain exactly one query type")
            return errors
        
        query_type = query_keys[0]
        if query_type not in supported_query_types:
            errors.append(f"Unsupported query type: {query_type}")
            return errors
        
        # Validation spécifique selon le type
        if query_type == "bool":
            errors.extend(ElasticsearchQueryValidator._validate_bool_query(query["bool"]))
        elif query_type == "range":
            errors.extend(ElasticsearchQueryValidator._validate_range_query(query["range"]))
        elif query_type == "terms":
            errors.extend(ElasticsearchQueryValidator._validate_terms_query(query["terms"]))
        
        return errors
    
    @staticmethod
    def _validate_bool_query(bool_query: Dict[str, Any]) -> List[str]:
        """Valide une requête bool Elasticsearch"""
        errors = []
        
        valid_clauses = ["must", "filter", "should", "must_not"]
        
        for clause_name, clause_value in bool_query.items():
            if clause_name not in valid_clauses:
                if clause_name != "minimum_should_match":  # Exception autorisée
                    errors.append(f"Invalid bool clause: {clause_name}")
                continue
            
            if not isinstance(clause_value, list):
                errors.append(f"Bool clause {clause_name} must be a list")
                continue
            
            # Valider chaque sous-requête
            for sub_query in clause_value:
                sub_errors = ElasticsearchQueryValidator._validate_query_clause(sub_query)
                errors.extend(sub_errors)
        
        return errors
    
    @staticmethod
    def _validate_range_query(range_query: Dict[str, Any]) -> List[str]:
        """Valide une requête range Elasticsearch"""
        errors = []
        
        if len(range_query) != 1:
            errors.append("Range query must contain exactly one field")
            return errors
        
        field_name = list(range_query.keys())[0]
        range_conditions = range_query[field_name]
        
        if not isinstance(range_conditions, dict):
            errors.append("Range conditions must be a dictionary")
            return errors
        
        valid_operators = ["gte", "gt", "lte", "lt"]
        for operator in range_conditions.keys():
            if operator not in valid_operators:
                errors.append(f"Invalid range operator: {operator}")
        
        return errors
    
    @staticmethod
    def _validate_terms_query(terms_query: Dict[str, Any]) -> List[str]:
        """Valide une requête terms Elasticsearch"""
        errors = []
        
        if len(terms_query) != 1:
            errors.append("Terms query must contain exactly one field")
            return errors
        
        field_name = list(terms_query.keys())[0]
        terms_values = terms_query[field_name]
        
        if not isinstance(terms_values, list):
            errors.append("Terms values must be a list")
            return errors
        
        if len(terms_values) == 0:
            errors.append("Terms list cannot be empty")
        elif len(terms_values) > 1000:  # Limite Elasticsearch
            errors.append("Too many terms (max 1000)")
        
        return errors
    
    @staticmethod
    def _validate_aggregations(aggs: Dict[str, Any]) -> List[str]:
        """Valide les agrégations Elasticsearch"""
        errors = []
        
        if not isinstance(aggs, dict):
            return ["Aggregations must be a dictionary"]
        
        for agg_name, agg_config in aggs.items():
            if not isinstance(agg_config, dict):
                errors.append(f"Aggregation {agg_name} config must be a dictionary")
                continue
            
            # Vérifier qu'il y a au moins un type d'agrégation
            agg_types = ["terms", "range", "date_histogram", "histogram", "sum", "avg", "min", "max", "cardinality"]
            found_type = False
            
            for agg_type in agg_types:
                if agg_type in agg_config:
                    found_type = True
                    break
            
            if not found_type:
                errors.append(f"Aggregation {agg_name} must contain a valid aggregation type")
        
        return errors
    
    @staticmethod
    def _validate_sort(sort: Union[List, Dict]) -> List[str]:
        """Valide les critères de tri Elasticsearch"""
        errors = []
        
        if isinstance(sort, dict):
            sort = [sort]
        elif not isinstance(sort, list):
            return ["Sort must be a list or dictionary"]
        
        for sort_item in sort:
            if not isinstance(sort_item, dict):
                errors.append("Each sort item must be a dictionary")
                continue
            
            if len(sort_item) != 1:
                errors.append("Each sort item must contain exactly one field")
                continue
            
            field_name = list(sort_item.keys())[0]
            sort_config = sort_item[field_name]
            
            if isinstance(sort_config, str):
                if sort_config not in ["asc", "desc"]:
                    errors.append(f"Invalid sort order: {sort_config}")
            elif isinstance(sort_config, dict):
                if "order" in sort_config:
                    if sort_config["order"] not in ["asc", "desc"]:
                        errors.append(f"Invalid sort order: {sort_config['order']}")
            else:
                errors.append("Sort config must be string or dictionary")
        
        return errors


class PerformanceValidator:
    """Validateur pour les aspects performance des requêtes"""
    
    @staticmethod
    def assess_query_complexity(query: SearchServiceQuery) -> str:
        """
        Évalue la complexité d'une requête pour optimisation
        
        Returns:
            str: "simple", "medium", "complex"
        """
        complexity_score = 0
        
        # Score basé sur les filtres
        total_filters = (
            len(query.filters.required) + 
            len(query.filters.optional) + 
            len(query.filters.ranges)
        )
        complexity_score += total_filters
        
        # Score basé sur la recherche textuelle
        if query.text_search:
            complexity_score += 2
            if len(query.text_search.fields) > 3:
                complexity_score += 1
            if query.text_search.fuzziness:
                complexity_score += 1
        
        # Score basé sur les agrégations
        if query.aggregations and query.aggregations.enabled:
            complexity_score += len(query.aggregations.types) * 2
            if len(query.aggregations.group_by) > 1:
                complexity_score += 2
        
        # Score basé sur la pagination
        if query.search_parameters.offset > 1000:
            complexity_score += 3
        
        # Score basé sur les options avancées
        if query.options.include_explanation:
            complexity_score += 1
        if query.options.include_highlights:
            complexity_score += 1
        
        # Classification finale
        if complexity_score <= 3:
            return "simple"
        elif complexity_score <= 8:
            return "medium"
        else:
            return "complex"
    
    @staticmethod
    def estimate_execution_time(query: SearchServiceQuery) -> int:
        """
        Estime le temps d'exécution d'une requête en millisecondes
        
        Returns:
            int: Temps estimé en ms
        """
        base_time = 20  # 20ms de base
        
        # Temps pour les filtres
        total_filters = (
            len(query.filters.required) + 
            len(query.filters.optional) + 
            len(query.filters.ranges)
        )
        base_time += total_filters * 5
        
        # Temps pour la recherche textuelle
        if query.text_search:
            base_time += 30
            base_time += len(query.text_search.fields) * 10
            if query.text_search.fuzziness:
                base_time += 20
        
        # Temps pour les agrégations
        if query.aggregations and query.aggregations.enabled:
            base_time += len(query.aggregations.types) * 25
            base_time += len(query.aggregations.group_by) * 15
        
        # Temps pour la pagination profonde
        if query.search_parameters.offset > 1000:
            base_time += 50
        
        # Temps pour les highlights
        if query.options.include_highlights:
            base_time += 15
        
        return min(base_time, 10000)  # Cap à 10s
    
    @staticmethod
    def validate_performance_limits(query: SearchServiceQuery) -> Tuple[bool, List[str]]:
        """
        Valide que la requête respecte les limites de performance
        
        Returns:
            Tuple[bool, List[str]]: (is_acceptable, warnings)
        """
        warnings = []
        
        # Vérifier complexité
        complexity = PerformanceValidator.assess_query_complexity(query)
        if complexity == "complex":
            warnings.append("Query complexity is high - consider simplifying")
        
        # Vérifier temps estimé
        estimated_time = PerformanceValidator.estimate_execution_time(query)
        if estimated_time > 5000:  # 5s
            warnings.append(f"Estimated execution time high: {estimated_time}ms")
        
        # Vérifier pagination profonde
        if query.search_parameters.offset > 5000:
            warnings.append("Deep pagination detected - consider using search_after")
        
        # Vérifier nombre de champs dans recherche textuelle
        if query.text_search and len(query.text_search.fields) > 5:
            warnings.append("Too many fields in text search - may impact performance")
        
        # Vérifier agrégations multiples
        if (query.aggregations and query.aggregations.enabled and 
            len(query.aggregations.types) > 3):
            warnings.append("Multiple aggregations may impact performance")
        
        # Une requête est acceptable si elle ne dépasse pas les limites critiques
        is_acceptable = (
            estimated_time <= 10000 and  # 10s max
            query.search_parameters.offset <= settings.max_pagination_offset
        )
        
        return is_acceptable, warnings


class BatchValidator:
    """Validateur pour les requêtes en lot"""
    
    @staticmethod
    def validate_batch_request(queries: List[SearchServiceQuery]) -> Tuple[bool, List[str]]:
        """
        Valide un lot de requêtes
        
        Args:
            queries: Liste des requêtes à valider
            
        Returns:
            Tuple[bool, List[str]]: (all_valid, errors)
        """
        errors = []
        
        # Validation de base
        if not queries:
            return False, ["Batch cannot be empty"]
        
        if len(queries) > 10:  # Limite arbitraire
            errors.append("Batch too large (max 10 queries)")
        
        # Validation de chaque requête
        for i, query in enumerate(queries):
            try:
                is_valid, query_errors = ContractValidator.validate_search_query(query)
                if not is_valid:
                    errors.extend([f"Query {i}: {error}" for error in query_errors])
            except Exception as e:
                errors.append(f"Query {i}: Validation failed - {e}")
        
        # Validation cohérence du lot
        user_ids = set(query.query_metadata.user_id for query in queries)
        if len(user_ids) > 1:
            errors.append("All queries in batch must be for the same user")
        
        # Estimation charge totale
        total_complexity = sum(
            1 if PerformanceValidator.assess_query_complexity(query) == "simple" else
            2 if PerformanceValidator.assess_query_complexity(query) == "medium" else 3
            for query in queries
        )
        
        if total_complexity > 20:  # Limite arbitraire
            errors.append("Batch complexity too high")
        
        return len(errors) == 0, errors


# === FACTORY DE VALIDATEURS ===

class ValidatorFactory:
    """Factory pour créer des validateurs spécialisés"""
    
    @staticmethod
    def create_contract_validator() -> ContractValidator:
        """Crée un validateur de contrats"""
        return ContractValidator()
    
    @staticmethod
    def create_security_validator() -> SecurityValidator:
        """Crée un validateur de sécurité"""
        return SecurityValidator()
    
    @staticmethod
    def create_filter_validator() -> FilterValidator:
        """Crée un validateur de filtres"""
        return FilterValidator()
    
    @staticmethod
    def create_elasticsearch_validator() -> ElasticsearchQueryValidator:
        """Crée un validateur Elasticsearch"""
        return ElasticsearchQueryValidator()
    
    @staticmethod
    def create_performance_validator() -> PerformanceValidator:
        """Crée un validateur de performance"""
        return PerformanceValidator()
    
    @staticmethod
    def validate_complete_request(query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Validation complète d'une requête avec tous les validateurs
        
        Returns:
            Dict avec résultats de validation détaillés
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "security_check": {"passed": False, "errors": []},
            "contract_check": {"passed": False, "errors": []},
            "performance_check": {"passed": False, "warnings": [], "complexity": "unknown"},
            "estimated_time_ms": 0
        }
        
        try:
            # 1. Validation sécurité
            try:
                SecurityValidator.validate_user_isolation(query)
                validation_result["security_check"]["passed"] = True
            except ValidationError as e:
                validation_result["security_check"]["errors"].append(e.message)
                validation_result["errors"].append(f"Security: {e.message}")
                validation_result["valid"] = False
            
            # 2. Validation contrat
            contract_valid, contract_errors = ContractValidator.validate_search_query(query)
            validation_result["contract_check"]["passed"] = contract_valid
            validation_result["contract_check"]["errors"] = contract_errors
            if not contract_valid:
                validation_result["errors"].extend([f"Contract: {error}" for error in contract_errors])
                validation_result["valid"] = False
            
            # 3. Validation performance
            complexity = PerformanceValidator.assess_query_complexity(query)
            estimated_time = PerformanceValidator.estimate_execution_time(query)
            perf_acceptable, perf_warnings = PerformanceValidator.validate_performance_limits(query)
            
            validation_result["performance_check"].update({
                "passed": perf_acceptable,
                "warnings": perf_warnings,
                "complexity": complexity
            })
            validation_result["estimated_time_ms"] = estimated_time
            validation_result["warnings"].extend(perf_warnings)
            
            if not perf_acceptable:
                validation_result["valid"] = False
                validation_result["errors"].append("Performance limits exceeded")
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation exception: {e}")
            logger.error(f"Validation failed with exception: {e}")
        
        return validation_result


# === UTILITAIRES DE VALIDATION ===

def sanitize_query_string(query: str) -> str:
    """Nettoie une chaîne de requête pour Elasticsearch"""
    if not query:
        return ""
    
    # Utiliser le sanitizer de sécurité
    return SecurityValidator.sanitize_text_input(query)


def is_valid_user_id(user_id: Any) -> bool:
    """Vérifie si un user_id est valide"""
    return isinstance(user_id, int) and user_id > 0


def get_field_type(field_name: str) -> Optional[str]:
    """Retourne le type d'un champ indexé"""
    if field_name in FIELD_CONFIGURATIONS:
        return FIELD_CONFIGURATIONS[field_name].field_type.value
    return None


def validate_query_timeout(timeout_ms: int) -> bool:
    """Valide un timeout de requête"""
    return (
        isinstance(timeout_ms, int) and
        100 <= timeout_ms <= settings.max_query_timeout_ms
    )


def estimate_result_size(query: SearchServiceQuery) -> int:
    """Estime la taille des résultats d'une requête"""
    base_size = query.search_parameters.limit * 1024  # 1KB par résultat estimé
    
    # Ajouter taille pour highlights
    if query.options.include_highlights:
        base_size += query.search_parameters.limit * 512
    
    # Ajouter taille pour agrégations
    if query.aggregations and query.aggregations.enabled:
        base_size += len(query.aggregations.types) * 2048
    
    return base_size


# === EXPORTS ===

__all__ = [
    # Exceptions
    "ValidationError",
    
    # Validateurs principaux
    "SecurityValidator",
    "ContractValidator", 
    "FilterValidator",
    "ElasticsearchQueryValidator",
    "PerformanceValidator",
    "BatchValidator",
    
    # Factory
    "ValidatorFactory",
    
    # Utilitaires
    "sanitize_query_string",
    "is_valid_user_id",
    "get_field_type",
    "validate_query_timeout",
    "estimate_result_size"
]