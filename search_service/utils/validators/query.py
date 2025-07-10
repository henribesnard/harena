"""
Validateur de requêtes Elasticsearch pour le Search Service.

Ce module fournit la validation complète des requêtes Elasticsearch,
avec vérification de la structure, sécurité et performance.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union

from .base import (
    BaseValidator, ValidationResult, QueryComplexity, ValidationLevel,
    QueryValidationError, SecurityValidationError
)
from .config import (
    ALLOWED_QUERY_TYPES, DANGEROUS_QUERY_TYPES, ALLOWED_SEARCH_FIELDS,
    VALIDATION_LIMITS, VALIDATION_CONFIG, ERROR_MESSAGES
)

logger = logging.getLogger(__name__)

class QueryValidator(BaseValidator):
    """
    Validateur spécialisé pour les requêtes Elasticsearch.
    
    Valide la structure, la sécurité et la performance des requêtes
    avant leur exécution sur Elasticsearch.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        self.config = VALIDATION_CONFIG[validation_level.value]
    
    def validate(self, query: Dict[str, Any], **kwargs) -> ValidationResult:
        """
        Valide une requête Elasticsearch complète.
        
        Args:
            query: Requête Elasticsearch à valider
            **kwargs: Options supplémentaires
            
        Returns:
            ValidationResult avec détails de validation
        """
        start_time = time.time()
        result = self._create_result()
        
        try:
            # Validation de base
            if not self._validate_basic_structure(query, result):
                return result
            
            # Validation de la requête principale
            if "query" in query:
                self._validate_query_body(query["query"], result)
            
            # Validation des paramètres
            self._validate_query_parameters(query, result)
            
            # Validation de la complexité
            if self.config.get("check_complexity", False):
                self._validate_query_complexity(query, result)
            
            # Validation de sécurité
            if self.config.get("check_security_patterns", False):
                self._validate_security(query, result)
            
            # Sanitisation si demandée
            if self.config.get("sanitize_input", False):
                result.sanitized_data = self._sanitize_query(query)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation: {e}")
            result.add_error(f"Erreur interne de validation: {str(e)}")
        
        result.validation_time_ms = self._measure_validation_time(start_time)
        return result
    
    def validate_search_query(self, query_body: Dict[str, Any]) -> ValidationResult:
        """
        Valide spécifiquement le corps d'une requête de recherche.
        
        Args:
            query_body: Corps de la requête Elasticsearch
            
        Returns:
            ValidationResult
        """
        return self.validate({"query": query_body})
    
    def _validate_basic_structure(self, query: Dict[str, Any], result: ValidationResult) -> bool:
        """Valide la structure de base de la requête."""
        if not isinstance(query, dict):
            result.add_error("La requête doit être un dictionnaire")
            return False
        
        if not query:
            result.add_error("La requête ne peut pas être vide")
            return False
        
        return True
    
    def _validate_query_body(self, query_body: Dict[str, Any], result: ValidationResult):
        """Valide le corps de la requête."""
        if not isinstance(query_body, dict):
            result.add_error("Le corps de la requête doit être un dictionnaire", "query")
            return
        
        # Validation récursive des types de requêtes
        self._validate_query_types(query_body, result, path="query")
    
    def _validate_query_types(self, query_node: Dict[str, Any], result: ValidationResult, 
                            path: str = "", depth: int = 0):
        """Valide récursivement les types de requêtes."""
        if depth > VALIDATION_LIMITS["max_nested_depth"]:
            result.add_error(f"Profondeur de requête trop importante (max: {VALIDATION_LIMITS['max_nested_depth']})", path)
            return
        
        for query_type, query_params in query_node.items():
            current_path = f"{path}.{query_type}" if path else query_type
            
            # Vérification du type de requête
            if query_type not in ALLOWED_QUERY_TYPES:
                if query_type in DANGEROUS_QUERY_TYPES:
                    result.add_error(f"Type de requête dangereux non autorisé: {query_type}", current_path)
                else:
                    result.add_warning(f"Type de requête non reconnu: {query_type}", current_path)
            
            # Validation spécifique par type
            if query_type == "bool":
                self._validate_bool_query(query_params, result, current_path, depth)
            elif query_type == "match":
                self._validate_match_query(query_params, result, current_path)
            elif query_type == "multi_match":
                self._validate_multi_match_query(query_params, result, current_path)
            elif query_type == "term":
                self._validate_term_query(query_params, result, current_path)
            elif query_type == "terms":
                self._validate_terms_query(query_params, result, current_path)
            elif query_type == "range":
                self._validate_range_query(query_params, result, current_path)
            elif query_type == "wildcard":
                self._validate_wildcard_query(query_params, result, current_path)
            elif query_type == "regexp":
                self._validate_regexp_query(query_params, result, current_path)
            elif query_type == "function_score":
                self._validate_function_score_query(query_params, result, current_path, depth)
    
    def _validate_bool_query(self, params: Dict[str, Any], result: ValidationResult, 
                           path: str, depth: int):
        """Valide une requête bool."""
        if not isinstance(params, dict):
            result.add_error("Les paramètres bool doivent être un dictionnaire", path)
            return
        
        bool_clauses = 0
        for clause_type in ["must", "should", "must_not", "filter"]:
            if clause_type in params:
                clauses = params[clause_type]
                if isinstance(clauses, list):
                    bool_clauses += len(clauses)
                    for i, clause in enumerate(clauses):
                        if isinstance(clause, dict):
                            self._validate_query_types(clause, result, f"{path}.{clause_type}[{i}]", depth + 1)
                elif isinstance(clauses, dict):
                    bool_clauses += 1
                    self._validate_query_types(clauses, result, f"{path}.{clause_type}", depth + 1)
        
        if bool_clauses > VALIDATION_LIMITS["max_bool_clauses"]:
            result.add_error(f"Trop de clauses bool (max: {VALIDATION_LIMITS['max_bool_clauses']})", path)
    
    def _validate_match_query(self, params: Dict[str, Any], result: ValidationResult, path: str):
        """Valide une requête match."""
        if not isinstance(params, dict):
            result.add_error("Les paramètres match doivent être un dictionnaire", path)
            return
        
        for field, query_params in params.items():
            if field not in ALLOWED_SEARCH_FIELDS:
                result.add_warning(f"Champ non autorisé pour la recherche: {field}", path)
            
            if isinstance(query_params, dict) and "query" in query_params:
                query_text = query_params["query"]
                if isinstance(query_text, str):
                    self._validate_query_text(query_text, result, f"{path}.{field}")
    
    def _validate_multi_match_query(self, params: Dict[str, Any], result: ValidationResult, path: str):
        """Valide une requête multi_match."""
        if not isinstance(params, dict):
            result.add_error("Les paramètres multi_match doivent être un dictionnaire", path)
            return
        
        if "query" in params:
            query_text = params["query"]
            if isinstance(query_text, str):
                self._validate_query_text(query_text, result, f"{path}.query")
        
        if "fields" in params:
            fields = params["fields"]
            if isinstance(fields, list):
                for field in fields:
                    field_name = field.split("^")[0]  # Enlève le boost
                    if field_name not in ALLOWED_SEARCH_FIELDS:
                        result.add_warning(f"Champ non autorisé: {field_name}", path)
    
    def _validate_term_query(self, params: Dict[str, Any], result: ValidationResult, path: str):
        """Valide une requête term."""
        if not isinstance(params, dict):
            result.add_error("Les paramètres term doivent être un dictionnaire", path)
            return
        
        for field in params.keys():
            if field not in ALLOWED_SEARCH_FIELDS:
                result.add_warning(f"Champ non autorisé: {field}", path)
    
    def _validate_terms_query(self, params: Dict[str, Any], result: ValidationResult, path: str):
        """Valide une requête terms."""
        if not isinstance(params, dict):
            result.add_error("Les paramètres terms doivent être un dictionnaire", path)
            return
        
        for field, values in params.items():
            if field not in ALLOWED_SEARCH_FIELDS:
                result.add_warning(f"Champ non autorisé: {field}", path)
            
            if isinstance(values, list) and len(values) > VALIDATION_LIMITS["max_filter_values"]:
                result.add_error(f"Trop de valeurs terms (max: {VALIDATION_LIMITS['max_filter_values']})", path)
    
    def _validate_range_query(self, params: Dict[str, Any], result: ValidationResult, path: str):
        """Valide une requête range."""
        if not isinstance(params, dict):
            result.add_error("Les paramètres range doivent être un dictionnaire", path)
            return
        
        for field, range_params in params.items():
            if field not in ALLOWED_SEARCH_FIELDS:
                result.add_warning(f"Champ non autorisé: {field}", path)
            
            if isinstance(range_params, dict):
                valid_operators = {"gte", "gt", "lte", "lt", "from", "to"}
                for operator in range_params.keys():
                    if operator not in valid_operators:
                        result.add_warning(f"Opérateur range non standard: {operator}", path)
    
    def _validate_wildcard_query(self, params: Dict[str, Any], result: ValidationResult, path: str):
        """Valide une requête wildcard."""
        if not isinstance(params, dict):
            result.add_error("Les paramètres wildcard doivent être un dictionnaire", path)
            return
        
        for field, query_params in params.items():
            if field not in ALLOWED_SEARCH_FIELDS:
                result.add_warning(f"Champ non autorisé: {field}", path)
            
            # Comptage des wildcards pour performance
            if isinstance(query_params, dict) and "value" in query_params:
                value = query_params["value"]
                if isinstance(value, str) and value.count("*") > 5:
                    result.add_warning("Trop de wildcards peuvent impacter les performances", path)
    
    def _validate_regexp_query(self, params: Dict[str, Any], result: ValidationResult, path: str):
        """Valide une requête regexp."""
        if not isinstance(params, dict):
            result.add_error("Les paramètres regexp doivent être un dictionnaire", path)
            return
        
        for field in params.keys():
            if field not in ALLOWED_SEARCH_FIELDS:
                result.add_warning(f"Champ non autorisé: {field}", path)
        
        # Les requêtes regexp peuvent être coûteuses
        result.add_warning("Les requêtes regexp peuvent impacter les performances", path)
    
    def _validate_function_score_query(self, params: Dict[str, Any], result: ValidationResult, 
                                     path: str, depth: int):
        """Valide une requête function_score."""
        if not isinstance(params, dict):
            result.add_error("Les paramètres function_score doivent être un dictionnaire", path)
            return
        
        if "query" in params:
            self._validate_query_types(params["query"], result, f"{path}.query", depth + 1)
        
        if "functions" in params and isinstance(params["functions"], list):
            if len(params["functions"]) > 10:
                result.add_warning("Trop de fonctions de scoring peuvent impacter les performances", path)
    
    def _validate_query_parameters(self, query: Dict[str, Any], result: ValidationResult):
        """Valide les paramètres de la requête."""
        # Validation size
        if "size" in query:
            size = query["size"]
            if not isinstance(size, int) or size < 0:
                result.add_error("'size' doit être un entier positif")
            elif size > VALIDATION_LIMITS["max_results_limit"]:
                result.add_error(f"'size' trop important (max: {VALIDATION_LIMITS['max_results_limit']})")
        
        # Validation from
        if "from" in query:
            from_value = query["from"]
            if not isinstance(from_value, int) or from_value < 0:
                result.add_error("'from' doit être un entier positif")
            elif from_value > 10000:
                result.add_error("'from' trop important (max: 10000)")
        
        # Validation timeout
        if "timeout" in query:
            timeout = query["timeout"]
            if isinstance(timeout, str):
                if not timeout.endswith(("ms", "s")):
                    result.add_error("Format timeout invalide (doit finir par 'ms' ou 's')")
            elif isinstance(timeout, int):
                if timeout > VALIDATION_LIMITS["max_timeout_ms"]:
                    result.add_error(f"Timeout trop important (max: {VALIDATION_LIMITS['max_timeout_ms']}ms)")
    
    def _validate_query_complexity(self, query: Dict[str, Any], result: ValidationResult):
        """Valide la complexité de la requête."""
        complexity = self._calculate_complexity(query)
        
        if complexity.is_complex:
            result.add_warning(f"Requête complexe détectée (score: {complexity.score})")
            if complexity.score > 200:
                result.add_error("Requête trop complexe pour être exécutée")
    
    def _calculate_complexity(self, query: Dict[str, Any]) -> QueryComplexity:
        """Calcule la complexité d'une requête."""
        complexity = QueryComplexity()
        
        def analyze_node(node: Any, depth: int = 0):
            complexity.nested_depth = max(complexity.nested_depth, depth)
            
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "bool":
                        complexity.bool_clauses += 1
                        complexity.score += 5
                    elif key in ["wildcard", "regexp"]:
                        if key == "wildcard":
                            complexity.wildcard_count += 1
                        else:
                            complexity.regexp_count += 1
                        complexity.score += 15
                    elif key == "function_score":
                        complexity.function_score_count += 1
                        complexity.score += 10
                    elif key in ["should", "must", "must_not"]:
                        if isinstance(value, list):
                            complexity.bool_clauses += len(value)
                            complexity.score += len(value) * 2
                    
                    analyze_node(value, depth + 1)
            elif isinstance(node, list):
                complexity.score += len(node)
                for item in node:
                    analyze_node(item, depth + 1)
        
        if "query" in query:
            analyze_node(query["query"])
        
        return complexity
    
    def _validate_security(self, query: Dict[str, Any], result: ValidationResult):
        """Valide la sécurité de la requête."""
        query_str = str(query)
        if not self._check_security_patterns(query_str, result):
            if self.config.get("block_dangerous_patterns", False):
                raise SecurityValidationError("Pattern dangereux détecté dans la requête")
    
    def _validate_query_text(self, text: str, result: ValidationResult, path: str):
        """Valide le texte d'une requête."""
        if not isinstance(text, str):
            return
        
        # Validation longueur
        if len(text) > VALIDATION_LIMITS["max_query_length"]:
            result.add_error(f"Texte de requête trop long (max: {VALIDATION_LIMITS['max_query_length']})", path)
        elif len(text) < VALIDATION_LIMITS["min_query_length"]:
            result.add_error(f"Texte de requête trop court (min: {VALIDATION_LIMITS['min_query_length']})", path)
        
        # Validation sécurité
        if self.config.get("check_security_patterns", False):
            self._check_security_patterns(text, result)
    
    def _sanitize_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitise une requête Elasticsearch."""
        def sanitize_node(node):
            if isinstance(node, dict):
                return {k: sanitize_node(v) for k, v in node.items()}
            elif isinstance(node, list):
                return [sanitize_node(item) for item in node]
            elif isinstance(node, str):
                return self._sanitize_string(node)
            else:
                return node
        
        return sanitize_node(query)

# ==================== FONCTIONS UTILITAIRES ====================

def validate_search_request(query: str, user_id: int, filters: Dict[str, Any] = None) -> ValidationResult:
    """
    Fonction utilitaire pour valider une requête de recherche complète.
    
    Args:
        query: Texte de recherche
        user_id: ID utilisateur
        filters: Filtres optionnels
        
    Returns:
        ValidationResult
    """
    validator = QueryValidator()
    result = validator._create_result()
    
    # Validation des paramètres de base
    if not query or not isinstance(query, str):
        result.add_error("Query text is required")
    elif len(query.strip()) == 0:
        result.add_error("Query text cannot be empty")
    
    if not user_id or not isinstance(user_id, int) or user_id <= 0:
        result.add_error("Valid user_id is required")
    
    # Construction et validation de la requête ES
    if result.is_valid:
        es_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}},
                        {"multi_match": {
                            "query": query,
                            "fields": ["searchable_text", "merchant_name", "clean_description"]
                        }}
                    ]
                }
            },
            "size": 20
        }
        
        # Ajout des filtres
        if filters:
            filter_clauses = []
            for field, value in filters.items():
                if field in ALLOWED_SEARCH_FIELDS:
                    filter_clauses.append({"term": {field: value}})
            
            if filter_clauses:
                es_query["query"]["bool"]["filter"] = filter_clauses
        
        # Validation de la requête construite
        validation_result = validator.validate(es_query)
        result.errors.extend(validation_result.errors)
        result.warnings.extend(validation_result.warnings)
        result.sanitized_data = validation_result.sanitized_data or es_query
        
        if validation_result.errors:
            result.is_valid = False
    
    return result