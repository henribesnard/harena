"""
Validateur de paramètres API pour le Search Service.

Ce module fournit la validation complète des paramètres d'API,
incluant pagination, tri, timeout et autres options de requête.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime

from .base import (
    BaseValidator, ValidationResult, ValidationLevel,
    ParameterValidationError, validate_user_id
)
from .config import (
    VALIDATION_LIMITS, DATA_TYPE_LIMITS, VALIDATION_CONFIG,
    ERROR_MESSAGES
)

logger = logging.getLogger(__name__)

class ParameterValidator(BaseValidator):
    """
    Validateur spécialisé pour les paramètres API.
    
    Valide tous les paramètres passés aux endpoints de recherche,
    incluant pagination, tri, timeout, format de réponse, etc.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        self.config = VALIDATION_CONFIG[validation_level.value]
    
    def validate(self, parameters: Dict[str, Any], **kwargs) -> ValidationResult:
        """
        Valide l'ensemble des paramètres API.
        
        Args:
            parameters: Dictionnaire des paramètres à valider
            **kwargs: Options supplémentaires
            
        Returns:
            ValidationResult avec détails de validation
        """
        start_time = time.time()
        result = self._create_result()
        
        try:
            # Validation de base
            if not self._validate_basic_structure(parameters, result):
                return result
            
            sanitized_params = {}
            
            # Validation de chaque paramètre
            for param_name, param_value in parameters.items():
                param_result = self._validate_single_parameter(param_name, param_value)
                
                # Ajout des erreurs/warnings
                result.errors.extend(param_result.errors)
                result.warnings.extend(param_result.warnings)
                result.security_flags.extend(param_result.security_flags)
                
                if param_result.errors:
                    result.is_valid = False
                else:
                    sanitized_params[param_name] = param_result.sanitized_data or param_value
            
            # Validation des combinaisons de paramètres
            if result.is_valid:
                self._validate_parameter_combinations(sanitized_params, result)
            
            result.sanitized_data = sanitized_params if result.is_valid else None
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation des paramètres: {e}")
            result.add_error(f"Erreur interne de validation: {str(e)}")
        
        result.validation_time_ms = self._measure_validation_time(start_time)
        return result
    
    def validate_pagination(self, size: int = None, from_: int = None, 
                          page: int = None) -> ValidationResult:
        """
        Valide spécifiquement les paramètres de pagination.
        
        Args:
            size: Nombre de résultats
            from_: Offset
            page: Numéro de page
            
        Returns:
            ValidationResult
        """
        result = self._create_result()
        
        # Validation size
        if size is not None:
            if not isinstance(size, int):
                result.add_error("'size' doit être un entier")
            elif size < 0:
                result.add_error("'size' doit être positif")
            elif size > VALIDATION_LIMITS.get("max_results_limit", 1000):
                result.add_error(f"'size' trop grand (max: {VALIDATION_LIMITS.get('max_results_limit', 1000)})")
        
        # Validation from_
        if from_ is not None:
            if not isinstance(from_, int):
                result.add_error("'from' doit être un entier")
            elif from_ < 0:
                result.add_error("'from' doit être positif")
            elif from_ > 10000:
                result.add_error("'from' ne peut pas dépasser 10000")
        
        # Validation page
        if page is not None:
            if not isinstance(page, int):
                result.add_error("'page' doit être un entier")
            elif page < 1:
                result.add_error("'page' doit être >= 1")
        
        # Validation des combinaisons
        if size is not None and from_ is not None:
            if from_ + size > 10000:
                result.add_error("from + size ne peut pas dépasser 10000")
        
        if page is not None and size is not None:
            calculated_from = (page - 1) * size
            if calculated_from > 10000:
                result.add_error("page * size dépasse la limite d'offset")
        
        return result
    
    def validate_sorting(self, sort: Union[str, List[str], Dict[str, str]]) -> ValidationResult:
        """
        Valide les paramètres de tri.
        
        Args:
            sort: Configuration de tri
            
        Returns:
            ValidationResult
        """
        result = self._create_result()
        
        valid_fields = {
            "_score", "transaction_date", "amount", "merchant_name",
            "category_id", "created_at", "updated_at"
        }
        
        valid_orders = {"asc", "desc"}
        
        if isinstance(sort, str):
            # Format: "field:order" ou juste "field"
            if ":" in sort:
                field, order = sort.split(":", 1)
                if field not in valid_fields:
                    result.add_warning(f"Champ de tri non standard: {field}")
                if order.lower() not in valid_orders:
                    result.add_error(f"Ordre de tri invalide: {order}")
            else:
                if sort not in valid_fields:
                    result.add_warning(f"Champ de tri non standard: {sort}")
        
        elif isinstance(sort, list):
            if len(sort) > 5:
                result.add_warning("Trop de critères de tri (max recommandé: 5)")
            
            for sort_item in sort:
                item_result = self.validate_sorting(sort_item)
                result.errors.extend(item_result.errors)
                result.warnings.extend(item_result.warnings)
        
        elif isinstance(sort, dict):
            for field, order in sort.items():
                if field not in valid_fields:
                    result.add_warning(f"Champ de tri non standard: {field}")
                if isinstance(order, str) and order.lower() not in valid_orders:
                    result.add_error(f"Ordre de tri invalide pour {field}: {order}")
        
        else:
            result.add_error("Format de tri invalide")
        
        return result
    
    def _validate_basic_structure(self, parameters: Dict[str, Any], result: ValidationResult) -> bool:
        """Valide la structure de base des paramètres."""
        if not isinstance(parameters, dict):
            result.add_error("Les paramètres doivent être un dictionnaire")
            return False
        
        return True
    
    def _validate_single_parameter(self, param_name: str, param_value: Any) -> ValidationResult:
        """Valide un paramètre individuel."""
        result = self._create_result()
        
        # Validation selon le nom du paramètre
        if param_name in ["size", "limit"]:
            self._validate_size_parameter(param_value, result)
        elif param_name in ["from", "offset"]:
            self._validate_from_parameter(param_value, result)
        elif param_name == "page":
            self._validate_page_parameter(param_value, result)
        elif param_name in ["sort", "order_by"]:
            sort_result = self.validate_sorting(param_value)
            result.errors.extend(sort_result.errors)
            result.warnings.extend(sort_result.warnings)
        elif param_name == "user_id":
            self._validate_user_id_parameter(param_value, result)
        elif param_name == "timeout":
            self._validate_timeout_parameter(param_value, result)
        elif param_name in ["format", "response_format"]:
            self._validate_format_parameter(param_value, result)
        elif param_name in ["highlight", "highlighting"]:
            self._validate_highlight_parameter(param_value, result)
        elif param_name in ["aggregations", "aggs"]:
            self._validate_aggregations_parameter(param_value, result)
        elif param_name in ["fields", "_source"]:
            self._validate_fields_parameter(param_value, result)
        elif param_name in ["explain", "debug"]:
            self._validate_boolean_parameter(param_name, param_value, result)
        elif param_name in ["track_total_hits", "track_scores"]:
            self._validate_boolean_parameter(param_name, param_value, result)
        elif param_name == "min_score":
            self._validate_min_score_parameter(param_value, result)
        elif param_name in ["include_metadata", "include_stats"]:
            self._validate_boolean_parameter(param_name, param_value, result)
        else:
            self._validate_custom_parameter(param_name, param_value, result)
        
        # Sanitisation si nécessaire
        if self.config.get("sanitize_input", False) and result.is_valid:
            result.sanitized_data = self._sanitize_parameter_value(param_name, param_value)
        
        return result
    
    def _validate_size_parameter(self, value: Any, result: ValidationResult):
        """Valide le paramètre size/limit."""
        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                result.add_error("'size' doit être un entier")
                return
        
        if value < 0:
            result.add_error("'size' doit être positif")
        elif value > VALIDATION_LIMITS.get("max_results_limit", 1000):
            result.add_error(f"'size' trop grand (max: {VALIDATION_LIMITS.get('max_results_limit', 1000)})")
        elif value == 0:
            result.add_warning("'size' de 0 ne retournera aucun résultat")
    
    def _validate_from_parameter(self, value: Any, result: ValidationResult):
        """Valide le paramètre from/offset."""
        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                result.add_error("'from' doit être un entier")
                return
        
        if value < 0:
            result.add_error("'from' doit être positif")
        elif value > 10000:
            result.add_error("'from' ne peut pas dépasser 10000 (limitation Elasticsearch)")
    
    def _validate_page_parameter(self, value: Any, result: ValidationResult):
        """Valide le paramètre page."""
        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                result.add_error("'page' doit être un entier")
                return
        
        if value < 1:
            result.add_error("'page' doit être >= 1")
        elif value > 1000:
            result.add_warning("Numéro de page très élevé, performances dégradées possibles")
    
    def _validate_user_id_parameter(self, value: Any, result: ValidationResult):
        """Valide le paramètre user_id."""
        if not validate_user_id(value):
            result.add_error("ID utilisateur invalide")
        
        # Ajout d'un flag de sécurité
        result.add_security_flag(f"Paramètre user_id: {value}")
    
    def _validate_timeout_parameter(self, value: Any, result: ValidationResult):
        """Valide le paramètre timeout."""
        if isinstance(value, str):
            # Format: "5s", "1000ms"
            if not value.endswith(("s", "ms")):
                result.add_error("Format timeout invalide (doit finir par 's' ou 'ms')")
                return
            
            try:
                numeric_part = value[:-2] if value.endswith("ms") else value[:-1]
                timeout_value = int(numeric_part)
                
                if value.endswith("ms"):
                    if timeout_value > VALIDATION_LIMITS.get("max_timeout_ms", 30000):
                        result.add_error(f"Timeout trop long (max: {VALIDATION_LIMITS.get('max_timeout_ms', 30000)}ms)")
                else:  # seconds
                    if timeout_value > 30:
                        result.add_error("Timeout trop long (max: 30s)")
            except ValueError:
                result.add_error("Valeur timeout invalide")
        
        elif isinstance(value, int):
            # Assumé en millisecondes
            if value > VALIDATION_LIMITS.get("max_timeout_ms", 30000):
                result.add_error(f"Timeout trop long (max: {VALIDATION_LIMITS.get('max_timeout_ms', 30000)}ms)")
            elif value < 100:
                result.add_warning("Timeout très court, risque d'échec")
        
        else:
            result.add_error("Type timeout invalide")
    
    def _validate_format_parameter(self, value: Any, result: ValidationResult):
        """Valide le paramètre format/response_format."""
        valid_formats = {"json", "xml", "csv", "simplified"}
        
        if not isinstance(value, str):
            result.add_error("Format doit être une chaîne")
        elif value.lower() not in valid_formats:
            result.add_error(f"Format non supporté: {value}")
    
    def _validate_highlight_parameter(self, value: Any, result: ValidationResult):
        """Valide le paramètre highlight."""
        if isinstance(value, bool):
            # Simple activation/désactivation
            pass
        elif isinstance(value, dict):
            # Configuration détaillée
            valid_highlight_keys = {
                "fields", "pre_tags", "post_tags", "fragment_size", 
                "number_of_fragments", "fragmenter", "boundary_chars"
            }
            
            for key in value.keys():
                if key not in valid_highlight_keys:
                    result.add_warning(f"Option highlight non standard: {key}")
            
            # Validation des champs
            if "fields" in value:
                if not isinstance(value["fields"], (list, dict)):
                    result.add_error("highlight.fields doit être une liste ou dictionnaire")
        else:
            result.add_error("highlight doit être un booléen ou dictionnaire")
    
    def _validate_aggregations_parameter(self, value: Any, result: ValidationResult):
        """Valide le paramètre aggregations."""
        if isinstance(value, bool):
            # Simple activation
            pass
        elif isinstance(value, list):
            # Liste de types d'agrégations
            valid_agg_types = {"merchants", "categories", "amounts", "dates", "currency"}
            for agg_type in value:
                if not isinstance(agg_type, str):
                    result.add_error("Types d'agrégation doivent être des chaînes")
                elif agg_type not in valid_agg_types:
                    result.add_warning(f"Type d'agrégation non standard: {agg_type}")
        elif isinstance(value, dict):
            # Configuration détaillée des agrégations
            if len(value) > 10:
                result.add_warning("Beaucoup d'agrégations demandées, impact performance possible")
        else:
            result.add_error("aggregations doit être un booléen, liste ou dictionnaire")
    
    def _validate_fields_parameter(self, value: Any, result: ValidationResult):
        """Valide le paramètre fields/_source."""
        if isinstance(value, bool):
            # true/false pour inclure/exclure tous les champs
            pass
        elif isinstance(value, list):
            # Liste de champs spécifiques
            if len(value) > 50:
                result.add_warning("Beaucoup de champs demandés")
            
            for field in value:
                if not isinstance(field, str):
                    result.add_error("Les noms de champs doivent être des chaînes")
        elif isinstance(value, dict):
            # Format include/exclude
            valid_keys = {"includes", "excludes", "include", "exclude"}
            for key in value.keys():
                if key not in valid_keys:
                    result.add_warning(f"Clé _source non standard: {key}")
        else:
            result.add_error("fields/_source doit être un booléen, liste ou dictionnaire")
    
    def _validate_boolean_parameter(self, param_name: str, value: Any, result: ValidationResult):
        """Valide un paramètre booléen."""
        if not isinstance(value, bool):
            if isinstance(value, str):
                if value.lower() in ["true", "1", "yes", "on"]:
                    result.sanitized_data = True
                elif value.lower() in ["false", "0", "no", "off"]:
                    result.sanitized_data = False
                else:
                    result.add_error(f"'{param_name}' doit être un booléen")
            elif isinstance(value, int):
                if value in [0, 1]:
                    result.sanitized_data = bool(value)
                else:
                    result.add_error(f"'{param_name}' doit être un booléen")
            else:
                result.add_error(f"'{param_name}' doit être un booléen")
    
    def _validate_min_score_parameter(self, value: Any, result: ValidationResult):
        """Valide le paramètre min_score."""
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                result.add_error("'min_score' doit être numérique")
                return
        
        if value < 0:
            result.add_error("'min_score' doit être positif")
        elif value > 100:
            result.add_warning("'min_score' très élevé, peu de résultats attendus")
    
    def _validate_custom_parameter(self, param_name: str, param_value: Any, result: ValidationResult):
        """Valide un paramètre custom/non standard."""
        # Validation de sécurité pour paramètres non reconnus
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            result.add_warning(f"Paramètre non standard: {param_name}")
        
        # Vérification des patterns de sécurité dans les valeurs string
        if isinstance(param_value, str):
            if not self._check_security_patterns(param_value, result):
                result.add_security_flag(f"Pattern suspect dans {param_name}")
    
    def _validate_parameter_combinations(self, parameters: Dict[str, Any], result: ValidationResult):
        """Valide les combinaisons de paramètres."""
        # Validation pagination
        if "page" in parameters and "from" in parameters:
            result.add_warning("'page' et 'from' spécifiés simultanément, 'from' prioritaire")
        
        # Validation timeout et performance
        if "timeout" in parameters and "size" in parameters:
            size = parameters.get("size", 20)
            if size > 100 and "timeout" not in str(parameters["timeout"]):
                result.add_warning("Grande taille de résultat sans timeout approprié")
        
        # Validation aggregations et size
        if parameters.get("aggregations") and parameters.get("size", 20) > 0:
            if parameters.get("size") > 100:
                result.add_warning("Grande taille + agrégations peuvent impacter les performances")
        
        # Validation highlighting et performance
        if parameters.get("highlight") and parameters.get("size", 20) > 50:
            result.add_warning("Highlighting + grande taille peuvent impacter les performances")
        
        # Validation explain/debug en production
        if parameters.get("explain") or parameters.get("debug"):
            result.add_warning("Options debug activées, impact performance possible")
    
    def _sanitize_parameter_value(self, param_name: str, value: Any) -> Any:
        """Sanitise la valeur d'un paramètre."""
        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, list):
            return [self._sanitize_parameter_value(param_name, item) for item in value]
        elif isinstance(value, dict):
            return {k: self._sanitize_parameter_value(param_name, v) for k, v in value.items()}
        else:
            return value

# ==================== FONCTIONS UTILITAIRES ====================

def validate_api_parameters(parameters: Dict[str, Any], 
                          validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
    """
    Fonction utilitaire pour valider les paramètres d'API.
    
    Args:
        parameters: Dictionnaire des paramètres
        validation_level: Niveau de validation
        
    Returns:
        ValidationResult
    """
    validator = ParameterValidator(validation_level)
    return validator.validate(parameters)

def validate_search_pagination(size: int = None, from_: int = None, 
                             page: int = None) -> ValidationResult:
    """
    Valide spécifiquement les paramètres de pagination.
    
    Args:
        size: Nombre de résultats
        from_: Offset
        page: Numéro de page
        
    Returns:
        ValidationResult
    """
    validator = ParameterValidator()
    return validator.validate_pagination(size, from_, page)

def normalize_pagination_params(size: int = None, from_: int = None, 
                              page: int = None) -> Dict[str, int]:
    """
    Normalise et convertit les paramètres de pagination.
    
    Args:
        size: Nombre de résultats
        from_: Offset
        page: Numéro de page
        
    Returns:
        Dictionnaire avec size et from normalisés
    """
    # Valeurs par défaut
    normalized_size = size or VALIDATION_LIMITS.get("default_results_limit", 20)
    normalized_from = from_ or 0
    
    # Si page est spécifié, calculer from
    if page is not None and page > 0:
        normalized_from = (page - 1) * normalized_size
    
    # Application des limites
    normalized_size = min(normalized_size, VALIDATION_LIMITS.get("max_results_limit", 1000))
    normalized_from = min(normalized_from, 10000)
    
    return {
        "size": max(0, normalized_size),
        "from": max(0, normalized_from)
    }

def build_elasticsearch_parameters(api_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convertit les paramètres API en paramètres Elasticsearch.
    
    Args:
        api_params: Paramètres API validés
        
    Returns:
        Paramètres Elasticsearch
    """
    es_params = {}
    
    # Pagination
    if "size" in api_params:
        es_params["size"] = api_params["size"]
    if "from" in api_params:
        es_params["from"] = api_params["from"]
    elif "page" in api_params and "size" in api_params:
        es_params["from"] = (api_params["page"] - 1) * api_params["size"]
    
    # Tri
    if "sort" in api_params:
        sort_config = api_params["sort"]
        if isinstance(sort_config, str):
            if ":" in sort_config:
                field, order = sort_config.split(":", 1)
                es_params["sort"] = [{field: {"order": order}}]
            else:
                es_params["sort"] = [{sort_config: {"order": "asc"}}]
        elif isinstance(sort_config, dict):
            es_params["sort"] = [{k: {"order": v}} for k, v in sort_config.items()]
    
    # Highlighting
    if api_params.get("highlight"):
        if isinstance(api_params["highlight"], bool):
            es_params["highlight"] = {
                "fields": {
                    "searchable_text": {},
                    "merchant_name": {},
                    "clean_description": {}
                }
            }
        else:
            es_params["highlight"] = api_params["highlight"]
    
    # Champs source
    if "fields" in api_params:
        es_params["_source"] = api_params["fields"]
    
    # Autres paramètres
    for param in ["timeout", "min_score", "track_total_hits"]:
        if param in api_params:
            es_params[param] = api_params[param]
    
    return es_params

def extract_api_metadata(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrait les métadonnées des paramètres API.
    
    Args:
        parameters: Paramètres API
        
    Returns:
        Métadonnées d'API
    """
    metadata = {
        "request_time": datetime.utcnow().isoformat(),
        "pagination": {},
        "options": {},
        "performance_flags": []
    }
    
    # Pagination
    if "size" in parameters:
        metadata["pagination"]["size"] = parameters["size"]
    if "from" in parameters:
        metadata["pagination"]["from"] = parameters["from"]
    if "page" in parameters:
        metadata["pagination"]["page"] = parameters["page"]
    
    # Options
    for option in ["highlight", "explain", "aggregations"]:
        if option in parameters:
            metadata["options"][option] = parameters[option]
    
    # Flags de performance
    if parameters.get("size", 0) > 100:
        metadata["performance_flags"].append("large_result_set")
    if parameters.get("explain"):
        metadata["performance_flags"].append("explain_enabled")
    if parameters.get("aggregations"):
        metadata["performance_flags"].append("aggregations_requested")
    
    return metadata