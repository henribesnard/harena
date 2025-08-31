"""
Validateur de requêtes search_service
Validation syntaxique et sémantique des requêtes Elasticsearch
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date

from conversation_service.models.contracts.search_service import (
    SearchQuery, QueryValidationResult, SUPPORTED_FIELD_TYPES,
    MAX_AGGREGATION_BUCKETS, MAX_NESTED_AGGREGATION_LEVELS,
    DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
)

logger = logging.getLogger("conversation_service.query_validator")


class QueryValidator:
    """Validateur de requêtes search_service avec optimisations automatiques"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.performance_thresholds = self._load_performance_thresholds()
    
    def validate_query(self, query_dict: Dict[str, Any]) -> QueryValidationResult:
        """
        Validation complète d'une requête search_service
        
        Args:
            query_dict: Dictionnaire représentant la requête
            
        Returns:
            QueryValidationResult: Résultat validation avec optimisations
        """
        try:
            # Validation schéma de base
            schema_errors = self._validate_schema(query_dict)
            
            # Validation contrats métier
            contract_errors = self._validate_contracts(query_dict)
            
            # Estimation performance
            performance_issues, performance_level = self._estimate_performance(query_dict)
            
            # Application optimisations automatiques
            optimizations = self._apply_automatic_optimizations(query_dict)
            
            # Validation post-optimisation
            post_optimization_errors = self._validate_optimized_query(query_dict)
            
            # Consolidation résultats
            all_errors = schema_errors + contract_errors + post_optimization_errors
            all_warnings = performance_issues
            
            schema_valid = len(schema_errors) == 0
            contract_compliant = len(contract_errors) == 0 and len(post_optimization_errors) == 0
            
            return QueryValidationResult(
                schema_valid=schema_valid,
                contract_compliant=contract_compliant,
                estimated_performance=performance_level,
                optimization_applied=optimizations,
                potential_issues=performance_issues,
                errors=all_errors,
                warnings=all_warnings
            )
            
        except Exception as e:
            logger.error(f"Erreur validation requête: {str(e)}")
            return QueryValidationResult(
                schema_valid=False,
                contract_compliant=False,
                estimated_performance="poor",
                optimization_applied=[],
                potential_issues=[],
                errors=[f"Erreur validation: {str(e)}"],
                warnings=[]
            )
    
    def _validate_schema(self, query_dict: Dict[str, Any]) -> List[str]:
        """Validation schéma de base"""
        errors = []
        
        # user_id obligatoire
        if "user_id" not in query_dict:
            errors.append("user_id est requis")
        elif not isinstance(query_dict["user_id"], int):
            errors.append("user_id doit être un entier")
        
        # Validation page_size
        if "page_size" in query_dict:
            page_size = query_dict["page_size"]
            if not isinstance(page_size, int) or page_size < 1 or page_size > MAX_PAGE_SIZE:
                errors.append(f"page_size doit être entre 1 et {MAX_PAGE_SIZE}")
        
        # Validation offset
        if "offset" in query_dict:
            offset = query_dict["offset"]
            if not isinstance(offset, int) or offset < 0:
                errors.append("offset doit être un entier positif")
        
        # Validation structure filters
        if "filters" in query_dict:
            filter_errors = self._validate_filters_schema(query_dict["filters"])
            errors.extend(filter_errors)
        
        # Validation structure aggregations
        if "aggregations" in query_dict:
            agg_errors = self._validate_aggregations_schema(query_dict["aggregations"])
            errors.extend(agg_errors)
        
        # Validation sort
        if "sort" in query_dict:
            sort_errors = self._validate_sort_schema(query_dict["sort"])
            errors.extend(sort_errors)
        
        return errors
    
    def _validate_filters_schema(self, filters: Dict[str, Any]) -> List[str]:
        """Validation schéma des filtres"""
        errors = []
        
        for field_name, filter_value in filters.items():
            if field_name not in SUPPORTED_FIELD_TYPES:
                errors.append(f"Champ de filtre non supporté: {field_name}")
                continue
            
            # Validation types de filtres pour ce champ
            supported_types = SUPPORTED_FIELD_TYPES[field_name]
            
            if isinstance(filter_value, dict):
                for filter_type in filter_value.keys():
                    if filter_type not in supported_types:
                        errors.append(f"Type de filtre '{filter_type}' non supporté pour '{field_name}'")
            
            # Validation spécifique dates
            if field_name == "date" and isinstance(filter_value, dict):
                date_errors = self._validate_date_filter(filter_value)
                errors.extend(date_errors)
            
            # Validation spécifique montants
            if field_name in ["amount", "amount_abs"] and isinstance(filter_value, dict):
                amount_errors = self._validate_amount_filter(filter_value)
                errors.extend(amount_errors)
        
        return errors
    
    def _validate_date_filter(self, date_filter: Dict[str, Any]) -> List[str]:
        """Validation filtres de dates"""
        errors = []
        
        date_fields = ["gte", "lte", "gt", "lt"]
        for field in date_fields:
            if field in date_filter:
                date_value = date_filter[field]
                if not self._is_valid_date_string(date_value):
                    errors.append(f"Format de date invalide pour '{field}': {date_value}")
        
        # Validation cohérence dates
        if "gte" in date_filter and "lte" in date_filter:
            try:
                gte_date = datetime.fromisoformat(date_filter["gte"].replace("Z", "+00:00"))
                lte_date = datetime.fromisoformat(date_filter["lte"].replace("Z", "+00:00"))
                if gte_date > lte_date:
                    errors.append("Date de début (gte) doit être antérieure à date de fin (lte)")
            except:
                pass  # Erreurs de format déjà capturées
        
        return errors
    
    def _validate_amount_filter(self, amount_filter: Dict[str, Any]) -> List[str]:
        """Validation filtres de montants"""
        errors = []
        
        amount_fields = ["gte", "lte", "gt", "lt"]
        for field in amount_fields:
            if field in amount_filter:
                amount_value = amount_filter[field]
                if not isinstance(amount_value, (int, float)):
                    errors.append(f"Montant doit être numérique pour '{field}': {amount_value}")
                elif amount_value < 0:
                    errors.append(f"Montant doit être positif pour '{field}': {amount_value}")
        
        return errors
    
    def _validate_aggregations_schema(self, aggregations: Dict[str, Any]) -> List[str]:
        """Validation schéma des agrégations"""
        errors = []
        
        if len(aggregations) > 10:
            errors.append("Maximum 10 agrégations par requête")
        
        for agg_name, agg_config in aggregations.items():
            if not isinstance(agg_config, dict):
                errors.append(f"Configuration agrégation invalide pour '{agg_name}'")
                continue
            
            # Validation structure agrégation
            agg_errors = self._validate_single_aggregation(agg_name, agg_config)
            errors.extend(agg_errors)
        
        return errors
    
    def _validate_single_aggregation(self, agg_name: str, agg_config: Dict[str, Any], 
                                   level: int = 1) -> List[str]:
        """Validation d'une agrégation individuelle"""
        errors = []
        
        # Vérification niveau imbrication
        if level > MAX_NESTED_AGGREGATION_LEVELS:
            errors.append(f"Niveau d'imbrication maximal dépassé pour '{agg_name}': {level}")
        
        # Validation types d'agrégation supportés
        supported_agg_types = ["terms", "sum", "avg", "max", "min", "value_count", 
                              "date_histogram", "cardinality"]
        
        agg_type = None
        for agg_type_key in agg_config.keys():
            if agg_type_key in supported_agg_types:
                agg_type = agg_type_key
                break
            elif agg_type_key == "aggs":
                # Validation agrégations imbriquées
                if isinstance(agg_config["aggs"], dict):
                    for sub_agg_name, sub_agg_config in agg_config["aggs"].items():
                        sub_errors = self._validate_single_aggregation(
                            sub_agg_name, sub_agg_config, level + 1
                        )
                        errors.extend(sub_errors)
        
        if not agg_type:
            errors.append(f"Type d'agrégation non reconnu pour '{agg_name}'")
        
        # Validation configuration selon type
        if agg_type == "terms":
            terms_config = agg_config[agg_type]
            if "field" not in terms_config:
                errors.append(f"Champ 'field' requis pour agrégation terms '{agg_name}'")
            
            if "size" in terms_config:
                size = terms_config["size"]
                if not isinstance(size, int) or size < 1 or size > MAX_AGGREGATION_BUCKETS:
                    errors.append(f"Size doit être entre 1 et {MAX_AGGREGATION_BUCKETS} pour '{agg_name}'")
        
        elif agg_type == "date_histogram":
            date_hist_config = agg_config[agg_type]
            if "field" not in date_hist_config:
                errors.append(f"Champ 'field' requis pour date_histogram '{agg_name}'")
            if "calendar_interval" not in date_hist_config:
                errors.append(f"Champ 'calendar_interval' requis pour date_histogram '{agg_name}'")
        
        elif agg_type in ["sum", "avg", "max", "min", "value_count", "cardinality"]:
            metric_config = agg_config[agg_type]
            if "field" not in metric_config:
                errors.append(f"Champ 'field' requis pour agrégation {agg_type} '{agg_name}'")
        
        return errors
    
    def _validate_sort_schema(self, sort_config: List[Dict[str, Any]]) -> List[str]:
        """Validation configuration tri"""
        errors = []
        
        if not isinstance(sort_config, list):
            errors.append("Configuration sort doit être une liste")
            return errors
        
        for i, sort_item in enumerate(sort_config):
            if not isinstance(sort_item, dict):
                errors.append(f"Élément sort[{i}] doit être un dictionnaire")
                continue
            
            if len(sort_item) != 1:
                errors.append(f"Élément sort[{i}] doit contenir exactement un champ")
                continue
            
            field_name = list(sort_item.keys())[0]
            sort_order_config = sort_item[field_name]
            
            if isinstance(sort_order_config, dict):
                if "order" not in sort_order_config:
                    errors.append(f"Configuration 'order' manquante pour sort '{field_name}'")
                elif sort_order_config["order"] not in ["asc", "desc"]:
                    errors.append(f"Order doit être 'asc' ou 'desc' pour '{field_name}'")
            elif isinstance(sort_order_config, str):
                if sort_order_config not in ["asc", "desc"]:
                    errors.append(f"Order doit être 'asc' ou 'desc' pour '{field_name}'")
        
        return errors
    
    def _validate_contracts(self, query_dict: Dict[str, Any]) -> List[str]:
        """Validation contrats métier"""
        errors = []
        
        # user_id requis au niveau racine
        if "user_id" not in query_dict:
            errors.append("user_id est requis au niveau racine")
        
        # user_id peut être dans filters (contrairement à l'ancienne logique)
        # Pas d'erreur si user_id est dans filters ET au niveau racine
        
        # Validation cohérence aggregation_only
        if query_dict.get("aggregation_only", False):
            if not query_dict.get("aggregations"):
                errors.append("aggregation_only=true requiert des agrégations")
            if "include_fields" in query_dict:
                errors.append("include_fields incompatible avec aggregation_only=true")
        
        return errors
    
    def _estimate_performance(self, query_dict: Dict[str, Any]) -> Tuple[List[str], str]:
        """Estimation performance de la requête"""
        issues = []
        performance_score = 100
        
        # Pénalités performance
        filters = query_dict.get("filters", {})
        aggregations = query_dict.get("aggregations", {})
        
        # Recherche textuelle libre coûteuse
        if "query" in query_dict and query_dict["query"]:
            performance_score -= 20
            issues.append("Recherche textuelle libre peut impacter les performances")
        
        # Filtres textuels floues coûteux
        for field_name, filter_value in filters.items():
            if isinstance(filter_value, dict) and "match" in filter_value:
                performance_score -= 10
                issues.append(f"Filtre textuel flou sur '{field_name}' peut être coûteux")
        
        # Agrégations multiples
        if len(aggregations) > 5:
            performance_score -= 15
            issues.append("Nombre élevé d'agrégations peut impacter les performances")
        
        # Buckets agrégations élevés
        for agg_name, agg_config in aggregations.items():
            if isinstance(agg_config, dict):
                terms_config = agg_config.get("terms", {})
                if "size" in terms_config and terms_config["size"] > 50:
                    performance_score -= 10
                    issues.append(f"Nombre élevé de buckets pour '{agg_name}'")
        
        # Page size élevée
        page_size = query_dict.get("page_size", DEFAULT_PAGE_SIZE)
        if page_size > 100:
            performance_score -= 5
            issues.append("Taille de page élevée peut ralentir la réponse")
        
        # Tri sans filtres
        if "sort" in query_dict and not filters:
            performance_score -= 15
            issues.append("Tri sans filtres peut être coûteux sur gros volumes")
        
        # Détermination niveau performance
        if performance_score >= 80:
            performance_level = "optimal"
        elif performance_score >= 60:
            performance_level = "good"
        else:
            performance_level = "poor"
        
        return issues, performance_level
    
    def _apply_automatic_optimizations(self, query_dict: Dict[str, Any]) -> List[str]:
        """Application optimisations automatiques"""
        optimizations = []
        
        # user_id peut être dans filters - plus de suppression automatique
        
        # Limitation buckets agrégations
        aggregations = query_dict.get("aggregations", {})
        for agg_name, agg_config in aggregations.items():
            if isinstance(agg_config, dict) and "terms" in agg_config:
                terms_config = agg_config["terms"]
                if "size" not in terms_config or terms_config["size"] > 20:
                    terms_config["size"] = 20
                    optimizations.append(f"Limitation buckets agrégation '{agg_name}' à 20")
        
        # include_fields par défaut
        if ("include_fields" not in query_dict and 
            not query_dict.get("aggregation_only", False)):
            query_dict["include_fields"] = [
                "transaction_id", "amount", "amount_abs", "merchant_name",
                "date", "primary_description", "category_name", "operation_type"
            ]
            optimizations.append("Ajout champs essentiels uniquement")
        
        # page_size par défaut
        if "page_size" not in query_dict:
            query_dict["page_size"] = DEFAULT_PAGE_SIZE
            optimizations.append(f"Configuration page_size par défaut: {DEFAULT_PAGE_SIZE}")
        
        return optimizations
    
    def _validate_optimized_query(self, query_dict: Dict[str, Any]) -> List[str]:
        """Validation post-optimisation"""
        errors = []
        
        # Validation cohérence finale
        if query_dict.get("aggregation_only", False) and "include_fields" in query_dict:
            del query_dict["include_fields"]
            # Pas d'erreur, correction automatique
        
        return errors
    
    def _is_valid_date_string(self, date_str: str) -> bool:
        """Validation format date"""
        if not isinstance(date_str, str):
            return False
        
        # Formats supportés: ISO date, ISO datetime
        try:
            # Format YYYY-MM-DD
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            try:
                # Format ISO avec timezone
                datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                return True
            except ValueError:
                return False
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Chargement règles de validation personnalisées"""
        return {
            "required_user_context": True,
            "auto_optimize_aggregations": True,
            "max_query_complexity": 100,
            "enforce_field_inclusion": True
        }
    
    def _load_performance_thresholds(self) -> Dict[str, Any]:
        """Chargement seuils de performance"""
        return {
            "max_aggregations": 10,
            "max_bucket_size": 50,
            "optimal_page_size": 20,
            "max_nested_levels": 3,
            "text_search_penalty": 20,
            "multiple_agg_penalty": 15
        }