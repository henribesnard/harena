"""
Validateur de filtres pour le Search Service.

Ce module fournit la validation complète des filtres de recherche,
avec vérification des types, valeurs et contraintes de sécurité.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, timedelta

from .base import (
    BaseValidator, ValidationResult, ValidationLevel,
    FilterValidationError, validate_user_id, validate_amount, validate_date
)
from .config import (
    ALLOWED_SEARCH_FIELDS, SENSITIVE_FIELDS, ALLOWED_FILTER_TYPES,
    ALLOWED_COMPARISON_OPERATORS, VALIDATION_LIMITS, DATA_TYPE_LIMITS,
    VALIDATION_CONFIG, ERROR_MESSAGES
)

logger = logging.getLogger(__name__)

class FilterValidator(BaseValidator):
    """
    Validateur spécialisé pour les filtres de recherche.
    
    Valide les filtres appliqués aux requêtes de recherche,
    avec vérification des types, valeurs et sécurité.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        self.config = VALIDATION_CONFIG[validation_level.value]
    
    def validate(self, filters: Dict[str, Any], **kwargs) -> ValidationResult:
        """
        Valide un dictionnaire de filtres.
        
        Args:
            filters: Dictionnaire des filtres à valider
            **kwargs: Options supplémentaires
            
        Returns:
            ValidationResult avec détails de validation
        """
        start_time = time.time()
        result = self._create_result()
        
        try:
            # Validation de base
            if not self._validate_basic_structure(filters, result):
                return result
            
            # Validation du nombre de filtres
            if not self._validate_filter_count(filters, result):
                return result
            
            # Validation de chaque filtre
            sanitized_filters = {}
            for field, filter_value in filters.items():
                field_result = self._validate_single_filter(field, filter_value)
                
                # Ajout des erreurs/warnings au résultat principal
                result.errors.extend(field_result.errors)
                result.warnings.extend(field_result.warnings)
                result.security_flags.extend(field_result.security_flags)
                
                if field_result.errors:
                    result.is_valid = False
                else:
                    # Ajout du filtre sanitisé
                    sanitized_filters[field] = field_result.sanitized_data or filter_value
            
            # Validation des combinaisons de filtres
            if result.is_valid:
                self._validate_filter_combinations(sanitized_filters, result)
            
            # Validation de sécurité globale
            if self.config.get("check_security_patterns", False):
                self._validate_security(sanitized_filters, result)
            
            result.sanitized_data = sanitized_filters if result.is_valid else None
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation des filtres: {e}")
            result.add_error(f"Erreur interne de validation: {str(e)}")
        
        result.validation_time_ms = self._measure_validation_time(start_time)
        return result
    
    def validate_single_filter(self, field: str, value: Any) -> ValidationResult:
        """
        Valide un filtre individuel.
        
        Args:
            field: Nom du champ
            value: Valeur du filtre
            
        Returns:
            ValidationResult pour ce filtre
        """
        return self._validate_single_filter(field, value)
    
    def _validate_basic_structure(self, filters: Dict[str, Any], result: ValidationResult) -> bool:
        """Valide la structure de base des filtres."""
        if not isinstance(filters, dict):
            result.add_error("Les filtres doivent être un dictionnaire")
            return False
        
        if not filters:
            result.add_warning("Aucun filtre fourni")
            return True
        
        return True
    
    def _validate_filter_count(self, filters: Dict[str, Any], result: ValidationResult) -> bool:
        """Valide le nombre de filtres."""
        max_filters = VALIDATION_LIMITS.get("max_filter_count", 20)
        
        if len(filters) > max_filters:
            result.add_error(f"Trop de filtres (max: {max_filters})")
            return False
        
        return True
    
    def _validate_single_filter(self, field: str, value: Any) -> ValidationResult:
        """Valide un filtre individuel."""
        result = self._create_result()
        
        # Validation du nom de champ
        if not self._validate_field_name(field, result):
            return result
        
        # Validation spécifique par type de champ
        if field == "user_id":
            self._validate_user_id_filter(value, result)
        elif field == "amount" or field.startswith("amount_"):
            self._validate_amount_filter(field, value, result)
        elif field.endswith("_date") or field in ["transaction_date", "created_at", "updated_at"]:
            self._validate_date_filter(field, value, result)
        elif field in ["category_id", "categories"]:
            self._validate_category_filter(value, result)
        elif field in ["merchant_name", "merchants"]:
            self._validate_merchant_filter(value, result)
        elif field in ["transaction_type", "operation_type"]:
            self._validate_type_filter(value, result)
        elif field.endswith("_range"):
            self._validate_range_filter(field, value, result)
        else:
            self._validate_generic_filter(field, value, result)
        
        # Sanitisation si nécessaire
        if self.config.get("sanitize_input", False) and result.is_valid:
            result.sanitized_data = self._sanitize_filter_value(field, value)
        
        return result
    
    def _validate_field_name(self, field: str, result: ValidationResult) -> bool:
        """Valide le nom d'un champ de filtre."""
        if not isinstance(field, str):
            result.add_error("Le nom du champ doit être une chaîne")
            return False
        
        if not field.strip():
            result.add_error("Le nom du champ ne peut pas être vide")
            return False
        
        # Vérification de la liste des champs autorisés
        base_field = field.split("_")[0]  # Pour gérer amount_min, date_range, etc.
        if field not in ALLOWED_SEARCH_FIELDS and base_field not in ALLOWED_SEARCH_FIELDS:
            if self.validation_level == ValidationLevel.PARANOID:
                result.add_error(f"Champ non autorisé: {field}")
                return False
            else:
                result.add_warning(f"Champ non standard: {field}")
        
        # Vérification des champs sensibles
        if field in SENSITIVE_FIELDS:
            result.add_security_flag(f"Filtre sur champ sensible: {field}")
        
        return True
    
    def _validate_user_id_filter(self, value: Any, result: ValidationResult):
        """Valide un filtre user_id."""
        if not validate_user_id(value):
            result.add_error("ID utilisateur invalide")
        
        # Vérification des limites
        limits = DATA_TYPE_LIMITS.get("user_id", {})
        if isinstance(value, int):
            if value < limits.get("min_value", 1):
                result.add_error(f"ID utilisateur trop petit (min: {limits.get('min_value', 1)})")
            elif value > limits.get("max_value", 999999999):
                result.add_error(f"ID utilisateur trop grand (max: {limits.get('max_value', 999999999)})")
    
    def _validate_amount_filter(self, field: str, value: Any, result: ValidationResult):
        """Valide un filtre de montant."""
        if field.endswith("_min") or field.endswith("_max"):
            # Filtre de plage de montant
            if not validate_amount(value):
                result.add_error(f"Montant invalide pour {field}")
                return
        elif field == "amount":
            # Montant exact
            if isinstance(value, dict):
                # Format range: {"min": x, "max": y}
                for key, amount_val in value.items():
                    if key in ["min", "max", "gte", "lte", "gt", "lt"]:
                        if not validate_amount(amount_val):
                            result.add_error(f"Montant invalide pour amount.{key}")
            elif isinstance(value, list):
                # Liste de montants
                for i, amount_val in enumerate(value):
                    if not validate_amount(amount_val):
                        result.add_error(f"Montant invalide à l'index {i}")
            else:
                # Montant unique
                if not validate_amount(value):
                    result.add_error("Montant invalide")
        
        # Vérification des limites
        limits = DATA_TYPE_LIMITS.get("amount", {})
        self._check_numeric_limits(value, limits, result, field)
    
    def _validate_date_filter(self, field: str, value: Any, result: ValidationResult):
        """Valide un filtre de date."""
        if isinstance(value, dict):
            # Format range: {"start": x, "end": y}
            for key, date_val in value.items():
                if key in ["start", "end", "gte", "lte", "gt", "lt"]:
                    if not validate_date(date_val):
                        result.add_error(f"Date invalide pour {field}.{key}")
        elif isinstance(value, list):
            # Liste de dates
            for i, date_val in enumerate(value):
                if not validate_date(date_val):
                    result.add_error(f"Date invalide à l'index {i}")
        else:
            # Date unique
            if not validate_date(value):
                result.add_error(f"Date invalide pour {field}")
        
        # Vérification des plages logiques
        if isinstance(value, dict) and "start" in value and "end" in value:
            try:
                start_date = datetime.fromisoformat(value["start"].replace('Z', '+00:00'))
                end_date = datetime.fromisoformat(value["end"].replace('Z', '+00:00'))
                if start_date > end_date:
                    result.add_error("La date de début doit être antérieure à la date de fin")
            except (ValueError, AttributeError):
                pass  # Erreur déjà signalée dans validate_date
    
    def _validate_category_filter(self, value: Any, result: ValidationResult):
        """Valide un filtre de catégorie."""
        if isinstance(value, list):
            if len(value) > VALIDATION_LIMITS.get("max_filter_values", 100):
                result.add_error(f"Trop de catégories (max: {VALIDATION_LIMITS.get('max_filter_values', 100)})")
            
            for i, cat in enumerate(value):
                if not isinstance(cat, (str, int)):
                    result.add_error(f"Catégorie invalide à l'index {i}")
        elif isinstance(value, (str, int)):
            # Catégorie unique
            pass
        else:
            result.add_error("Les catégories doivent être des chaînes, entiers ou listes")
    
    def _validate_merchant_filter(self, value: Any, result: ValidationResult):
        """Valide un filtre de marchand."""
        if isinstance(value, list):
            if len(value) > VALIDATION_LIMITS.get("max_filter_values", 100):
                result.add_error(f"Trop de marchands (max: {VALIDATION_LIMITS.get('max_filter_values', 100)})")
            
            for i, merchant in enumerate(value):
                if not isinstance(merchant, str):
                    result.add_error(f"Nom de marchand invalide à l'index {i}")
                elif len(merchant.strip()) == 0:
                    result.add_error(f"Nom de marchand vide à l'index {i}")
                elif len(merchant) > 200:
                    result.add_error(f"Nom de marchand trop long à l'index {i}")
        elif isinstance(value, str):
            if len(value.strip()) == 0:
                result.add_error("Nom de marchand ne peut pas être vide")
            elif len(value) > 200:
                result.add_error("Nom de marchand trop long")
        else:
            result.add_error("Les noms de marchands doivent être des chaînes ou listes")
    
    def _validate_type_filter(self, value: Any, result: ValidationResult):
        """Valide un filtre de type de transaction."""
        valid_types = {
            "card_payment", "transfer", "withdrawal", "deposit", 
            "direct_debit", "check", "fee", "interest", "other"
        }
        
        if isinstance(value, list):
            for i, trans_type in enumerate(value):
                if not isinstance(trans_type, str):
                    result.add_error(f"Type de transaction invalide à l'index {i}")
                elif trans_type.lower() not in valid_types:
                    result.add_warning(f"Type de transaction non standard à l'index {i}: {trans_type}")
        elif isinstance(value, str):
            if value.lower() not in valid_types:
                result.add_warning(f"Type de transaction non standard: {value}")
        else:
            result.add_error("Les types de transaction doivent être des chaînes ou listes")
    
    def _validate_range_filter(self, field: str, value: Any, result: ValidationResult):
        """Valide un filtre de plage (range)."""
        if not isinstance(value, dict):
            result.add_error(f"Le filtre de plage {field} doit être un dictionnaire")
            return
        
        valid_operators = {"min", "max", "gte", "lte", "gt", "lt", "from", "to"}
        found_operators = set(value.keys())
        
        if not found_operators.intersection(valid_operators):
            result.add_error(f"Aucun opérateur de plage valide trouvé dans {field}")
            return
        
        invalid_operators = found_operators - valid_operators
        if invalid_operators:
            result.add_warning(f"Opérateurs non standard dans {field}: {invalid_operators}")
        
        # Validation des valeurs selon le type de champ
        if "amount" in field:
            for op, val in value.items():
                if op in valid_operators and not validate_amount(val):
                    result.add_error(f"Valeur invalide pour {field}.{op}")
        elif "date" in field:
            for op, val in value.items():
                if op in valid_operators and not validate_date(val):
                    result.add_error(f"Date invalide pour {field}.{op}")
    
    def _validate_generic_filter(self, field: str, value: Any, result: ValidationResult):
        """Valide un filtre générique."""
        # Validation basique du type
        if value is None:
            result.add_warning(f"Valeur nulle pour le filtre {field}")
            return
        
        # Validation de la longueur pour les chaînes
        if isinstance(value, str):
            max_length = DATA_TYPE_LIMITS.get("string_field", {}).get("max_length", 1000)
            if len(value) > max_length:
                result.add_error(f"Valeur trop longue pour {field} (max: {max_length})")
        
        # Validation des listes
        elif isinstance(value, list):
            if len(value) > VALIDATION_LIMITS.get("max_filter_values", 100):
                result.add_error(f"Trop de valeurs pour {field} (max: {VALIDATION_LIMITS.get('max_filter_values', 100)})")
    
    def _validate_filter_combinations(self, filters: Dict[str, Any], result: ValidationResult):
        """Valide les combinaisons de filtres."""
        # Vérification des conflits de plages
        if "amount_min" in filters and "amount_max" in filters:
            try:
                min_val = float(filters["amount_min"])
                max_val = float(filters["amount_max"])
                if min_val > max_val:
                    result.add_error("amount_min ne peut pas être supérieur à amount_max")
            except (ValueError, TypeError):
                pass  # Erreur déjà signalée dans la validation individuelle
        
        # Vérification des dates
        date_fields = [field for field in filters.keys() if "date" in field]
        if len(date_fields) > 1:
            result.add_warning("Multiples filtres de date peuvent créer des conflits")
        
        # Vérification de la cohérence user_id
        if "user_id" in filters:
            user_id = filters["user_id"]
            # Vérifier que tous les autres filtres sont cohérents avec cet utilisateur
            result.add_security_flag(f"Filtre utilisateur appliqué: {user_id}")
    
    def _validate_security(self, filters: Dict[str, Any], result: ValidationResult):
        """Valide la sécurité des filtres."""
        for field, value in filters.items():
            if isinstance(value, str):
                if not self._check_security_patterns(value, result):
                    if self.config.get("block_dangerous_patterns", False):
                        result.add_error(f"Pattern dangereux détecté dans le filtre {field}")
    
    def _sanitize_filter_value(self, field: str, value: Any) -> Any:
        """Sanitise la valeur d'un filtre."""
        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, list):
            return [self._sanitize_filter_value(field, item) for item in value]
        elif isinstance(value, dict):
            return {k: self._sanitize_filter_value(field, v) for k, v in value.items()}
        else:
            return value
    
    def _check_numeric_limits(self, value: Any, limits: Dict[str, Any], 
                            result: ValidationResult, field: str):
        """Vérifie les limites numériques."""
        if not isinstance(value, (int, float)):
            return
        
        min_val = limits.get("min_value")
        max_val = limits.get("max_value")
        
        if min_val is not None and value < min_val:
            result.add_error(f"Valeur trop petite pour {field} (min: {min_val})")
        
        if max_val is not None and value > max_val:
            result.add_error(f"Valeur trop grande pour {field} (max: {max_val})")

# ==================== FONCTIONS UTILITAIRES ====================

def validate_filters_dict(filters: Dict[str, Any], 
                         validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
    """
    Fonction utilitaire pour valider un dictionnaire de filtres.
    
    Args:
        filters: Dictionnaire des filtres
        validation_level: Niveau de validation
        
    Returns:
        ValidationResult
    """
    validator = FilterValidator(validation_level)
    return validator.validate(filters)

def sanitize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitise un dictionnaire de filtres.
    
    Args:
        filters: Filtres à sanitiser
        
    Returns:
        Filtres sanitisés
    """
    validator = FilterValidator(ValidationLevel.STANDARD)
    result = validator.validate(filters)
    return result.sanitized_data if result.sanitized_data else filters

def build_elasticsearch_filters(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convertit les filtres en clauses Elasticsearch.
    
    Args:
        filters: Dictionnaire des filtres
        
    Returns:
        Liste des clauses de filtre Elasticsearch
    """
    es_filters = []
    
    for field, value in filters.items():
        if field.endswith("_range") or isinstance(value, dict) and any(k in value for k in ["min", "max", "gte", "lte"]):
            # Filtre range
            range_clause = {"range": {field.replace("_range", ""): {}}}
            
            if isinstance(value, dict):
                for op, val in value.items():
                    if op in ["min", "gte"]:
                        range_clause["range"][field.replace("_range", "")]["gte"] = val
                    elif op in ["max", "lte"]:
                        range_clause["range"][field.replace("_range", "")]["lte"] = val
                    elif op == "gt":
                        range_clause["range"][field.replace("_range", "")]["gt"] = val
                    elif op == "lt":
                        range_clause["range"][field.replace("_range", "")]["lt"] = val
            
            es_filters.append(range_clause)
        
        elif isinstance(value, list):
            # Filtre terms
            es_filters.append({"terms": {field: value}})
        
        else:
            # Filtre term
            es_filters.append({"term": {field: value}})
    
    return es_filters