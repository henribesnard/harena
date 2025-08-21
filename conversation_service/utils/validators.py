"""
Validation système pour contrats et données Conversation Service MVP.

Ce module fournit des validateurs pour tous les contrats de service,
modèles de données et interfaces API. Optimisé pour performance avec
validation stricte configurable.

Responsabilités :
- Validation contrats Search Service
- Validation modèles financial_models
- Validation agent responses
- Reporting erreurs structuré

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import re
from config_service.config import settings

# Imports Pydantic pour validation
from pydantic import ValidationError as PydanticValidationError

# Imports locaux des modèles (à ajuster selon structure)
try:
    from ..models.service_contracts import (
        SearchServiceQuery,
        SearchServiceResponse,
        QueryMetadata,
        SearchParameters,
        SearchFilters
    )
    from ..models.financial_models import (
        IntentResult,
        FinancialEntity,
        EntityType,
        IntentCategory
    )
    MODELS_AVAILABLE = True
except ImportError:
    # Graceful fallback si les modèles ne sont pas encore disponibles
    MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception personnalisée pour erreurs de validation."""
    
    def __init__(self, message: str, errors: List[str] = None, field: str = None):
        super().__init__(message)
        self.errors = errors or []
        self.field = field
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'erreur en dictionnaire pour logging/API."""
        return {
            "message": str(self),
            "errors": self.errors,
            "field": self.field,
            "timestamp": self.timestamp.isoformat(),
            "error_type": "validation_error"
        }


class ContractValidator:
    """
    Validateur principal pour tous les contrats de service.
    
    Features :
    - Validation Pydantic native
    - Règles métier spécialisées
    - Performance optimisée
    - Reporting détaillé des erreurs
    """
    
    def __init__(self, strict_mode: bool = None):
        """
        Initialise le validateur.
        
        Args:
            strict_mode: Mode strict (défaut depuis env var)
        """
        strict_env = settings.VALIDATION_STRICT
        self.strict_mode = strict_mode if strict_mode is not None else strict_env
        
        logger.debug(f"ContractValidator initialized: strict_mode={self.strict_mode}")
    
    @staticmethod
    def validate_search_query(query: Union[Dict[str, Any], 'SearchServiceQuery']) -> List[str]:
        """
        Valide un contrat SearchServiceQuery.
        
        Args:
            query: Query à valider (dict ou modèle Pydantic)
            
        Returns:
            Liste des erreurs (vide si valide)
        """
        errors = []
        
        if not MODELS_AVAILABLE:
            logger.warning("Models not available - skipping Pydantic validation")
            return ContractValidator._validate_search_query_basic(query)
        
        try:
            # Si c'est un dict, essaie de créer le modèle Pydantic
            if isinstance(query, dict):
                SearchServiceQuery(**query)
            elif hasattr(query, 'model_validate'):
                # Déjà un modèle Pydantic, validation automatique
                pass
            else:
                errors.append("Query must be dict or SearchServiceQuery instance")
                return errors
            
        except PydanticValidationError as e:
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                errors.append(f"{field_path}: {error['msg']}")
        
        # Validations métier additionnelles
        if isinstance(query, dict):
            errors.extend(ContractValidator._validate_search_query_business_rules(query))
        
        return errors
    
    @staticmethod
    def _validate_search_query_basic(query: Dict[str, Any]) -> List[str]:
        """Validation basique sans Pydantic (fallback)."""
        errors = []
        
        # Vérifications de base
        if not isinstance(query, dict):
            errors.append("Query must be a dictionary")
            return errors
        
        # Vérification structure minimale
        required_fields = ["query_metadata", "search_parameters", "filters"]
        for field in required_fields:
            if field not in query:
                errors.append(f"Missing required field: {field}")
        
        # Validation query_metadata
        if "query_metadata" in query:
            metadata = query["query_metadata"]
            if not isinstance(metadata, dict):
                errors.append("query_metadata must be a dictionary")
            else:
                required_metadata = ["conversation_id", "user_id", "intent_type"]
                for field in required_metadata:
                    if field not in metadata:
                        errors.append(f"query_metadata missing: {field}")
        
        return errors
    
    @staticmethod
    def _validate_search_query_business_rules(query: Dict[str, Any]) -> List[str]:
        """Validation des règles métier pour SearchServiceQuery."""
        errors = []
        
        # Validation des paramètres de recherche
        if "search_parameters" in query:
            params = query["search_parameters"]
            
            if "max_results" in params:
                max_results = params["max_results"]
                if not isinstance(max_results, int) or max_results <= 0 or max_results > 1000:
                    errors.append("max_results must be positive integer ≤ 1000")
            
            if "search_strategy" in params:
                strategy = params["search_strategy"]
                valid_strategies = ["lexical", "semantic"]
                if strategy not in valid_strategies:
                    errors.append(f"search_strategy must be one of: {valid_strategies}")
        
        # Validation des filtres
        if "filters" in query:
            filters = query["filters"]
            
            # Validation date si présent
            if "date" in filters:
                date_filter = filters["date"]
                if isinstance(date_filter, dict):
                    if "gte" in date_filter and "lte" in date_filter:
                        try:
                            start = datetime.fromisoformat(date_filter["gte"].replace('Z', '+00:00'))
                            end = datetime.fromisoformat(date_filter["lte"].replace('Z', '+00:00'))
                            if start > end:
                                errors.append("date: gte must be before lte")
                        except (ValueError, TypeError):
                            errors.append("date: invalid date format (use ISO 8601)")

            # Validation amount si présent
            if "amount" in filters:
                amount_filter = filters["amount"]
                if isinstance(amount_filter, dict):
                    if "gte" in amount_filter and "lte" in amount_filter:
                        if amount_filter["gte"] > amount_filter["lte"]:
                            errors.append("amount: gte must be <= lte")
        
        return errors
    
    @staticmethod
    def validate_search_response(response: Union[Dict[str, Any], 'SearchServiceResponse']) -> List[str]:
        """
        Valide un contrat SearchServiceResponse.
        
        Args:
            response: Response à valider
            
        Returns:
            Liste des erreurs
        """
        errors = []
        
        if not MODELS_AVAILABLE:
            return ContractValidator._validate_search_response_basic(response)
        
        try:
            if isinstance(response, dict):
                SearchServiceResponse(**response)
            
        except PydanticValidationError as e:
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                errors.append(f"{field_path}: {error['msg']}")
        
        return errors
    
    @staticmethod
    def _validate_search_response_basic(response: Dict[str, Any]) -> List[str]:
        """Validation basique SearchServiceResponse."""
        errors = []
        
        if not isinstance(response, dict):
            errors.append("Response must be a dictionary")
            return errors
        
        # Structure minimale
        required_fields = ["response_metadata", "results"]
        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")
        
        # Validation response_metadata
        if "response_metadata" in response:
            metadata = response["response_metadata"]
            if not isinstance(metadata, dict):
                errors.append("response_metadata must be a dictionary")
            else:
                if "total_results" not in metadata:
                    errors.append("response_metadata missing: total_results")
                if "returned_results" not in metadata:
                    errors.append("response_metadata missing: returned_results")
                if "processing_time_ms" not in metadata:
                    errors.append("response_metadata missing: processing_time_ms")
        
        # Validation results
        if "results" in response:
            results = response["results"]
            if not isinstance(results, list):
                errors.append("results must be a list")
        
        return errors
    
    @staticmethod
    def validate_intent_result(result: Union[Dict[str, Any], 'IntentResult']) -> List[str]:
        """
        Valide un IntentResult.
        
        Args:
            result: IntentResult à valider
            
        Returns:
            Liste des erreurs
        """
        errors = []
        
        if not MODELS_AVAILABLE:
            return ContractValidator._validate_intent_result_basic(result)
        
        try:
            if isinstance(result, dict):
                IntentResult(**result)
            
        except PydanticValidationError as e:
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                errors.append(f"{field_path}: {error['msg']}")
        
        # Validation métier
        if isinstance(result, dict):
            errors.extend(ContractValidator._validate_intent_result_business_rules(result))
        
        return errors
    
    @staticmethod
    def _validate_intent_result_basic(result: Dict[str, Any]) -> List[str]:
        """Validation basique IntentResult."""
        errors = []
        
        if not isinstance(result, dict):
            errors.append("IntentResult must be a dictionary")
            return errors
        
        # Champs requis
        required_fields = ["intent_type", "intent_category", "confidence", "method"]
        for field in required_fields:
            if field not in result:
                errors.append(f"Missing required field: {field}")
        
        return errors
    
    @staticmethod
    def _validate_intent_result_business_rules(result: Dict[str, Any]) -> List[str]:
        """Validation règles métier IntentResult."""
        errors = []
        
        # Validation confidence
        if "confidence" in result:
            confidence = result["confidence"]
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                errors.append("confidence must be float between 0.0 and 1.0")
        
        # Validation intent_category contre enum si disponible
        if "intent_category" in result and MODELS_AVAILABLE:
            category = result["intent_category"]
            try:
                IntentCategory(category)
            except ValueError:
                valid_categories = [cat.value for cat in IntentCategory]
                errors.append(f"intent_category must be one of: {valid_categories}")
        
        # Validation entities
        if "entities" in result:
            entities = result["entities"]
            if not isinstance(entities, list):
                errors.append("entities must be a list")
            else:
                for i, entity in enumerate(entities):
                    if not isinstance(entity, dict):
                        errors.append(f"entities[{i}] must be a dictionary")
                        continue
                    
                    # Validation entité individuelle
                    entity_errors = ContractValidator._validate_financial_entity(entity)
                    for error in entity_errors:
                        errors.append(f"entities[{i}].{error}")
        
        return errors
    
    @staticmethod
    def _validate_financial_entity(entity: Dict[str, Any]) -> List[str]:
        """Validation d'une entité financière."""
        errors = []
        
        # Champs requis
        required_fields = ["entity_type", "raw_value", "confidence"]
        for field in required_fields:
            if field not in entity:
                errors.append(f"missing {field}")
        
        # Validation entity_type
        if "entity_type" in entity and MODELS_AVAILABLE:
            entity_type = entity["entity_type"]
            try:
                EntityType(entity_type)
            except ValueError:
                valid_types = [t.value for t in EntityType]
                errors.append(f"entity_type must be one of: {valid_types}")
        
        # Validation confidence
        if "confidence" in entity:
            confidence = entity["confidence"]
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                errors.append("confidence must be float between 0.0 and 1.0")
        
        return errors
    
    def validate_agent_response(self, response: Dict[str, Any]) -> List[str]:
        """
        Valide une réponse d'agent AutoGen.
        
        Args:
            response: Réponse à valider
            
        Returns:
            Liste des erreurs
        """
        errors = []
        
        if not isinstance(response, dict):
            errors.append("Agent response must be a dictionary")
            return errors
        
        # Structure de base attendue
        required_fields = ["agent_name", "content"]
        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")
        
        # Validation contenu
        if "content" in response:
            content = response["content"]
            if not isinstance(content, str):
                errors.append("content must be a string")
            elif len(content.strip()) == 0:
                errors.append("content cannot be empty")
            elif len(content) > 10000:  # Limite raisonnable
                errors.append("content too long (max 10000 characters)")
        
        # Validation metadata si présent
        if "metadata" in response:
            metadata = response["metadata"]
            if not isinstance(metadata, dict):
                errors.append("metadata must be a dictionary")
        
        return errors
    
    def validate_conversation_context(self, context: Dict[str, Any]) -> List[str]:
        """Valide un contexte de conversation."""
        errors = []
        
        if not isinstance(context, dict):
            errors.append("Conversation context must be a dictionary")
            return errors
        
        # Validation de base
        required_fields = ["conversation_id", "user_id", "turns"]
        for field in required_fields:
            if field not in context:
                errors.append(f"Missing required field: {field}")
        
        # Validation turns
        if "turns" in context:
            turns = context["turns"]
            if not isinstance(turns, list):
                errors.append("turns must be a list")
            else:
                for i, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        errors.append(f"turns[{i}] must be a dictionary")
                        continue
                    
                    # Validation turn individuel
                    turn_required = ["turn_id", "user_message", "assistant_response"]
                    for field in turn_required:
                        if field not in turn:
                            errors.append(f"turns[{i}] missing: {field}")
        
        return errors


# Fonctions utilitaires pour validation rapide
def validate_search_query_contract(query: Union[Dict[str, Any], 'SearchServiceQuery']) -> List[str]:
    """
    Validation rapide SearchServiceQuery.
    
    Args:
        query: Query à valider
        
    Returns:
        Liste des erreurs
    """
    return ContractValidator.validate_search_query(query)


def validate_search_response_contract(response: Union[Dict[str, Any], 'SearchServiceResponse']) -> List[str]:
    """
    Validation rapide SearchServiceResponse.
    
    Args:
        response: Response à valider
        
    Returns:
        Liste des erreurs
    """
    return ContractValidator.validate_search_response(response)


def validate_intent_result_contract(result: Union[Dict[str, Any], 'IntentResult']) -> List[str]:
    """
    Validation rapide IntentResult.
    
    Args:
        result: IntentResult à valider
        
    Returns:
        Liste des erreurs
    """
    return ContractValidator.validate_intent_result(result)


def is_valid_uuid(uuid_string: str) -> bool:
    """
    Valide un UUID.
    
    Args:
        uuid_string: String à valider
        
    Returns:
        True si UUID valide
    """
    import uuid
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False


def is_valid_email(email: str) -> bool:
    """
    Valide une adresse email basique.
    
    Args:
        email: Email à valider
        
    Returns:
        True si email valide
    """
    if not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Nettoie et tronque une string.
    
    Args:
        value: String à nettoyer
        max_length: Longueur maximale
        
    Returns:
        String nettoyée
    """
    if not isinstance(value, str):
        return str(value)
    
    # Supprime caractères de contrôle
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
    
    # Tronque si nécessaire
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    
    return cleaned.strip()


# Validation rapide pour types communs
def validate_positive_integer(value: Any, field_name: str = "value") -> Optional[str]:
    """Valide un entier positif."""
    if not isinstance(value, int):
        return f"{field_name} must be an integer"
    if value <= 0:
        return f"{field_name} must be positive"
    return None


def validate_probability(value: Any, field_name: str = "value") -> Optional[str]:
    """Valide une probabilité (0.0-1.0)."""
    if not isinstance(value, (int, float)):
        return f"{field_name} must be a number"
    if not (0.0 <= value <= 1.0):
        return f"{field_name} must be between 0.0 and 1.0"
    return None


def validate_non_empty_string(value: Any, field_name: str = "value") -> Optional[str]:
    """Valide une string non vide."""
    if not isinstance(value, str):
        return f"{field_name} must be a string"
    if len(value.strip()) == 0:
        return f"{field_name} cannot be empty"
    return None
    return re.match(pattern, email) is not None


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Nettoie et tronque une string.
    
    Args:
        value: String à nettoyer
        max_length: Longueur maximale
        
    Returns:
        String nettoyée
    """
    if not isinstance(value, str):
        return str(value)
    
    # Supprime caractères de contrôle
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
    
    # Tronque si nécessaire
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    
    return cleaned.strip()


# Validation rapide pour types communs
def validate_positive_integer(value: Any, field_name: str = "value") -> Optional[str]:
    """Valide un entier positif."""
    if not isinstance(value, int):
        return f"{field_name} must be an integer"
    if value <= 0:
        return f"{field_name} must be positive"
    return None


def validate_probability(value: Any, field_name: str = "value") -> Optional[str]:
    """Valide une probabilité (0.0-1.0)."""
    if not isinstance(value, (int, float)):
        return f"{field_name} must be a number"
    if not (0.0 <= value <= 1.0):
        return f"{field_name} must be between 0.0 and 1.0"
    return None


def validate_non_empty_string(value: Any, field_name: str = "value") -> Optional[str]:
    """Valide une string non vide."""
    if not isinstance(value, str):
        return f"{field_name} must be a string"
    if len(value.strip()) == 0:
        return f"{field_name} cannot be empty"
    return None