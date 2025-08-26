"""
Utilitaires de validation optimisés pour conversation service
"""
import logging
import json
from typing import Any, Dict, List, Optional
from conversation_service.models.responses.conversation_responses import IntentClassificationResult
from conversation_service.prompts.harena_intents import HarenaIntentType, INTENT_CATEGORIES
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.validation")

async def validate_intent_response(classification_result: IntentClassificationResult) -> bool:
    """
    Validation qualité résultat classification intention avec règles dynamiques
    
    Args:
        classification_result: Résultat de classification à valider
        
    Returns:
        bool: True si validation réussie, False sinon
    """
    
    try:
        # Validation existence intention dans taxonomie
        if not _is_valid_harena_intent(classification_result.intent_type):
            logger.error(f"Intention non reconnue: {classification_result.intent_type}")
            return False
        
        # Validation confidence dans plage valide
        if not _is_valid_confidence(classification_result.confidence):
            logger.error(f"Confidence invalide: {classification_result.confidence}")
            return False
        
        # Validation seuil minimum avec logique flexible
        min_threshold = getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5)
        if classification_result.confidence < min_threshold:
            # Warning mais pas échec pour intentions ambiguës légitimes
            if classification_result.intent_type not in [HarenaIntentType.UNCLEAR_INTENT, HarenaIntentType.UNKNOWN]:
                logger.warning(f"Confidence sous seuil: {classification_result.confidence} < {min_threshold}")
        
        # Validation cohérence catégorie avec correction automatique
        expected_category = _determine_intent_category(classification_result.intent_type)
        if classification_result.category != expected_category:
            logger.warning(f"Catégorie incohérente: {classification_result.category} != {expected_category}")
            # Correction automatique pour robustesse
            classification_result.category = expected_category
        
        # Validation alternatives avec nettoyage
        validated_alternatives = []
        for alt in classification_result.alternatives:
            if _validate_alternative(alt):
                validated_alternatives.append(alt)
            else:
                logger.debug(f"Alternative invalide ignorée: {alt}")
        
        classification_result.alternatives = validated_alternatives
        
        # Validation reasoning non vide avec tolérance
        if not classification_result.reasoning or len(classification_result.reasoning.strip()) < 3:
            logger.warning("Reasoning très court ou vide")
            # Pas d'échec - génération automatique si nécessaire
            if not classification_result.reasoning:
                classification_result.reasoning = f"Classification: {classification_result.intent_type.value}"
        
        # Validation cohérence support
        actual_support = _is_intent_actually_supported(classification_result.intent_type)
        if classification_result.is_supported != actual_support:
            logger.debug(f"Correction support intention: {classification_result.intent_type.value} -> {actual_support}")
            classification_result.is_supported = actual_support
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur validation classification: {str(e)}")
        return False

def _is_valid_harena_intent(intent_type: HarenaIntentType) -> bool:
    """Validation dynamique que l'intention fait partie de la taxonomie Harena"""
    try:
        return isinstance(intent_type, HarenaIntentType)
    except (ValueError, TypeError):
        return False

def _is_valid_confidence(confidence: float) -> bool:
    """Validation plage confidence avec tolérance numérique"""
    try:
        return isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
    except (ValueError, TypeError):
        return False

def _determine_intent_category(intent_type: HarenaIntentType) -> str:
    """Détermine dynamiquement la catégorie d'une intention"""
    for category, intents in INTENT_CATEGORIES.items():
        if intent_type in intents:
            return category
    
    logger.warning(f"Catégorie non trouvée pour {intent_type}, fallback vers UNKNOWN_CATEGORY")
    return "UNKNOWN_CATEGORY"

def _is_intent_actually_supported(intent_type: HarenaIntentType) -> bool:
    """Vérification dynamique si intention est supportée"""
    unsupported = INTENT_CATEGORIES.get("UNSUPPORTED", [])
    return intent_type not in unsupported

def _validate_alternative(alternative) -> bool:
    """Validation alternative intention"""
    try:
        return (
            hasattr(alternative, 'intent_type') and
            hasattr(alternative, 'confidence') and
            _is_valid_harena_intent(alternative.intent_type) and
            _is_valid_confidence(alternative.confidence)
        )
    except Exception:
        return False

def validate_user_message(message: str) -> Dict[str, Any]:
    """
    Validation message utilisateur avec règles de sécurité flexibles
    
    Args:
        message: Message utilisateur à valider
        
    Returns:
        Dict: Résultat validation avec erreurs et warnings
    """
    
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "sanitized_message": ""
    }
    
    # Validation message non vide
    if not message or not message.strip():
        validation_result["valid"] = False
        validation_result["errors"].append("Message vide")
        return validation_result
    
    # Nettoyage préliminaire
    cleaned_message = message.strip()
    validation_result["sanitized_message"] = cleaned_message
    
    # Validation longueur avec limites configurables
    max_length = getattr(settings, 'MAX_MESSAGE_LENGTH', 1000)
    if len(cleaned_message) > max_length:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Message trop long (>{max_length} caractères)")
    
    min_length = getattr(settings, 'MIN_MESSAGE_LENGTH', 1)
    if len(cleaned_message) < min_length:
        validation_result["warnings"].append("Message très court")
    
    # Détection patterns suspects avec tolérance
    suspicious_patterns = _detect_suspicious_patterns(cleaned_message)
    if suspicious_patterns:
        validation_result["warnings"].extend(suspicious_patterns)
    
    # Validation encodage et caractères
    encoding_issues = _validate_message_encoding(cleaned_message)
    if encoding_issues:
        validation_result["warnings"].extend(encoding_issues)
    
    # Validation structure basique
    structure_issues = _validate_message_structure(cleaned_message)
    if structure_issues:
        validation_result["warnings"].extend(structure_issues)
    
    return validation_result

def _detect_suspicious_patterns(message: str) -> List[str]:
    """Détection patterns suspects avec logique flexible"""
    warnings = []
    message_upper = message.upper()
    
    # Patterns injection potentielle
    injection_patterns = ['<SCRIPT', '<?PHP', 'JAVASCRIPT:', 'SELECT ', 'DROP TABLE', 'UNION SELECT']
    detected_injections = [pattern for pattern in injection_patterns if pattern in message_upper]
    
    if detected_injections:
        warnings.append(f"Patterns suspects détectés: {', '.join(detected_injections)}")
    
    # Répétition excessive caractères
    if len(set(message.replace(' ', ''))) < max(3, len(message) / 20):
        warnings.append("Répétition excessive de caractères")
    
    # Trop de caractères spéciaux
    special_char_ratio = sum(1 for c in message if not c.isalnum() and c not in ' .,!?-\'') / max(len(message), 1)
    if special_char_ratio > 0.5:
        warnings.append("Proportion élevée de caractères spéciaux")
    
    return warnings

def _validate_message_encoding(message: str) -> List[str]:
    """Validation encodage et caractères du message"""
    warnings = []
    
    # Caractères de contrôle
    control_chars = [c for c in message if ord(c) < 32 and c not in '\n\t\r']
    if control_chars:
        warnings.append("Caractères de contrôle détectés")
    
    # Caractères non-printables
    non_printable = sum(1 for c in message if not c.isprintable() and c not in '\n\t')
    if non_printable > 0:
        warnings.append(f"{non_printable} caractères non-imprimables")
    
    return warnings

def _validate_message_structure(message: str) -> List[str]:
    """Validation structure basique du message"""
    warnings = []
    
    # Messages avec uniquement ponctuation
    if not any(c.isalnum() for c in message):
        warnings.append("Message sans caractères alphanumériques")
    
    # Messages avec uniquement majuscules (potentiellement spam)
    if len(message) > 10 and message.isupper():
        warnings.append("Message entièrement en majuscules")
    
    # Messages avec uniquement chiffres
    if message.replace(' ', '').isdigit() and len(message) > 5:
        warnings.append("Message composé uniquement de chiffres")
    
    return warnings

def sanitize_user_input(message: str) -> str:
    """
    Nettoyage sécurisé et robuste de l'input utilisateur
    
    Args:
        message: Message à nettoyer
        
    Returns:
        str: Message nettoyé et sécurisé
    """
    
    if not message:
        return ""
    
    # Suppression caractères de contrôle dangereux
    sanitized = ''.join(
        char for char in message 
        if ord(char) >= 32 or char in '\n\t\r'
    )
    
    # Limitation longueur avec limite configurable
    max_length = getattr(settings, 'MAX_MESSAGE_LENGTH', 1000)
    sanitized = sanitized[:max_length]
    
    # Nettoyage whitespace excessif
    sanitized = ' '.join(sanitized.split())
    
    # Suppression patterns dangereux basiques
    dangerous_patterns = ['<script', '</script', '<?php', '<?xml']
    for pattern in dangerous_patterns:
        sanitized = sanitized.replace(pattern.lower(), '')
        sanitized = sanitized.replace(pattern.upper(), '')
    
    return sanitized.strip()

def validate_deepseek_response(response: Dict[str, Any]) -> bool:
    """
    Validation robuste structure réponse DeepSeek avec JSON Output
    
    Args:
        response: Réponse DeepSeek à valider
        
    Returns:
        bool: True si structure valide, False sinon
    """
    
    try:
        # Validation structure de base
        if not isinstance(response, dict):
            logger.error("Réponse DeepSeek n'est pas un dictionnaire")
            return False
        
        if "choices" not in response:
            logger.error("Réponse DeepSeek manque 'choices'")
            return False
        
        choices = response["choices"]
        if not isinstance(choices, list) or len(choices) == 0:
            logger.error("Réponse DeepSeek choices vide ou invalide")
            return False
        
        # Validation premier choice
        first_choice = choices[0]
        if not isinstance(first_choice, dict) or "message" not in first_choice:
            logger.error("Choice manque 'message'")
            return False
        
        message = first_choice["message"]
        if not isinstance(message, dict) or "content" not in message:
            logger.error("Message manque 'content'")
            return False
        
        content = message["content"]
        if not isinstance(content, str) or len(content.strip()) == 0:
            logger.error("Content vide ou invalide")
            return False
        
        # Validation JSON si response_format était demandé
        if _appears_to_be_json_response(response):
            if not _validate_json_content(content):
                logger.error("JSON Output attendu mais contenu invalide")
                return False
        
        # Validation usage tokens si disponible
        if "usage" in response:
            if not _validate_token_usage(response["usage"]):
                logger.warning("Usage tokens invalide ou excessif")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur validation réponse DeepSeek: {str(e)}")
        return False

def _appears_to_be_json_response(response: Dict[str, Any]) -> bool:
    """Détecte si la réponse était censée être du JSON"""
    # Heuristique: si model ou content suggère JSON
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content.strip().startswith('{') and content.strip().endswith('}')

def _validate_json_content(content: str) -> bool:
    """Validation que le contenu est du JSON valide"""
    try:
        parsed = json.loads(content.strip())
        return isinstance(parsed, dict)  # JSON Object attendu
    except (json.JSONDecodeError, TypeError):
        return False

def _validate_token_usage(usage: Dict[str, Any]) -> bool:
    """Validation usage tokens avec seuils configurables"""
    try:
        total_tokens = usage.get("total_tokens", 0)
        max_tokens = getattr(settings, 'DEEPSEEK_MAX_TOKENS', 8192)
        
        if total_tokens > max_tokens * 1.5:
            logger.warning(f"Usage tokens très élevé: {total_tokens}")
            return False
        
        return True
        
    except (ValueError, TypeError):
        return False

def calculate_confidence_adjustment(
    base_confidence: float, 
    context_factors: Dict[str, Any]
) -> float:
    """
    Ajustement dynamique confidence selon contexte avec règles configurables
    
    Args:
        base_confidence: Confidence initiale
        context_factors: Facteurs contextuels
        
    Returns:
        float: Confidence ajustée
    """
    
    if not _is_valid_confidence(base_confidence):
        logger.warning(f"Base confidence invalide: {base_confidence}")
        return 0.5
    
    adjusted_confidence = base_confidence
    
    # Facteurs d'ajustement avec pondération configurable
    adjustments = []
    
    # Facteur longueur message
    message_length = context_factors.get("message_length", 0)
    if message_length < 3:
        adjustments.append(-0.1)  # Pénalité message trop court
    elif message_length > 100:
        adjustments.append(0.05)  # Bonus message détaillé
    
    # Facteur alternatives
    alternatives_count = context_factors.get("alternatives_count", 0)
    if alternatives_count > 0:
        adjustments.append(-0.05 * alternatives_count)  # Pénalité si alternatives
    
    # Facteur contexte utilisateur
    has_context = context_factors.get("has_user_context", False)
    if has_context:
        adjustments.append(0.03)  # Bonus contexte disponible
    
    # Facteur qualité message
    quality_score = context_factors.get("message_quality", 1.0)
    if quality_score < 0.8:
        adjustments.append(-0.08)  # Pénalité qualité faible
    
    # Application ajustements avec limite
    total_adjustment = sum(adjustments)
    adjusted_confidence += total_adjustment
    
    # Maintenir dans [0, 1] avec marge de sécurité
    return max(0.1, min(0.99, adjusted_confidence))

def get_unsupported_intent_message(intent_type: str) -> str:
    """
    Message contextuel pour intentions non supportées avec personnalisation
    
    Args:
        intent_type: Type d'intention non supportée
        
    Returns:
        str: Message explicatif personnalisé
    """
    
    # Messages personnalisés par intention avec suggestions
    contextual_messages = {
        "TRANSFER_REQUEST": {
            "message": "Les virements ne sont pas supportés dans cette interface.",
            "suggestion": "Utilisez votre application bancaire ou le site web de votre banque."
        },
        "PAYMENT_REQUEST": {
            "message": "Les paiements ne sont pas supportés dans cette interface.",
            "suggestion": "Utilisez votre application bancaire pour effectuer des paiements."
        },
        "CARD_BLOCK": {
            "message": "Le blocage de carte ne peut pas être effectué ici.",
            "suggestion": "Contactez immédiatement votre banque au numéro d'urgence."
        },
        "BUDGET_INQUIRY": {
            "message": "La gestion de budget n'est pas encore disponible.",
            "suggestion": "Cette fonctionnalité sera ajoutée dans une future mise à jour."
        },
        "GOAL_TRACKING": {
            "message": "Le suivi d'objectifs financiers n'est pas encore disponible.",
            "suggestion": "Nous travaillons sur cette fonctionnalité pour une prochaine version."
        },
        "EXPORT_REQUEST": {
            "message": "L'export de données n'est pas encore disponible.",
            "suggestion": "Cette fonctionnalité sera bientôt disponible."
        },
        "OUT_OF_SCOPE": {
            "message": "Cette demande est en dehors du domaine financier.",
            "suggestion": "Je peux vous aider avec vos transactions, dépenses et soldes."
        }
    }
    
    # Récupération message contextuel ou fallback
    intent_info = contextual_messages.get(intent_type)
    if intent_info:
        return f"{intent_info['message']} {intent_info['suggestion']}"
    
    # Message générique pour intentions inconnues
    return "Cette fonctionnalité n'est pas encore supportée. Nous travaillons constamment à améliorer nos services."

def validate_cache_key(cache_key: str) -> bool:
    """
    Validation clé cache avec règles de format
    
    Args:
        cache_key: Clé de cache à valider
        
    Returns:
        bool: True si valide, False sinon
    """
    try:
        return (
            isinstance(cache_key, str) and
            len(cache_key) > 0 and
            len(cache_key) <= 250 and  # Limite Redis
            cache_key.isascii()  # ASCII seulement pour compatibilité
        )
    except Exception:
        return False