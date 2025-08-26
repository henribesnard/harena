"""
Utilitaires de validation pour conversation service
"""
import logging
from typing import Any, Dict, List
from conversation_service.models.responses.conversation_responses import IntentClassificationResult
from conversation_service.prompts.harena_intents import HarenaIntentType, INTENT_CATEGORIES
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.validation")

async def validate_intent_response(classification_result: IntentClassificationResult) -> bool:
    """Validation qualité résultat classification intention"""
    
    try:
        # Validation intention supportée
        if classification_result.intent_type not in [intent.value for intent in HarenaIntentType]:
            logger.error(f"Intention non reconnue: {classification_result.intent_type}")
            return False
        
        # Validation confidence
        if not (0.0 <= classification_result.confidence <= 1.0):
            logger.error(f"Confidence invalide: {classification_result.confidence}")
            return False
        
        # Validation seuil minimum
        if classification_result.confidence < settings.MIN_CONFIDENCE_THRESHOLD:
            logger.warning(f"Confidence sous seuil: {classification_result.confidence} < {settings.MIN_CONFIDENCE_THRESHOLD}")
            # Pas d'échec, juste warning
        
        # Validation cohérence catégorie
        expected_category = get_intent_category(classification_result.intent_type)
        if classification_result.category != expected_category:
            logger.warning(f"Catégorie incohérente: {classification_result.category} != {expected_category}")
            # Correction automatique
            classification_result.category = expected_category
        
        # Validation alternatives
        for alt in classification_result.alternatives:
            if not (0.0 <= alt.confidence <= 1.0):
                logger.error(f"Alternative confidence invalide: {alt.confidence}")
                return False
        
        # Validation reasoning non vide
        if not classification_result.reasoning or len(classification_result.reasoning.strip()) < 5:
            logger.warning("Reasoning trop court ou vide")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur validation classification: {str(e)}")
        return False

def get_intent_category(intent_type: str) -> str:
    """Trouve la catégorie d'une intention"""
    try:
        intent_enum = HarenaIntentType(intent_type)
        for category, intents in INTENT_CATEGORIES.items():
            if intent_enum in intents:
                return category
        return "UNKNOWN_CATEGORY"
    except ValueError:
        return "UNKNOWN_CATEGORY"

def validate_user_message(message: str) -> Dict[str, Any]:
    """Validation message utilisateur"""
    
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Message vide
    if not message or not message.strip():
        validation_result["valid"] = False
        validation_result["errors"].append("Message vide")
        return validation_result
    
    # Longueur message
    if len(message) > 1000:
        validation_result["valid"] = False
        validation_result["errors"].append("Message trop long (>1000 caractères)")
    
    if len(message) < 2:
        validation_result["warnings"].append("Message très court")
    
    # Caractères suspects
    suspicious_chars = ['<script', '<?php', 'SELECT ', 'DROP TABLE']
    if any(suspicious in message.upper() for suspicious in suspicious_chars):
        validation_result["warnings"].append("Caractères suspects détectés")
    
    # Répétition excessive
    if len(set(message.replace(' ', ''))) < len(message) / 10:
        validation_result["warnings"].append("Répétition excessive de caractères")
    
    return validation_result

def sanitize_user_input(message: str) -> str:
    """Nettoyage sécurisé input utilisateur"""
    
    if not message:
        return ""
    
    # Suppression caractères de contrôle
    sanitized = ''.join(char for char in message if ord(char) >= 32 or char in '\n\t')
    
    # Limitation longueur
    sanitized = sanitized[:1000]
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    return sanitized

def validate_deepseek_response(response: Dict[str, Any]) -> bool:
    """Validation structure réponse DeepSeek"""
    
    try:
        # Structure de base
        if "choices" not in response:
            logger.error("Réponse DeepSeek manque 'choices'")
            return False
        
        choices = response["choices"]
        if not choices or len(choices) == 0:
            logger.error("Réponse DeepSeek choices vide")
            return False
        
        # Premier choice
        first_choice = choices[0]
        if "message" not in first_choice:
            logger.error("Choice manque 'message'")
            return False
        
        message = first_choice["message"]
        if "content" not in message:
            logger.error("Message manque 'content'")
            return False
        
        content = message["content"]
        if not isinstance(content, str) or len(content.strip()) == 0:
            logger.error("Content vide ou invalide")
            return False
        
        # Usage tokens si disponible
        if "usage" in response:
            usage = response["usage"]
            total_tokens = usage.get("total_tokens", 0)
            if total_tokens > settings.DEEPSEEK_MAX_TOKENS * 1.5:
                logger.warning(f"Usage tokens élevé: {total_tokens}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur validation réponse DeepSeek: {str(e)}")
        return False

def is_intent_supported(intent_type: str) -> bool:
    """Vérification si intention est supportée"""
    try:
        intent_enum = HarenaIntentType(intent_type)
        unsupported = INTENT_CATEGORIES.get("UNSUPPORTED", [])
        return intent_enum not in unsupported
    except ValueError:
        return False

def get_unsupported_intent_message(intent_type: str) -> str:
    """Message pour intentions non supportées"""
    
    messages = {
        "TRANSFER_REQUEST": "Les virements ne sont pas supportés. Utilisez votre app bancaire.",
        "PAYMENT_REQUEST": "Les paiements ne sont pas supportés. Utilisez votre app bancaire.",
        "CARD_BLOCK": "Le blocage de carte n'est pas supporté. Contactez votre banque.",
        "BUDGET_INQUIRY": "La gestion de budget n'est pas encore disponible.",
        "GOAL_TRACKING": "Le suivi d'objectifs n'est pas encore disponible.",
        "EXPORT_REQUEST": "L'export de données n'est pas encore disponible.",
        "OUT_OF_SCOPE": "Cette demande est hors du domaine financier."
    }
    
    return messages.get(intent_type, "Cette fonctionnalité n'est pas encore supportée.")

def calculate_confidence_adjustment(
    base_confidence: float, 
    context_factors: Dict[str, Any]
) -> float:
    """Ajustement confidence selon contexte"""
    
    adjusted_confidence = base_confidence
    
    # Facteur longueur message
    message_length = context_factors.get("message_length", 0)
    if message_length < 3:
        adjusted_confidence *= 0.9  # Pénalité message trop court
    elif message_length > 100:
        adjusted_confidence *= 1.05  # Bonus message détaillé
    
    # Facteur alternatives
    alternatives_count = context_factors.get("alternatives_count", 0)
    if alternatives_count > 0:
        adjusted_confidence *= 0.95  # Légère pénalité si alternatives
    
    # Maintenir dans [0, 1]
    return max(0.0, min(1.0, adjusted_confidence))