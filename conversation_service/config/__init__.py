# conversation_service/config/__init__.py
"""
Configuration module pour le Conversation Service

Ce module contient toute la configuration centralisée du service :
- Settings avec validation
- Variables d'environnement
- Configuration DeepSeek
- Paramètres de performance
"""

from .settings import settings, ConversationSettings

__version__ = "1.0.0"
__all__ = [
    "settings", 
    "ConversationSettings"
]

# Validation automatique de la configuration au niveau du module
def validate_config():
    """Valide la configuration globale"""
    validation_result = settings.validate_configuration()
    if not validation_result["valid"]:
        raise RuntimeError(f"Configuration invalide: {validation_result['errors']}")
    return validation_result

# Export des constantes importantes
MIN_CONFIDENCE_THRESHOLD = settings.MIN_CONFIDENCE_THRESHOLD
API_VERSION = settings.API_VERSION
DEEPSEEK_MODEL = settings.DEEPSEEK_CHAT_MODEL