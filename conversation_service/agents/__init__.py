"""
Agents module pour le Conversation Service

Ce module contient tous les agents intelligents du service :
- Agent de classification d'intentions (MVP)
- Futures extensions : agents d'extraction d'entités, génération de requêtes, etc.
"""

from .intent_classifier import (
    intent_classifier,
    IntentClassifierAgent
)

__version__ = "1.0.0"
__all__ = [
    "intent_classifier",      # Instance globale configurée
    "IntentClassifierAgent"   # Classe pour créer d'autres instances
]

# Instance globale configurée
classifier = intent_classifier

# Helpers pour les agents
async def classify_message(message: str):
    """Helper simple pour classifier un message"""
    return await intent_classifier.classify_intent(message)

def get_all_agents_metrics():
    """Récupère les métriques de tous les agents"""
    return {
        "intent_classifier": intent_classifier.get_metrics()
    }

def reset_all_agents_metrics():
    """Remet à zéro les métriques de tous les agents"""
    intent_classifier.reset_metrics()

def get_agents_status():
    """Statut de tous les agents"""
    classifier_metrics = intent_classifier.get_metrics()
    
    return {
        "intent_classifier": {
            "status": "operational",
            "total_classifications": classifier_metrics["total_classifications"],
            "success_rate": classifier_metrics["success_rate"],
            "last_activity": classifier_metrics.get("last_classification_time")
        }
    }

# Constantes utiles
SUPPORTED_AGENTS = ["intent_classifier"]
DEFAULT_CONFIDENCE_THRESHOLD = 0.7