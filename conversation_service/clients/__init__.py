"""
Clients module pour le Conversation Service

Ce module contient les clients pour les services externes :
- Client DeepSeek optimisé avec cache et retry
- Futures extensions pour d'autres LLMs ou services
"""

from .deepseek_client import (
    deepseek_client,
    DeepSeekClient,
    DeepSeekResponse
)

__version__ = "1.0.0"
__all__ = [
    "deepseek_client",    # Instance globale prête à l'emploi
    "DeepSeekClient",     # Classe pour créer d'autres instances
    "DeepSeekResponse"    # Modèle de réponse
]

# Instance globale configurée
client = deepseek_client

# Helpers pour le client
async def health_check_all_clients():
    """Vérifie la santé de tous les clients"""
    results = {}
    
    try:
        deepseek_health = await deepseek_client.health_check()
        results["deepseek"] = deepseek_health
    except Exception as e:
        results["deepseek"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return results

def get_all_clients_metrics():
    """Récupère les métriques de tous les clients"""
    return {
        "deepseek": deepseek_client.get_metrics()
    }

def clear_all_caches():
    """Vide le cache de tous les clients"""
    deepseek_client.clear_cache()
    
def get_cache_info():
    """Informations sur les caches de tous les clients"""
    return {
        "deepseek": deepseek_client.get_cache_info()
    }