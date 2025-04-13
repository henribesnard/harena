"""
Utilitaires pour compter les tokens dans les textes.

Ce module fournit des fonctions pour estimer le nombre de tokens
dans les textes, utilisé pour le suivi de la consommation du modèle LLM.
"""

import re
from typing import Union, List, Dict, Any


def count_tokens(text: Union[str, List, Dict]) -> int:
    """
    Estime le nombre de tokens dans un texte ou une structure de données.
    
    Cette fonction utilise une heuristique simple pour estimer le nombre de tokens:
    - Environ 4 caractères par token en moyenne pour la plupart des langues occidentales
    - La tokenisation réelle dépend du tokenizer spécifique du modèle
    
    Args:
        text: Texte ou structure de données contenant du texte
        
    Returns:
        Nombre estimé de tokens
    """
    if text is None:
        return 0
    
    if isinstance(text, str):
        # Heuristique simple: ~4 caractères par token en moyenne
        # Cette estimation est approximative et peut varier selon la langue et le modèle
        return max(1, len(text) // 4)
    
    elif isinstance(text, list):
        # Récursivement compter les tokens dans une liste
        return sum(count_tokens(item) for item in text)
    
    elif isinstance(text, dict):
        # Récursivement compter les tokens dans un dictionnaire
        return sum(count_tokens(key) + count_tokens(value) for key, value in text.items())
    
    else:
        # Convertir en chaîne pour les autres types
        return count_tokens(str(text))


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Tronque un texte pour respecter une limite de tokens.
    
    Args:
        text: Texte à tronquer
        max_tokens: Nombre maximum de tokens
        
    Returns:
        Texte tronqué
    """
    if count_tokens(text) <= max_tokens:
        return text
    
    # Estimation grossière: tronquer à max_tokens * 4 caractères
    # puis ajuster si nécessaire
    estimated_chars = max_tokens * 4
    
    # Tronquer aux caractères estimés
    truncated = text[:estimated_chars]
    
    # Vérifier et ajuster si nécessaire
    while count_tokens(truncated) > max_tokens and len(truncated) > 0:
        truncated = truncated[:-100]  # Retirer 100 caractères à la fois
    
    return truncated


def truncate_conversation_history(
    messages: List[Dict[str, str]],
    max_tokens: int,
    keep_system_prompt: bool = True
) -> List[Dict[str, str]]:
    """
    Tronque l'historique de conversation pour respecter une limite de tokens.
    
    Args:
        messages: Liste de messages de la conversation
        max_tokens: Nombre maximum de tokens
        keep_system_prompt: Conserver le message système initial
        
    Returns:
        Liste de messages tronquée
    """
    if not messages:
        return []
    
    # Compter les tokens actuels
    total_tokens = count_tokens(messages)
    
    if total_tokens <= max_tokens:
        return messages
    
    # Conserver le message système si demandé
    system_message = None
    if keep_system_prompt and messages and messages[0]["role"] == "system":
        system_message = messages[0]
        messages = messages[1:]
    
    # Tronquer les messages en commençant par les plus anciens
    while messages and count_tokens(system_message) + count_tokens(messages) > max_tokens:
        # Retirer le message le plus ancien (en conservant les plus récents)
        messages.pop(0)
    
    # Réinsérer le message système si nécessaire
    if system_message:
        messages.insert(0, system_message)
    
    return messages