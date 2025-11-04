"""
Utilitaires partagés pour le service utilisateur.

Ce module contient des fonctions utilitaires réutilisables dans tout le service.
"""
from typing import Dict, Any
from copy import deepcopy


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Effectue un merge profond de deux dictionnaires.
    Les valeurs de 'updates' prennent la priorité sur celles de 'base'.

    Args:
        base: Dictionnaire de base
        updates: Dictionnaire des mises à jour à appliquer

    Returns:
        Dict: Nouveau dictionnaire résultant du merge profond

    Example:
        >>> base = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> updates = {"a": {"b": 10}, "e": 5}
        >>> deep_merge(base, updates)
        {"a": {"b": 10, "c": 2}, "d": 3, "e": 5}
    """
    result = deepcopy(base)

    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Merge récursif pour les dictionnaires imbriqués
            result[key] = deep_merge(result[key], value)
        else:
            # Remplacement direct pour les autres types
            result[key] = deepcopy(value)

    return result
