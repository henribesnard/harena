"""Package de prompts génériques.

Ce package fournit des modules simples pour gérer différents types de prompts
(intentions, entités, requêtes et réponses). Chaque module expose des
fonctionnalités pour charger dynamiquement un prompt et manipuler des exemples
few‑shot.
"""
from . import intent_prompts, entity_prompts, query_prompts, response_prompts

__all__ = [
    "intent_prompts",
    "entity_prompts",
    "query_prompts",
    "response_prompts",
]
