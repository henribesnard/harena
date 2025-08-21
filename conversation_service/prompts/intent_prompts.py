"""Prompts pour la détection d'intention dans le service de conversation.

Ce module gère un prompt par défaut et des exemples few‑shot pour la
classification d'intentions. Il offre des utilitaires pour charger un prompt
depuis un fichier externe ou un cache en mémoire, ainsi que pour consulter ou
mettre à jour les exemples utilisés.
"""

from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_PROMPT = "Analyse l'intention du message utilisateur."

_EXAMPLES: List[Dict[str, str]] = [
    {"input": "Bonjour", "output": "GREETING"},
]

_PROMPT_CACHE: Dict[str, str] = {}

def load_prompt(path: Optional[str] = None, *, cache: Optional[Dict[str, str]] = None, cache_key: str = "default") -> str:
    """Charger le prompt depuis un fichier ou un cache.

    Args:
        path: chemin optionnel vers un fichier contenant le prompt.
        cache: dictionnaire utilisé comme cache en mémoire.
        cache_key: clé sous laquelle stocker/récupérer le prompt.
    Returns:
        Le texte du prompt.
    """
    cache = _PROMPT_CACHE if cache is None else cache
    if cache_key in cache:
        return cache[cache_key]
    if path:
        prompt = Path(path).read_text(encoding="utf-8")
        cache[cache_key] = prompt
        return prompt
    return DEFAULT_PROMPT


def get_examples() -> List[Dict[str, str]]:
    """Récupérer la liste actuelle des exemples few‑shot."""
    return list(_EXAMPLES)


def update_examples(examples: List[Dict[str, str]]) -> None:
    """Remplacer les exemples few‑shot par une nouvelle liste."""
    _EXAMPLES.clear()
    _EXAMPLES.extend(examples)
