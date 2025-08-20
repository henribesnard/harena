"""Prompts pour la génération de réponses.

Ce module fournit un prompt de réponse par défaut ainsi que des fonctions pour
charger un prompt personnalisé depuis un fichier ou un cache. Les exemples
few‑shot peuvent être consultés ou modifiés afin d'ajuster le style de réponse.
"""

from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_PROMPT = "Formule une réponse utilisateur à partir des données fournies."

_EXAMPLES: List[Dict[str, str]] = [
    {
        "input": "transactions Amazon du mois dernier",
        "output": "Vous avez 3 transactions chez Amazon en avril."},
]

_PROMPT_CACHE: Dict[str, str] = {}

def load_prompt(path: Optional[str] = None, *, cache: Optional[Dict[str, str]] = None, cache_key: str = "default") -> str:
    """Charger le prompt depuis un fichier ou un cache."""
    cache = _PROMPT_CACHE if cache is None else cache
    if cache_key in cache:
        return cache[cache_key]
    if path:
        prompt = Path(path).read_text(encoding="utf-8")
        cache[cache_key] = prompt
        return prompt
    return DEFAULT_PROMPT


def get_examples() -> List[Dict[str, str]]:
    """Retourner les exemples actuels de génération de réponses."""
    return list(_EXAMPLES)


def update_examples(examples: List[Dict[str, str]]) -> None:
    """Mettre à jour la liste des exemples few‑shot."""
    _EXAMPLES.clear()
    _EXAMPLES.extend(examples)
