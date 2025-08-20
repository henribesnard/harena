"""Prompts pour l'extraction d'entités.

Ce module propose un prompt par défaut et une gestion simplifiée des exemples
few‑shot pour extraire des entités d'un texte utilisateur. Les prompts peuvent
être chargés depuis un fichier externe ou récupérés depuis un cache.
"""

from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_PROMPT = "Identifie les entités présentes dans le message utilisateur."

_EXAMPLES: List[Dict[str, str]] = [
    {"input": "J'ai dépensé 50€ chez Amazon", "output": "AMOUNT:50, MERCHANT:Amazon"},
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
    """Récupérer les exemples actuels."""
    return list(_EXAMPLES)


def update_examples(examples: List[Dict[str, str]]) -> None:
    """Mettre à jour les exemples few‑shot."""
    _EXAMPLES.clear()
    _EXAMPLES.extend(examples)
