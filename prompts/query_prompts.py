"""Prompts pour la génération de requêtes.

Utilitaires minimalistes permettant de charger un prompt de génération de
requêtes depuis un fichier ou un cache et de gérer les exemples few‑shot
associés.
"""

from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_PROMPT = "Génère une requête basée sur l'intention détectée."

_EXAMPLES: List[Dict[str, str]] = [
    {"input": "transactions supérieures à 100€", "output": "amount>100"},
    {
        "input": "dépenses chez Amazon en 2024",
        "output": "merchant:Amazon AND date:2024*",
    },
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


def get_prompt(
    path: Optional[str] = None,
    *,
    cache: Optional[Dict[str, str]] = None,
    cache_key: str = "default",
) -> str:
    """Retourner le prompt à utiliser par l'agent."""
    return load_prompt(path=path, cache=cache, cache_key=cache_key)


def get_examples() -> List[Dict[str, str]]:
    """Obtenir les exemples few‑shot utilisés pour la génération de requêtes."""
    return list(_EXAMPLES)


def update_examples(examples: List[Dict[str, str]]) -> None:
    """Mettre à jour les exemples few‑shot."""
    _EXAMPLES.clear()
    _EXAMPLES.extend(examples)
