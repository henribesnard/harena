"""Prompts pour la génération de réponses dans le service de conversation.

Ce module fournit un prompt système, des exemples few‑shot et des
modèles de réponse par intention. Il offre des utilitaires pour charger
le prompt depuis un fichier ou un cache et pour gérer dynamiquement les
exemples utilisés."""

from pathlib import Path
from typing import Dict, List, Optional

RESPONSE_GENERATION_SYSTEM_PROMPT = (
    "Tu synthétises les résultats de recherche en réponses claires et utiles."
)

RESPONSE_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "input": "transactions Amazon du mois dernier",
        "output": "Vous avez 3 transactions chez Amazon en avril."
    }
]

INTENT_RESPONSE_TEMPLATES: Dict[str, str] = {
    "BALANCE_INQUIRY": "Votre solde est de {balance}€.",
}

FINANCIAL_FORMATTING_RULES: Dict[str, str] = {
    "currency": "EUR",
    "decimal_separator": ",",
}

_PROMPT_CACHE: Dict[str, str] = {}

def load_prompt(path: Optional[str] = None, *, cache: Optional[Dict[str, str]] = None, cache_key: str = "system") -> str:
    """Charger le prompt système depuis un fichier ou un cache."""
    cache = _PROMPT_CACHE if cache is None else cache
    if cache_key in cache:
        return cache[cache_key]
    if path:
        prompt = Path(path).read_text(encoding="utf-8")
        cache[cache_key] = prompt
        return prompt
    return RESPONSE_GENERATION_SYSTEM_PROMPT


def get_examples() -> List[Dict[str, str]]:
    """Retourner les exemples few-shot de génération de réponses."""
    return list(RESPONSE_FEW_SHOT_EXAMPLES)


def update_examples(examples: List[Dict[str, str]]) -> None:
    """Mettre à jour les exemples few-shot."""
    RESPONSE_FEW_SHOT_EXAMPLES.clear()
    RESPONSE_FEW_SHOT_EXAMPLES.extend(examples)

