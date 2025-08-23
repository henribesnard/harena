"""Prompt templates and normalization utilities for entity extraction."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

ENTITY_EXTRACTION_SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé dans l'extraction d'entités financières. "
    "À partir d'un message, identifie toutes les entités suivantes lorsqu'elles sont présentes: "
    "AMOUNT, TEMPORAL, MERCHANT, CATEGORY. "
    "Réponds uniquement avec un tableau JSON d'objets {\"entity_type\": ..., \"value\": ...}. "
    "N'ajoute pas de texte supplémentaire."
)

FEW_SHOT_EXAMPLES: List[List[Dict[str, str]]] = [
    [
        {"role": "user", "content": "J'ai dépensé 12,50€ chez Carrefur le 5 Janv 2024"},
        {
            "role": "assistant",
            "content": json.dumps(
                [
                    {
                        "entity_type": "AMOUNT",
                        "value": "12,50€",
                        "normalized": {"amount": "12.50", "currency": "EUR"},
                    },
                    {"entity_type": "MERCHANT", "value": "Carrefur"},
                    {
                        "entity_type": "TEMPORAL",
                        "value": "5 Janv 2024",
                        "normalized": {"date": "2024-01-05"},
                    },
                ],
                ensure_ascii=False,
            ),
        },
    ],
    [
        {"role": "user", "content": "Paiement de 30 USD a Walmrt avant-hier"},
        {
            "role": "assistant",
            "content": json.dumps(
                [
                    {
                        "entity_type": "AMOUNT",
                        "value": "30 USD",
                        "normalized": {"amount": "30", "currency": "USD"},
                    },
                    {"entity_type": "MERCHANT", "value": "Walmrt"},
                    {"entity_type": "TEMPORAL", "value": "avant-hier"},
                ],
                ensure_ascii=False,
            ),
        },
    ],
    [
        {
            "role": "user",
            "content": "Achat de 3000¥ chez Uniqlo le 12/07/23, catégorie vêtements",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                [
                    {
                        "entity_type": "AMOUNT",
                        "value": "3000¥",
                        "normalized": {"amount": "3000", "currency": "JPY"},
                    },
                    {"entity_type": "MERCHANT", "value": "Uniqlo"},
                    {
                        "entity_type": "TEMPORAL",
                        "value": "12/07/23",
                        "normalized": {"date": "2023-07-12"},
                    },
                    {
                        "entity_type": "CATEGORY",
                        "value": "vêtements",
                        "normalized": "vetements",
                    },
                ],
                ensure_ascii=False,
            ),
        },
    ],
]

_CURRENCY_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY"}

_CURRENCY_NAMES = {
    "euro": "EUR",
    "euros": "EUR",
    "dollar": "USD",
    "dollars": "USD",
    "livre": "GBP",
    "livres": "GBP",
    "yen": "JPY",
}

_CATEGORY_SYNONYMS = {
    "resto": "restaurant",
    "restaurants": "restaurant",
    "courses": "courses",
    "epicerie": "courses",
    "épicerie": "courses",
    "vetements": "habillement",
    "vêtements": "habillement",
}


def normalize_amount(text: str) -> Optional[Dict[str, str]]:
    """Return normalized amount and currency if possible."""
    if not text:
        return None
    pattern = re.compile(
        r"(?P<symbol>[$€£¥])?\s*(?P<value>[0-9]+(?:[.,][0-9]+)?)\s*(?P<code>[A-Za-z]{3})?"
    )
    match = pattern.search(text)
    if not match:
        return None
    value = match.group("value").replace(",", ".")
    try:
        amount = Decimal(value)
    except InvalidOperation:
        return None
    currency = (
        _CURRENCY_SYMBOLS.get(match.group("symbol") or "")
        or _CURRENCY_NAMES.get((match.group("code") or "").lower())
        or (match.group("code") or "").upper()
        or "EUR"
    )
    return {"amount": f"{amount.normalize()}", "currency": currency}


def _parse_date(text: str) -> Optional[datetime]:
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def normalize_date(text: str) -> Optional[str]:
    """Normalize a date string to ISO format (YYYY-MM-DD)."""
    if not text:
        return None
    dt = _parse_date(text.strip())
    if dt:
        return dt.strftime("%Y-%m-%d")
    try:
        from dateutil import parser

        dt = parser.parse(text, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _strip_accents(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", value) if unicodedata.category(c) != "Mn"
    )


def normalize_category(text: str) -> str:
    """Normalize category names by lower-casing and removing accents."""
    if not text:
        return ""
    key = _strip_accents(text).lower().strip()
    return _CATEGORY_SYNONYMS.get(key, key)


def normalize_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    """Return entity with an additional 'normalized' field when possible."""
    entity_type = entity.get("entity_type")
    value = entity.get("value", "")
    normalized: Any = None
    if entity_type == "AMOUNT":
        normalized = normalize_amount(value)
    elif entity_type in {"TEMPORAL", "DATE", "PERIOD"}:
        date = normalize_date(value)
        if date:
            normalized = {"date": date}
    elif entity_type == "CATEGORY":
        normalized = normalize_category(value)
    if normalized is not None:
        entity["normalized"] = normalized
    return entity


def normalize_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize all entities in a list."""
    return [normalize_entity(e) for e in entities]

