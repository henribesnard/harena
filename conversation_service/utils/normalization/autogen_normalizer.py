"""Normalisation globale des transactions pour le conversation service."""
from __future__ import annotations

from typing import Any, Dict

from .amount_normalizer import normalize_amount
from .date_normalizer import normalize_date
from .merchant_normalizer import normalize_merchant


def normalize_transaction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Applique les normaliseurs aux champs connus d'une transaction.

    Args:
        data: Dictionnaire contenant potentiellement les clés ``date``, ``amount`` et
            ``merchant``.

    Returns:
        ``dict`` enrichi avec les valeurs normalisées.
    """
    normalized = dict(data)
    if "date" in data:
        normalized["date"] = normalize_date(data["date"])
    if "amount" in data:
        normalized["amount"] = normalize_amount(str(data["amount"]))
    if "merchant" in data:
        normalized["merchant"] = normalize_merchant(data["merchant"])
    return normalized
