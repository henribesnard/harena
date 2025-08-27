"""Normalisation de montants pour le conversation service."""
from __future__ import annotations

import re
from typing import Optional


def normalize_amount(amount_str: str) -> Optional[float]:
    """Convertit une chaîne représentant un montant en ``float``.

    La fonction accepte des formats variés (``1 234,56 €``, ``$1,234.56``, etc.) et
    gère automatiquement les séparateurs de milliers et décimaux.

    Args:
        amount_str: Chaîne contenant un montant.

    Returns:
        Montant sous forme ``float`` ou ``None`` si la conversion échoue.
    """
    if not isinstance(amount_str, str) or not amount_str.strip():
        return None

    # Suppression des symboles monétaires et des espaces non nécessaires
    cleaned = re.sub(r"[^0-9,.-]", "", amount_str.strip())

    # Gestion des formats européens (virgule comme séparateur décimal)
    if cleaned.count(",") == 1 and cleaned.count(".") == 0:
        cleaned = cleaned.replace(",", ".")
    else:
        # Suppression des virgules utilisées comme séparateur de milliers
        cleaned = cleaned.replace(",", "")

    try:
        return float(cleaned)
    except ValueError:
        return None
