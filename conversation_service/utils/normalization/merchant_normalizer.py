"""Normalisation des noms de commerçant pour le conversation service."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional


def normalize_merchant(name: str) -> Optional[str]:
    """Nettoie et met en forme le nom d'un commerçant.

    La normalisation supprime la ponctuation et met la chaîne en majuscules afin de
    faciliter les comparaisons.

    Args:
        name: Nom du commerçant tel qu'extrait d'une transaction.

    Returns:
        Nom normalisé en majuscules ou ``None`` si la chaîne est vide.
    """
    if not isinstance(name, str) or not name.strip():
        return None

    cleaned = re.sub(r"[^\w\s]", " ", name.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Suppression des accents pour harmoniser les comparaisons
    normalized = unicodedata.normalize("NFD", cleaned)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")

    return normalized.upper()
