"""Normalisation de dates pour le conversation service."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from dateutil import parser


def normalize_date(date_str: str) -> Optional[str]:
    """Normalise une chaîne de date vers le format ISO ``YYYY-MM-DD``.

    Args:
        date_str: Chaîne représentant une date potentiellement partielle ou au format local.

    Returns:
        Date normalisée au format ISO ou ``None`` si la date est invalide.
    """
    if not isinstance(date_str, str) or not date_str.strip():
        return None

    # Détection rapide du format ISO pour éviter les inversions jour/mois
    iso_pattern = r"^\d{4}-\d{2}-\d{2}$"
    try:
        if re.match(iso_pattern, date_str.strip()):
            return datetime.fromisoformat(date_str.strip()).date().isoformat()

        dt: datetime = parser.parse(date_str, dayfirst=True)
        return dt.date().isoformat()
    except (parser.ParserError, ValueError, TypeError):
        return None
