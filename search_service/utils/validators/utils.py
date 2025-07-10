"""
Fonctions utilitaires de validation de haut niveau.

Ce module fournit des fonctions utilitaires simples pour
les validations courantes sans avoir à instancier des classes.
"""

import re
from typing import Dict, Any, Union, Optional
from datetime import datetime
from decimal import Decimal

from .base import (
    BaseValidator, ValidationError, ParameterValidationError,
    DEFAULT_LIMITS, ES_SPECIAL_CHARS
)
from .filters import FilterValidator

def validate_search_request(
    query: str,
    user_id: Union[int, str],
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 20,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Valide une requête de recherche complète.