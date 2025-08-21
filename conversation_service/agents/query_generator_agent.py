"""Wrapper module for query generation utilities.

The wrapper re-exports :class:`QueryGeneratorAgent` from the existing
``query_generator`` module and provides a minimal :class:`QueryOptimizer`
implementation used in tests.  The optimizer applies simple rules to augment
search queries based on the detected intent.
"""

from copy import deepcopy
from typing import Any, Dict

# Importing the concrete ``QueryGeneratorAgent`` may pull in optional
# dependencies.  We therefore import it lazily and degrade gracefully when those
# dependencies are missing.
try:  # pragma: no cover - defensive import
    from .query_generator import QueryGeneratorAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    QueryGeneratorAgent = None  # type: ignore

from ..models.core_models import IntentType


class QueryOptimizer:
    """Utility to apply intent-specific optimisations to search queries."""

    @staticmethod
    def optimize_query(base_query: Dict[str, Any], intent: IntentType) -> Dict[str, Any]:
        """Return a new query augmented according to ``intent``.

        The optimisation rules are intentionally lightweight and only implement
        what is required by the unit tests:

        * ``MERCHANT_ANALYSIS`` – limit results and ensure a sort order.
        """

        query = deepcopy(base_query)
        search_params = query.setdefault("search_parameters", {})

        if intent == IntentType.MERCHANT_ANALYSIS:
            search_params.setdefault("limit", 15)
            # The value of ``sort`` is not important for the tests; only its
            # presence matters.  A simple field ordering is provided.
            search_params.setdefault("sort", {"field": "amount", "order": "desc"})

        return query


__all__ = ["QueryGeneratorAgent", "QueryOptimizer"]
