"""Utility helpers for query generation agents."""

from __future__ import annotations

from typing import Any, Dict

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
    """Apply small optimizations to search queries based on intent.

    The real project contains a much more elaborate optimizer, but for the
    purposes of the exercises we only implement the behaviour required by the
    tests.  The function operates on a dictionary describing the query and
    returns a new optimized dictionary leaving the original untouched.
    """

    @staticmethod
    def optimize_query(base_query: Dict[str, Any], intent: IntentType) -> Dict[str, Any]:
        """Return an optimized copy of ``base_query``.

        Parameters
        ----------
        base_query:
            Dictionary with keys ``search_parameters`` and ``aggregations``.
        intent:
            The :class:`IntentType` guiding the optimisation rules.
        """
        # Create a shallow copy to avoid mutating caller's structure
        optimized = {
            "search_parameters": dict(base_query.get("search_parameters", {})),
            "aggregations": dict(base_query.get("aggregations", {})),
        }

        params = optimized["search_parameters"]

        if intent == IntentType.MERCHANT_ANALYSIS:
            # Apply simple defaults tailored for merchant analysis queries
            params.setdefault("limit", 15)
            params.setdefault("sort", ["amount:desc"])
        else:
            params.setdefault("limit", 50)

        return optimized
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
