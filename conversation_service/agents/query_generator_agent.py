"""Utility helpers for query generation agents."""

from __future__ import annotations

from typing import Any, Dict

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
