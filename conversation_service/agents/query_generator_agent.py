"""Query optimization utilities for search queries.

This module provides a minimal `QueryOptimizer` used in tests. It applies
simple heuristic tweaks to a base query depending on the intent.
"""
from typing import Any, Dict

from ..models.enums import IntentType


class QueryOptimizer:
    """Utility class to optimize search queries for common intents."""

    @staticmethod
    def optimize_query(base_query: Dict[str, Any], intent: IntentType) -> Dict[str, Any]:
        """Return an optimized copy of ``base_query``.

        The optimizer sets a default limit and ensures results are sorted by
        date in descending order. Additional intent-specific rules can be
        added in the future.
        """
        query = {"search_parameters": {}, "aggregations": {}}
        query.update(base_query)

        params = query.setdefault("search_parameters", {})
        params.setdefault("limit", 15)
        params.setdefault("sort", {"date": "desc"})

        return query
