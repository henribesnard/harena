"""Utility helpers for query generation agents used in tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    from .query_generator import QueryGeneratorAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    QueryGeneratorAgent = None  # type: ignore

from ..models.core_models import IntentType


class QueryOptimizer:
    """Apply small optimisations to search queries based on detected intent."""

    _MERCHANT_LIMIT = 15

    @staticmethod
    def optimize_query(base_query: Dict[str, Any], intent: IntentType) -> Dict[str, Any]:
        """Return an optimized copy of ``base_query``."""
        query = deepcopy(base_query)
        params = query.setdefault("search_parameters", {})
        if intent == IntentType.MERCHANT_ANALYSIS:
            params.setdefault("limit", QueryOptimizer._MERCHANT_LIMIT)
            params.setdefault("sort", [{"total_spent": {"order": "desc"}}])
        return query


__all__ = ["QueryGeneratorAgent", "QueryOptimizer"]
