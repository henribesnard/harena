"""Utility helpers for query generation agents used in tests."""

from __future__ import annotations

from typing import Any, Dict

from ..models.core_models import IntentType


class QueryOptimizer:
    """Apply small optimisations to search queries based on detected intent."""

    _MERCHANT_LIMIT = 15

    @staticmethod
    def optimize_query(base_query: Dict[str, Any], intent: IntentType) -> Dict[str, Any]:
        optimized = {
            "search_parameters": dict(base_query.get("search_parameters", {})),
            "aggregations": dict(base_query.get("aggregations", {})),
        }
        params = optimized["search_parameters"]
        if intent == IntentType.MERCHANT_ANALYSIS:
            params.setdefault("limit", QueryOptimizer._MERCHANT_LIMIT)
            params.setdefault("sort", [{"total_spent": {"order": "desc"}}])
        return optimized


__all__ = ["QueryOptimizer"]
