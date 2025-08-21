"""Query optimization utilities for conversation service.

This module exposes :class:`QueryOptimizer`, a lightweight helper that applies
post-processing rules to search queries before they are sent to the search
service.  The optimizer understands domain specific intents and adjusts the
query accordingly so that downstream services receive well scoped requests.

Supported optimization rules
----------------------------
* :class:`~conversation_service.models.enums.IntentType.MERCHANT_ANALYSIS` –
  limits the number of returned merchants to 15 and applies a deterministic
  sort on the aggregated spend so that the highest spending merchants appear
  first.

Additional rules can be added in the future as new intents require special
handling.
"""

from __future__ import annotations

from typing import Any, Dict

from ..models.core_models import IntentType


class QueryOptimizer:
    """Apply intent specific tweaks to a search query."""

    _MERCHANT_LIMIT = 15

    @staticmethod
    def optimize_query(query: Dict[str, Any], intent: IntentType) -> Dict[str, Any]:
        """Return an optimized version of ``query`` based on ``intent``.

        Parameters
        ----------
        query:
            Base query structure targeting the search service. It must contain a
            ``"search_parameters"`` mapping that will be updated in place.
        intent:
            The detected user intent guiding which optimization rules are
            applied.

        For :class:`IntentType.MERCHANT_ANALYSIS` the method enforces a limit of
        15 merchants and adds a default sort clause ordering results by
        descending spend.
        """

        # Ensure the query has the expected container for search parameters.
        search_params = query.setdefault("search_parameters", {})

        if intent == IntentType.MERCHANT_ANALYSIS:
            # Restrict the number of results returned to the top merchants and
            # apply a deterministic sort so that results are ordered by the
            # total amount spent in descending order.
            search_params["limit"] = QueryOptimizer._MERCHANT_LIMIT
            search_params.setdefault("sort", [{"total_spent": {"order": "desc"}}])

        return query
