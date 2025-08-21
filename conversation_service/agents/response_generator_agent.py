"""Lightweight response generator agent with in-memory caching.

This module provides a minimal :class:`ResponseGeneratorAgent` used in tests.
It analyses provided search results and context, generates a text response via a
configured LLM client (``self.agent``) and caches results for 60 seconds to
avoid repeated work.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Tuple


class ResponseGeneratorAgent:
    """Generate natural language responses from search results.

    Parameters
    ----------
    agent:
        LLM client exposing an ``async generate(prompt: str) -> str`` method.
    ttl:
        Cache time-to-live in seconds. Defaults to ``60``.
    """

    def __init__(self, agent: Any, ttl: int = 60) -> None:
        self.agent = agent
        self.ttl = ttl
        self._cache: Dict[str, Tuple[float, str]] = {}

    @staticmethod
    def _make_cache_key(
        user_id: str, search_results: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> str:
        """Create a stable cache key for the inputs."""

        payload = json.dumps(
            {"user": user_id, "results": search_results, "context": context},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def generate(
        self, user_id: str, search_results: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> str:
        """Return a response for ``search_results`` within ``context``.

        The method analyses the top search result alongside basic context
        information and delegates response creation to ``self.agent``. Results
        are cached for ``ttl`` seconds.
        """

        key = self._make_cache_key(user_id, search_results, context)
        now = time.time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < self.ttl:
            return cached[1]

        # --- Analyse search results and context to build a prompt ------------
        result_count = len(search_results)
        top_result = search_results[0] if search_results else {}
        snippet = (
            top_result.get("summary")
            or top_result.get("snippet")
            or top_result.get("title")
            or json.dumps(top_result, ensure_ascii=False)
        )
        user_name = context.get("user_profile", {}).get("name", "client")
        prompt = (
            f"Utilisateur: {user_name}. Résultats: {result_count}. "
            f"Principal: {snippet}. Fournis une réponse appropriée."
        )

        # --- LLM generation -------------------------------------------------
        response = await self.agent.generate(prompt)

        # --- Cache storage --------------------------------------------------
        self._cache[key] = (now, response)
        return response


async def stream_response(message: str) -> AsyncGenerator[str, None]:
    """Fallback async generator returning the message directly."""

    yield f"Response: {message}"


__all__ = ["ResponseGeneratorAgent", "stream_response"]

