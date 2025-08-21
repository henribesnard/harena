"""Generate personalised responses from search results using OpenAI.

This module implements a lightweight :class:`ResponseGeneratorAgent` which
builds a short prompt from search results and conversational context before
delegating the response creation to an OpenAI client. The produced answer is
cached in-memory for a short period to avoid repeated API calls when the same
inputs are supplied again.

The agent is purposely minimal – it only expects the OpenAI client to expose a
``chat_completion`` coroutine compatible with the :class:`OpenAIClient` wrapper
used throughout the project. During tests a dummy object providing the same
method can be supplied.
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
    openai_client:
        Client exposing an ``async chat_completion`` method. In production this
        is :class:`openai_client.OpenAIClient` but tests may provide a stub.
    model:
        Model name used for the OpenAI call. Defaults to ``"gpt-4o-mini"``.
    ttl:
        Cache time-to-live in seconds. Defaults to ``60``.
    """

    def __init__(self, openai_client: Any, *, model: str = "gpt-4o-mini", ttl: int = 60) -> None:
        self._client = openai_client
        self._model = model
        self.ttl = ttl
        self._cache: Dict[str, Tuple[float, str]] = {}

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(search_results: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Compose the prompt from search results and conversation context."""

        result_count = len(search_results)
        top_result = search_results[0] if search_results else {}
        snippet = (
            top_result.get("summary")
            or top_result.get("snippet")
            or top_result.get("title")
            or json.dumps(top_result, ensure_ascii=False)
        )

        user_profile = context.get("user_profile", {})
        user_name = user_profile.get("name", "client")
        preferences = user_profile.get("preferences", {})
        intent = context.get("intent")
        entities = context.get("entities", [])

        parts = [
            f"Utilisateur: {user_name}.",
            f"Résultats: {result_count}.",
            f"Principal: {snippet}.",
        ]
        if intent:
            parts.append(f"Intention: {intent}.")
        if entities:
            parts.append("Entités: " + json.dumps(entities, ensure_ascii=False) + ".")
        if preferences:
            parts.append(
                "Préférences: " + json.dumps(preferences, ensure_ascii=False) + "."
            )

        return " ".join(parts) + " Fournis une réponse appropriée."

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate(
        self, user_id: str, search_results: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> str:
        """Return a response for ``search_results`` within ``context``.

        The method analyses the top search result alongside context information
        (intent, entities and user preferences) and delegates response creation
        to ``self._client``. Results are cached for ``ttl`` seconds.
        """

        key = self._make_cache_key(user_id, search_results, context)
        now = time.time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < self.ttl:
            return cached[1]

        prompt = self._build_prompt(search_results, context)

        # --- LLM generation -------------------------------------------------
        try:
            response = await self._client.chat_completion(
                model=self._model, messages=[{"role": "user", "content": prompt}]
            )
            # ``openai`` returns objects with attribute access while the tests
            # use plain dictionaries. Support both forms for robustness.
            try:
                content = response["choices"][0]["message"]["content"]
            except Exception:  # pragma: no cover - library object
                content = response.choices[0].message["content"]
            text = content.strip()
        except Exception:
            top_result = search_results[0] if search_results else {}
            suggestion = (
                top_result.get("url")
                or top_result.get("title")
                or "réessaie avec une autre requête"
            )
            return (
                "Je rencontre un problème pour générer la réponse. "
                f"Suggestion: {suggestion}"
            )

        # --- Cache storage --------------------------------------------------
        self._cache[key] = (now, text)
        return text


async def stream_response(message: str) -> AsyncGenerator[str, None]:
    """Fallback async generator returning the message directly."""

    yield f"Response: {message}"


__all__ = ["ResponseGeneratorAgent", "stream_response"]

