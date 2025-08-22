"""Minimal client for interacting with the search service."""

from __future__ import annotations

from typing import Any, Dict

import httpx

from conversation_service.settings import settings


class SearchClient:
    """Asynchronous HTTP client for the search service."""

    def __init__(self) -> None:
        self._url = settings.search_service_url
        self._client = httpx.AsyncClient()

    async def search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a search request against the search service."""

        response = await self._client.post(self._url, json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()
