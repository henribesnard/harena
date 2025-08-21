from __future__ import annotations

"""Asynchronous client for the search micro-service.

The real project communicates with an external search service over HTTP.
This module provides a small :class:`SearchServiceClient` wrapper around
:mod:`aiohttp` that exposes a single :py:meth:`search` method.  The client
handles authentication headers, timeouts and basic error handling so that
callers can focus on crafting the request payload.
"""

from typing import Any, Dict, Optional

import aiohttp


class SearchServiceError(RuntimeError):
    """Raised when the search service request fails."""


class SearchServiceClient:
    """HTTP client used to communicate with the search service."""

    def __init__(
        self,
        base_url: str,
        *,
        token: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            headers = {"Accept": "application/json"}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            self._session = aiohttp.ClientSession(headers=headers, timeout=self._timeout)
        return self._session

    async def search(self, payload: Dict[str, Any], endpoint: str = "/search") -> Dict[str, Any]:
        """Send ``payload`` to the search service and return the JSON body."""

        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        try:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientError as exc:  # pragma: no cover - network error
            raise SearchServiceError(f"Search service request failed: {exc}") from exc

    async def close(self) -> None:
        """Close the underlying HTTP session."""

        if self._session is not None:
            await self._session.close()
            self._session = None


# Backwards compatibility alias
SearchClient = SearchServiceClient
