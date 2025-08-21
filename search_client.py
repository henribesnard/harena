"""Asynchronous client for the internal search service.

The real application communicates with a separate micro-service that
provides search capabilities.  The tests in this kata only need a very
small portion of that behaviour: performing authenticated HTTP requests
and returning the JSON payload.

This module exposes :class:`SearchClient` which wraps :mod:`aiohttp` and
handles authentication headers, session management and basic error
handling.
"""

from __future__ import annotations

import aiohttp
from typing import Any, Dict, Optional


class SearchClient:
    """HTTP client used to talk to ``search_service``."""

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
        """Execute a search query and return the JSON response."""

        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

