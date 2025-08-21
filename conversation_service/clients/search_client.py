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

import asyncio
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
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._max_retries = max_retries

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            headers = {"Accept": "application/json"}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            self._session = aiohttp.ClientSession(headers=headers, timeout=self._timeout)
        return self._session

    async def search(self, user_id: int, payload: Dict[str, Any], endpoint: str = "/search") -> Dict[str, Any]:
        """Execute a search query for ``user_id`` and return the JSON response."""

        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        payload = dict(payload)
        payload["user_id"] = user_id
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(2 ** (attempt - 1))
        assert last_error is not None
        raise last_error

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

