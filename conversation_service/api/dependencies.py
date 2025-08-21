"""Dependency helpers for the conversation service API.

This module lazily instantiates shared client instances used across the
application.  It avoids duplicate imports and definitions while providing a
single entry-point for each client.
"""

from typing import Optional

from fastapi import HTTPException, WebSocket, status

from ..clients import CacheClient, OpenAIClient, SearchClient
from config.autogen_config import AutogenConfig, autogen_settings
from config_service.config import settings
from openai_config import openai_config


_openai_client: Optional[OpenAIClient] = None
_search_client: Optional[SearchClient] = None
_cache_client: Optional[CacheClient] = None


async def get_session_id(websocket: WebSocket) -> str:
    """Authenticate websocket connections using a session token."""

    session_id = websocket.query_params.get("session")
    if not session_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session non authentifiée",
        )
    return session_id


def get_openai_client() -> OpenAIClient:
    """Return a shared :class:`OpenAIClient` instance."""

    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient(
            api_key=openai_config.api_key,
            base_url=openai_config.base_url,
        )
    return _openai_client


def get_search_client() -> SearchClient:
    """Return a shared :class:`SearchClient` instance."""

    global _search_client
    if _search_client is None:
        base_url = getattr(settings, "SEARCHBOX_URL", None) or getattr(
            settings, "ELASTICSEARCH_URL", "http://localhost"
        )
        _search_client = SearchClient(base_url)
    return _search_client


def get_cache_client() -> CacheClient:
    """Return a shared :class:`CacheClient` instance."""

    global _cache_client
    if _cache_client is None:
        _cache_client = CacheClient(
            settings.REDIS_URL,
            prefix=settings.REDIS_CACHE_PREFIX,
        )
    return _cache_client


def get_autogen_config() -> AutogenConfig:
    """Provide AutoGen configuration settings."""

    return autogen_settings

