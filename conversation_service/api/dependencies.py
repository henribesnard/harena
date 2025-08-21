from typing import Optional

from fastapi import WebSocket, HTTPException, status

from ..clients import OpenAIClient, SearchClient, CacheClient
from openai_config import openai_config
from config_service.config import settings


_openai_client: Optional[OpenAIClient] = None
_search_client: Optional[SearchClient] = None
_cache_client: Optional[CacheClient] = None
from clients.cache_client import CacheClient
from clients.openai_client import OpenAIClient
from config.autogen_config import AutogenConfig, autogen_settings

async def get_session_id(websocket: WebSocket) -> str:
    """Authenticate websocket connections using a session token.

    The token is expected in the query string as ``?session=<token>``. If the
    token is missing the connection is closed with an appropriate code and a
    403 error is raised so the dependency chain is halted.
    """
    session_id = websocket.query_params.get("session")
    if not session_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session non authentifiée"
        )
    return session_id


async def get_openai_client() -> OpenAIClient:
    """Return a shared :class:`OpenAIClient` instance."""

    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient(
            api_key=openai_config.api_key, base_url=openai_config.base_url
        )
    return _openai_client


async def get_search_client() -> SearchClient:
    """Return a shared :class:`SearchClient` instance."""

    global _search_client
    if _search_client is None:
        base_url = getattr(settings, "SEARCHBOX_URL", None) or getattr(
            settings, "ELASTICSEARCH_URL", "http://localhost"
        )
        _search_client = SearchClient(base_url)
    return _search_client


async def get_cache_client() -> CacheClient:
    """Return a shared :class:`CacheClient` instance."""

    global _cache_client
    if _cache_client is None:
        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379")
        _cache_client = CacheClient(redis_url)
    return _cache_client
_cache_client = CacheClient()
_openai_client = OpenAIClient(cache=_cache_client)


def get_cache_client() -> CacheClient:
    """Return the shared cache client instance."""
    return _cache_client


def get_openai_client() -> OpenAIClient:
    """Return the shared OpenAI client configured with caching."""
    return _openai_client


def get_autogen_config() -> AutogenConfig:
    """Provide AutoGen configuration settings."""
    return autogen_settings
