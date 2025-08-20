import logging
from typing import Any, Dict, Optional

import httpx

from conversation_service.utils.metrics import get_default_metrics_collector

logger = logging.getLogger(__name__)


class SearchClient:
    """HTTP client for the search_service API."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 10,
        transport: Optional[Any] = None,
    ) -> None:
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)
        self.metrics = get_default_metrics_collector()

    async def search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a search query against the search service."""
        timer_id = self.metrics.performance_monitor.start_timer("search_service")
        self.metrics.record_request("search_service", payload.get("user_id", 0))
        try:
            response = await self.client.post("/search", json=payload)
            response.raise_for_status()
            self.metrics.record_success("search_service")
            logger.info("Search request successful: %s", payload.get("query"))
            return response.json()
        except Exception as e:
            self.metrics.record_error("search_service", str(e))
            logger.exception("Search request failed")
            raise
        finally:
            duration_ms = self.metrics.performance_monitor.end_timer(timer_id)
            self.metrics.record_response_time("search_service", duration_ms)

    async def close(self) -> None:
        await self.client.aclose()
