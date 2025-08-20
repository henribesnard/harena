import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from conversation_service.utils.metrics import get_default_metrics_collector

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client OpenAI gérant streaming, retries et timeouts."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: int = 30,
        max_retries: int = 3,
        async_client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self.client = async_client or AsyncOpenAI(api_key=api_key or "test")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.metrics = get_default_metrics_collector()

    async def _create_completion(
        self, messages: List[Dict[str, str]], *, stream: bool, **kwargs: Any
    ) -> Any:
        """Internal wrapper with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=stream,
                    timeout=self.timeout,
                    **kwargs,
                )
            except Exception as e:  # pragma: no cover - network errors
                logger.warning(
                    "OpenAI request failed (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def stream_chat(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream completions as text chunks."""
        timer_id = self.metrics.performance_monitor.start_timer("openai_chat")
        self.metrics.record_request("openai_chat", 0)
        try:
            response = await self._create_completion(messages, stream=True, **kwargs)
            async for chunk in response:  # type: AsyncIterator[Any]
                text = chunk.choices[0].delta.content or ""
                if text:
                    logger.debug("OpenAI chunk received: %s", text)
                    yield text
            self.metrics.record_success("openai_chat")
        except Exception as e:
            self.metrics.record_error("openai_chat", str(e))
            logger.exception("OpenAI streaming failed")
            raise
        finally:
            duration_ms = self.metrics.performance_monitor.end_timer(timer_id)
            self.metrics.record_response_time("openai_chat", duration_ms)

    async def close(self) -> None:
        """Close underlying HTTP session."""
        await self.client.close()
