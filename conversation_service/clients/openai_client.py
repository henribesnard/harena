from __future__ import annotations

"""Simple OpenAI async client wrapper.

This module provides :class:`OpenAIClient`, a thin wrapper around
:class:`openai.AsyncOpenAI` that adds two features that are useful for the
project:

* **Retry logic** – failed requests are retried with an exponential
  backoff.  Transient network errors are common when talking to the
  OpenAI API and retrying greatly improves reliability.
* **Cost tracking** – the client keeps track of the number of tokens used
  and an estimated cost in USD.  The pricing table can easily be
  customised per model.

The implementation is intentionally lightweight; it only exposes the
``chat_completion`` method which is the only one currently used by the
project.  It can be extended later if more endpoints are required.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger("openai_client")

# Default pricing table (USD per 1K tokens).  The values do not need to be
# perfectly accurate for the tests but give a reasonable estimation.  New
# models can be added by consumers of the client by passing a custom
# ``model_pricing`` mapping to ``OpenAIClient``.
DEFAULT_MODEL_PRICING: Dict[str, float] = {
    "gpt-4o-mini": 0.00015,
}


class OpenAIClient:
    """Wrapper around :class:`AsyncOpenAI` with retries and cost tracking."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        *,
        model_pricing: Optional[Dict[str, float]] = None,
        max_retries: int = 3,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._max_retries = max_retries
        self._model_pricing = model_pricing or DEFAULT_MODEL_PRICING
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0

    async def chat_completion(
        self, *, model: str, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Any:
        """Call the OpenAI chat completion endpoint.

        Retries the request on failure and updates cost metrics using the
        pricing table.  The raw response from ``openai`` is returned so the
        caller can access all available data.
        """

        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )

                usage = getattr(response, "usage", None)
                tokens = usage.total_tokens if usage else 0
                self.total_tokens += tokens

                price = self._model_pricing.get(model, 0.0)
                cost = tokens / 1000 * price
                self.total_cost_usd += cost

                logger.debug(
                    "OpenAI call succeeded",
                    extra={"model": model, "tokens": tokens, "cost": cost},
                )
                return response
            except Exception as exc:  # pragma: no cover - network errors
                last_error = exc
                logger.warning(
                    "OpenAI call failed (attempt %s/%s): %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(2 ** (attempt - 1))

        # If we get here all retries failed, re-raise the last exception
        assert last_error is not None
        raise last_error

    def get_total_cost(self) -> float:
        """Return the accumulated estimated cost in USD."""

        return self.total_cost_usd

    async def close(self) -> None:
        """Close the underlying HTTP session."""

        await self._client.close()

