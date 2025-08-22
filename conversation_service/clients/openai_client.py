from __future__ import annotations

"""Asynchronous OpenAI client with retries, rate limiting and metrics."""

from collections import OrderedDict, deque
import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import openai
from prometheus_client import Counter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from monitoring.performance import record_openai_cost

logger = logging.getLogger(__name__)

# Prometheus metrics
TOKENS_IN = Counter("openai_prompt_tokens_total", "Total prompt tokens sent to OpenAI")
TOKENS_OUT = Counter("openai_completion_tokens_total", "Total completion tokens received from OpenAI")
TOTAL_COST = Counter("openai_cost_usd_total", "Total cost of OpenAI requests in USD")
TOTAL_CALLS = Counter("openai_api_calls_total", "Total OpenAI API calls")


class LRUCache:
    """Simple in-memory LRU cache."""

    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str) -> Any:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)


class SlidingWindowRateLimiter:
    """Naive sliding window rate limiter."""

    def __init__(self, max_calls: int, window_seconds: int) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls: "deque[float]" = deque()

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > self.window_seconds:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return
            sleep_for = self.window_seconds - (now - self.calls[0])
            await asyncio.sleep(max(sleep_for, 0))


class CircuitBreaker:
    """Simple circuit breaker."""

    def __init__(self, threshold: int, timeout: int) -> None:
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.open_until = 0.0

    def allow(self) -> None:
        if time.monotonic() < self.open_until:
            raise RuntimeError("OpenAI circuit open")

    def success(self) -> None:
        self.failures = 0

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.open_until = time.monotonic() + self.timeout
            logger.warning("Circuit opened for %ss after %s failures", self.timeout, self.failures)


class OpenAIClient:
    """Wrapper around :class:`openai.AsyncOpenAI` with resiliency features."""

    def __init__(
        self,
        api_key: str,
        *,
        max_requests_per_minute: int = 60,
        retry_attempts: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60,
        cache: Optional[LRUCache] = None,
    ) -> None:
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.rate_limiter = SlidingWindowRateLimiter(max_requests_per_minute, 60)
        self.circuit_breaker = CircuitBreaker(circuit_breaker_threshold, circuit_breaker_timeout)
        self.cache = cache or LRUCache()
        self.retry_attempts = retry_attempts

    def _cache_key(self, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        payload = {"model": model, "messages": messages, "kwargs": kwargs}
        return json.dumps(payload, sort_keys=True)

    async def chat_completion(
        self, model: str, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Any:
        """Create a chat completion with retries, caching, and metrics."""
        key = self._cache_key(model, messages, **kwargs)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        self.circuit_breaker.allow()
        await self.rate_limiter.acquire()

        try:
            response = await self._create_completion(model=model, messages=messages, **kwargs)
        except Exception:
            self.circuit_breaker.failure()
            raise
        self.circuit_breaker.success()
        TOTAL_CALLS.inc()

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt = getattr(usage, "prompt_tokens", 0)
            completion = getattr(usage, "completion_tokens", 0)
            TOKENS_IN.inc(prompt)
            TOKENS_OUT.inc(completion)
            total_cost = (
                getattr(usage, "total_cost", None)
                or getattr(usage, "prompt_tokens_cost", 0)
                + getattr(usage, "completion_tokens_cost", 0)
            )
            TOTAL_COST.inc(total_cost)
            record_openai_cost(total_cost)

        self.cache.set(key, response)
        return response

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
    )
    async def _create_completion(self, **kwargs: Any) -> Any:
        return await self.client.chat.completions.create(**kwargs)
