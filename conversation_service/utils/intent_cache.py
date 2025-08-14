from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional

from .cache import LRUCache
from ..models.financial_models import IntentResult


@dataclass
class _CacheItem:
    """Internal cache item storing the result and metadata."""

    result: IntentResult
    vector: Dict[str, float]


class IntentResultCache:
    """Cache mapping user messages to :class:`IntentResult`.

    The cache uses an underlying :class:`LRUCache` for eviction and TTL support.
    A simple bag-of-words cosine similarity is employed to deduplicate semantically
    similar messages.  TTL is adapted dynamically based on the confidence score of
    the stored :class:`IntentResult`.
    """

    def __init__(
        self,
        max_size: int = 100,
        min_ttl: int = 30,
        max_ttl: int = 300,
        similarity_threshold: float = 0.9,
    ) -> None:
        self._store = LRUCache(maxsize=max_size, default_ttl=max_ttl)
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _vectorize(text: str) -> Dict[str, float]:
        tokens = [t for t in text.lower().split() if t]
        counts = Counter(tokens)
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return {k: v / norm for k, v in counts.items()}

    @staticmethod
    def _cosine_sim(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        return sum(v1.get(k, 0.0) * v2.get(k, 0.0) for k in v1)

    def _compute_ttl(self, confidence: float) -> int:
        span = self.max_ttl - self.min_ttl
        return int(self.min_ttl + max(0.0, min(confidence, 1.0)) * span)

    def _find_similar_key(self, vector: Dict[str, float]) -> Optional[str]:
        # Access underlying storage directly to compute similarity
        with self._store._lock:  # type: ignore[attr-defined]
            remove_keys = []
            for key, entry in list(self._store._cache.items()):  # type: ignore[attr-defined]
                if entry.is_expired():
                    remove_keys.append(key)
                    continue
                item: _CacheItem = entry.value
                if self._cosine_sim(vector, item.vector) >= self.similarity_threshold:
                    return key
            for k in remove_keys:
                del self._store._cache[k]  # type: ignore[attr-defined]
        return None

    # ------------------------------------------------------------------
    def get(self, message: str) -> Optional[IntentResult]:
        vector = self._vectorize(message)
        key = self._find_similar_key(vector)
        if key is None:
            self.misses += 1
            return None
        cached = self._store.get(key)
        if cached is None:
            self.misses += 1
            return None
        self.hits += 1
        item: _CacheItem = cached
        return item.result

    def set(self, message: str, result: IntentResult) -> None:
        vector = self._vectorize(message)
        key = self._find_similar_key(vector)
        if key is None:
            key = hashlib.sha256(message.lower().encode()).hexdigest()
        ttl = self._compute_ttl(result.confidence)
        item = _CacheItem(result=result, vector=vector)
        self._store.set(key, item, ttl=ttl)

    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": self._store._stats.size,  # type: ignore[attr-defined]
        }


__all__ = ["IntentResultCache"]
