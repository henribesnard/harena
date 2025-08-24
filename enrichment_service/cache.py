from collections import OrderedDict
from typing import Any, Dict, Optional

class AccountLRUCache:
    """Simple LRU cache for account information.

    This implementation keeps the API intentionally tiny: ``get`` returns
    a previously cached value or ``None`` and ``set`` stores a value while
    enforcing a maximum size.  The behaviour is sufficient for the unit
    tests which only require basic caching semantics.
    """

    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self._store: "OrderedDict[int, Any]" = OrderedDict()

    def get(self, key: int) -> Optional[Any]:
        try:
            value = self._store.pop(key)
        except KeyError:
            return None
        self._store[key] = value  # move to end (most recent)
        return value

    def set(self, key: int, value: Any) -> None:
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self.maxsize:
            self._store.popitem(last=False)
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()


class MerchantCategoryCache:
    """Very small in-memory cache mapping merchant names to categories."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, merchant: str) -> Optional[Any]:
        return self._store.get(merchant)

    def set(self, merchant: str, category: Any) -> None:
        self._store[merchant] = category

    def clear(self) -> None:
        self._store.clear()
