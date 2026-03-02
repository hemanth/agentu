"""Pluggable cache storage backends for agentu."""

import time
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class CacheStorageBackend(Protocol):
    async def get(self, key: str) -> Optional[dict]: ...
    async def set(self, key: str, value: dict, ttl: Optional[int] = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...
    async def stats(self) -> Dict[str, Any]: ...


class MemoryBackend:
    """In-memory LRU cache backend."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._expiry: Dict[str, float] = {}

    async def get(self, key: str) -> Optional[dict]:
        if key not in self._store:
            return None
        if key in self._expiry and time.time() > self._expiry[key]:
            del self._store[key]
            del self._expiry[key]
            return None
        self._store.move_to_end(key)
        return dict(self._store[key])

    async def set(self, key: str, value: dict, ttl: Optional[int] = None) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if ttl is not None:
            self._expiry[key] = time.time() + ttl
        elif key in self._expiry:
            del self._expiry[key]
        while len(self._store) > self.max_size:
            oldest_key, _ = self._store.popitem(last=False)
            self._expiry.pop(oldest_key, None)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._expiry.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()
        self._expiry.clear()

    async def stats(self) -> Dict[str, Any]:
        return {"backend": "memory", "entries": len(self._store), "max_size": self.max_size}
