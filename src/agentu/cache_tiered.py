"""TieredCache - unified cache orchestrator with optional semantic matching."""

import hashlib
import json
import logging
from typing import Optional, Dict, Any, List, Union

from .cache_storage_backends import CacheStorageBackend
from .cache_semantic import SemanticIndex

logger = logging.getLogger(__name__)

# A prompt can be a plain string or a conversation (list of message dicts)
Prompt = Union[str, List[Dict[str, str]]]


class TieredCache:
    """Orchestrates tiered storage backends with optional semantic index."""

    def __init__(
        self,
        backends: List[CacheStorageBackend],
        ttl: int = 3600,
        semantic_index: Optional[SemanticIndex] = None,
    ):
        self.backends = backends
        self.ttl = ttl
        self.semantic_index = semantic_index
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._tier_hits: Dict[str, int] = {}
        self._dirty = False

    @staticmethod
    def _normalize_prompt(prompt: Prompt) -> str:
        """Normalize a prompt to a deterministic string for hashing."""
        if isinstance(prompt, list):
            return json.dumps(prompt, sort_keys=True, ensure_ascii=False)
        return prompt

    def _make_key(self, prompt: Prompt, namespace: str, **kwargs) -> str:
        key_data = {"prompt": self._normalize_prompt(prompt), "model": namespace, "temperature": kwargs.get("temperature")}
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    async def get(self, prompt: Prompt, namespace: str, **kwargs) -> Optional[str]:
        cache_key = self._make_key(prompt, namespace, **kwargs)

        # Try exact match across tiers
        for i, backend in enumerate(self.backends):
            result = await backend.get(cache_key)
            if result is not None:
                self._exact_hits += 1
                backend_stats = await backend.stats()
                tier_name = backend_stats.get("backend", f"tier_{i}")
                self._tier_hits[tier_name] = self._tier_hits.get(tier_name, 0) + 1
                # Promote to higher tiers
                for j in range(i):
                    await self.backends[j].set(cache_key, result, ttl=self.ttl)
                return result.get("response")

        # Try semantic match
        if self.semantic_index is not None:
            matched_key = await self.semantic_index.search(prompt)
            if matched_key is not None:
                for backend in self.backends:
                    result = await backend.get(matched_key)
                    if result is not None:
                        self._semantic_hits += 1
                        return result.get("response")

        self._misses += 1
        return None

    async def set(self, prompt: Prompt, namespace: str, response: str, **kwargs) -> None:
        cache_key = self._make_key(prompt, namespace, **kwargs)
        normalized = self._normalize_prompt(prompt)
        value = {"response": response, "model": namespace, "prompt_hash": hashlib.sha256(normalized.encode()).hexdigest()}

        for backend in self.backends:
            await backend.set(cache_key, value, ttl=self.ttl)

        if self.semantic_index is not None:
            await self.semantic_index.add(prompt, cache_key)

        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def clear_dirty(self):
        self._dirty = False

    async def clear(self) -> None:
        for backend in self.backends:
            await backend.clear()
        if self.semantic_index:
            await self.semantic_index.clear()
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._tier_hits.clear()
        self._dirty = False

    async def invalidate(self, prompt: Prompt, namespace: str, **kwargs) -> None:
        cache_key = self._make_key(prompt, namespace, **kwargs)
        for backend in self.backends:
            await backend.delete(cache_key)
        if self.semantic_index:
            await self.semantic_index.remove(cache_key)

    async def get_stats(self) -> Dict[str, Any]:
        total = self._exact_hits + self._semantic_hits + self._misses
        stats = {
            "exact_hits": self._exact_hits,
            "semantic_hits": self._semantic_hits,
            "misses": self._misses,
            "hit_rate": round((self._exact_hits + self._semantic_hits) / total, 3) if total > 0 else 0.0,
            "tier_hits": dict(self._tier_hits),
        }
        if self.semantic_index:
            sem_stats = await self.semantic_index.stats()
            stats["embedding_count"] = sem_stats["embedding_count"]
        return stats
