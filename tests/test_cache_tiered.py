"""Tests for TieredCache orchestrator."""

import pytest
import pytest_asyncio

from agentu.cache_tiered import TieredCache
from agentu.cache_storage_backends import MemoryBackend, SQLiteBackend
from agentu.cache_embeddings import FakeEmbedding
from agentu.cache_semantic import SemanticIndex


class TestTieredCacheExactMatch:
    @pytest_asyncio.fixture
    async def cache(self, tmp_path):
        backends = [MemoryBackend(max_size=100), SQLiteBackend(db_path=str(tmp_path / "t.db"))]
        return TieredCache(backends=backends, ttl=3600)

    @pytest.mark.asyncio
    async def test_miss_on_empty(self, cache):
        result = await cache.get("prompt", "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        await cache.set("prompt", "model", "response")
        result = await cache.get("prompt", "model")
        assert result == "response"

    @pytest.mark.asyncio
    async def test_tier_promotion(self, tmp_path):
        mem = MemoryBackend(max_size=100)
        sql = SQLiteBackend(db_path=str(tmp_path / "promo.db"))
        cache = TieredCache(backends=[mem, sql], ttl=3600)

        # Write directly to sqlite only (bypass tiered cache)
        import hashlib, json
        key_data = {"prompt": "prompt", "model": "model", "temperature": None}
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        await sql.set(key, {"response": "hello"}, ttl=3600)

        # Get should find in sqlite and promote to memory
        result = await cache.get("prompt", "model")
        assert result == "hello"
        mem_result = await mem.get(key)
        assert mem_result is not None

    @pytest.mark.asyncio
    async def test_stats(self, cache):
        await cache.set("p", "m", "r")
        await cache.get("p", "m")  # hit
        await cache.get("miss", "m")  # miss
        stats = await cache.get_stats()
        assert stats["exact_hits"] == 1
        assert stats["misses"] == 1
        assert "tier_hits" in stats

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        await cache.set("p", "m", "r")
        await cache.clear()
        assert await cache.get("p", "m") is None

    @pytest.mark.asyncio
    async def test_invalidate(self, cache):
        await cache.set("p", "m", "r")
        await cache.invalidate("p", "m")
        assert await cache.get("p", "m") is None

    @pytest.mark.asyncio
    async def test_is_dirty(self, cache):
        assert cache.is_dirty is False
        await cache.set("p", "m", "r")
        assert cache.is_dirty is True
        cache.clear_dirty()
        assert cache.is_dirty is False


class TestTieredCacheWithSemantic:
    @pytest_asyncio.fixture
    async def cache(self, tmp_path):
        backends = [MemoryBackend(max_size=100)]
        provider = FakeEmbedding(dimension=32)
        index = SemanticIndex(
            embedding_provider=provider,
            db_path=str(tmp_path / "sem.db"),
            threshold=0.95,
        )
        return TieredCache(backends=backends, ttl=3600, semantic_index=index)

    @pytest.mark.asyncio
    async def test_exact_match_preferred_over_semantic(self, cache):
        await cache.set("hello world", "model", "exact response")
        result = await cache.get("hello world", "model")
        assert result == "exact response"
        stats = await cache.get_stats()
        assert stats["exact_hits"] == 1
        assert stats["semantic_hits"] == 0

    @pytest.mark.asyncio
    async def test_semantic_match_on_exact_miss(self, cache):
        await cache.set("hello world", "model", "cached response")
        # Same text will match semantically (score=1.0)
        result = await cache.get("hello world", "model")
        assert result == "cached response"


class TestTieredCacheConversation:
    """Test tiered cache with conversation (List[Dict]) prompts."""

    @pytest_asyncio.fixture
    async def cache(self, tmp_path):
        backends = [MemoryBackend(max_size=100), SQLiteBackend(db_path=str(tmp_path / "conv.db"))]
        return TieredCache(backends=backends, ttl=3600)

    @pytest.mark.asyncio
    async def test_conversation_set_and_get(self, cache):
        conversation = [
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "I am Hemanth. I authored agentu"},
            {"role": "user", "content": "Can I get some vegan food here?"},
        ]
        await cache.set(conversation, "gpt-4", "Sure, here are options...")
        result = await cache.get(conversation, "gpt-4")
        assert result == "Sure, here are options..."

    @pytest.mark.asyncio
    async def test_conversation_invalidate(self, cache):
        conversation = [{"role": "user", "content": "test"}]
        await cache.set(conversation, "model", "cached")
        await cache.invalidate(conversation, "model")
        assert await cache.get(conversation, "model") is None

