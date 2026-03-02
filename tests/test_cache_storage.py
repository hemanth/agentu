"""Tests for pluggable cache storage backends."""

import pytest
import pytest_asyncio
import asyncio

from agentu.cache_storage_backends import CacheStorageBackend, MemoryBackend, SQLiteBackend
from agentu.cache_storage_backends import RedisBackend, FilesystemBackend


class TestMemoryBackend:
    @pytest_asyncio.fixture
    async def backend(self):
        return MemoryBackend(max_size=5)

    @pytest.mark.asyncio
    async def test_get_miss(self, backend):
        result = await backend.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, backend):
        await backend.set("key1", {"response": "hello"})
        result = await backend.get("key1")
        assert result == {"response": "hello"}

    @pytest.mark.asyncio
    async def test_delete(self, backend):
        await backend.set("key1", {"response": "hello"})
        await backend.delete("key1")
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear(self, backend):
        await backend.set("k1", {"r": "1"})
        await backend.set("k2", {"r": "2"})
        await backend.clear()
        assert await backend.get("k1") is None
        assert await backend.get("k2") is None

    @pytest.mark.asyncio
    async def test_stats(self, backend):
        await backend.set("k1", {"r": "1"})
        stats = await backend.stats()
        assert stats["entries"] == 1
        assert stats["backend"] == "memory"

    @pytest.mark.asyncio
    async def test_lru_eviction(self, backend):
        for i in range(6):
            await backend.set(f"key{i}", {"r": str(i)})
        # key0 should be evicted (max_size=5)
        assert await backend.get("key0") is None
        assert await backend.get("key5") == {"r": "5"}

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, backend):
        await backend.set("key1", {"r": "1"}, ttl=0)
        await asyncio.sleep(0.1)
        assert await backend.get("key1") is None


class TestSQLiteBackend:
    @pytest_asyncio.fixture
    async def backend(self, tmp_path):
        return SQLiteBackend(db_path=str(tmp_path / "test.db"))

    @pytest.mark.asyncio
    async def test_get_miss(self, backend):
        assert await backend.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, backend):
        await backend.set("key1", {"response": "hello"})
        result = await backend.get("key1")
        assert result == {"response": "hello"}

    @pytest.mark.asyncio
    async def test_delete(self, backend):
        await backend.set("key1", {"response": "hello"})
        await backend.delete("key1")
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear(self, backend):
        await backend.set("k1", {"r": "1"})
        await backend.set("k2", {"r": "2"})
        await backend.clear()
        assert await backend.get("k1") is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, backend):
        await backend.set("key1", {"r": "1"}, ttl=0)
        await asyncio.sleep(0.1)
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path):
        db = str(tmp_path / "persist.db")
        b1 = SQLiteBackend(db_path=db)
        await b1.set("key1", {"r": "1"})
        b2 = SQLiteBackend(db_path=db)
        assert await b2.get("key1") == {"r": "1"}

    @pytest.mark.asyncio
    async def test_stats(self, backend):
        await backend.set("k1", {"r": "1"})
        stats = await backend.stats()
        assert stats["backend"] == "sqlite"
        assert stats["entries"] == 1


class TestRedisBackend:
    """Tests use no real Redis — test graceful degradation."""

    @pytest.mark.asyncio
    async def test_unavailable_gracefully(self):
        backend = RedisBackend(url="redis://localhost:59999")
        result = await backend.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_stats_when_unavailable(self):
        backend = RedisBackend(url="redis://localhost:59999")
        stats = await backend.stats()
        assert stats["backend"] == "redis"
        assert stats["available"] is False


class TestFilesystemBackend:
    @pytest_asyncio.fixture
    async def backend(self, tmp_path):
        return FilesystemBackend(cache_dir=str(tmp_path / "fs_cache"))

    @pytest.mark.asyncio
    async def test_get_miss(self, backend):
        assert await backend.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, backend):
        await backend.set("key1", {"response": "hello"})
        result = await backend.get("key1")
        assert result == {"response": "hello"}

    @pytest.mark.asyncio
    async def test_delete(self, backend):
        await backend.set("key1", {"r": "1"})
        await backend.delete("key1")
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear(self, backend):
        await backend.set("k1", {"r": "1"})
        await backend.set("k2", {"r": "2"})
        await backend.clear()
        assert await backend.get("k1") is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, backend):
        import asyncio
        await backend.set("key1", {"r": "1"}, ttl=0)
        await asyncio.sleep(0.1)
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_stats(self, backend):
        await backend.set("k1", {"r": "1"})
        stats = await backend.stats()
        assert stats["backend"] == "filesystem"
        assert stats["entries"] == 1
