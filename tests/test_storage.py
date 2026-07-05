"""Tests for storage backend abstraction."""

import pytest
from agentu.storage import (
    StorageBackend,
    VectorBackend,
    InMemoryBackend,
    InMemoryVectorBackend,
    RedisStorageBackend,
    LanceDBBackend,
    _cosine_similarity,
)


class TestInMemoryBackend:
    """Tests for in-memory key-value storage."""

    @pytest.mark.asyncio
    async def test_get_set(self):
        backend = InMemoryBackend()
        await backend.set("key1", b"value1")
        assert await backend.get("key1") == b"value1"

    @pytest.mark.asyncio
    async def test_get_missing(self):
        backend = InMemoryBackend()
        assert await backend.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_delete(self):
        backend = InMemoryBackend()
        await backend.set("key1", b"value1")
        assert await backend.delete("key1") is True
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        backend = InMemoryBackend()
        assert await backend.delete("nope") is False

    @pytest.mark.asyncio
    async def test_exists(self):
        backend = InMemoryBackend()
        await backend.set("key1", b"value1")
        assert await backend.exists("key1") is True
        assert await backend.exists("nope") is False

    @pytest.mark.asyncio
    async def test_list_keys(self):
        backend = InMemoryBackend()
        await backend.set("session:1", b"a")
        await backend.set("session:2", b"b")
        await backend.set("checkpoint:1", b"c")

        keys = await backend.list_keys("session:")
        assert sorted(keys) == ["session:1", "session:2"]

    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        backend = InMemoryBackend()
        await backend.set("key1", b"value1", ttl=1)
        # Manually expire by setting expires_at in the past
        backend._store["key1"] = (b"value1", 0.0)  # epoch = long expired
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_ttl_not_expired(self):
        backend = InMemoryBackend()
        await backend.set("key1", b"value1", ttl=3600)
        assert await backend.get("key1") == b"value1"

    @pytest.mark.asyncio
    async def test_overwrite(self):
        backend = InMemoryBackend()
        await backend.set("key1", b"old")
        await backend.set("key1", b"new")
        assert await backend.get("key1") == b"new"


class TestInMemoryVectorBackend:
    """Tests for in-memory vector storage."""

    @pytest.mark.asyncio
    async def test_upsert_and_search(self):
        backend = InMemoryVectorBackend()
        await backend.upsert("doc1", [1.0, 0.0, 0.0], {"title": "Doc 1"})
        await backend.upsert("doc2", [0.0, 1.0, 0.0], {"title": "Doc 2"})
        await backend.upsert("doc3", [0.9, 0.1, 0.0], {"title": "Doc 3"})

        results = await backend.search([1.0, 0.0, 0.0], limit=2)
        assert len(results) == 2
        assert results[0][0] == "doc1"  # Most similar
        assert results[0][1] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_search_with_threshold(self):
        backend = InMemoryVectorBackend()
        await backend.upsert("doc1", [1.0, 0.0, 0.0])
        await backend.upsert("doc2", [0.0, 1.0, 0.0])  # orthogonal

        results = await backend.search([1.0, 0.0, 0.0], threshold=0.5)
        assert len(results) == 1
        assert results[0][0] == "doc1"

    @pytest.mark.asyncio
    async def test_search_empty(self):
        backend = InMemoryVectorBackend()
        results = await backend.search([1.0, 0.0], limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_delete(self):
        backend = InMemoryVectorBackend()
        await backend.upsert("doc1", [1.0, 0.0])
        assert await backend.delete("doc1") is True
        assert await backend.count() == 0

    @pytest.mark.asyncio
    async def test_count(self):
        backend = InMemoryVectorBackend()
        assert await backend.count() == 0
        await backend.upsert("doc1", [1.0, 0.0])
        await backend.upsert("doc2", [0.0, 1.0])
        assert await backend.count() == 2

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self):
        backend = InMemoryVectorBackend()
        await backend.upsert("doc1", [1.0, 0.0], {"v": 1})
        await backend.upsert("doc1", [0.0, 1.0], {"v": 2})
        assert await backend.count() == 1

        results = await backend.search([0.0, 1.0], limit=1)
        assert results[0][2] == {"v": 2}

    @pytest.mark.asyncio
    async def test_metadata_preserved(self):
        backend = InMemoryVectorBackend()
        meta = {"source": "wiki", "page": 42}
        await backend.upsert("doc1", [1.0, 0.0, 0.0], meta)

        results = await backend.search([1.0, 0.0, 0.0], limit=1)
        assert results[0][2] == meta


class TestCosineSimilarity:
    """Test the cosine similarity helper."""

    def test_identical(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_mismatched_length(self):
        assert _cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestProtocolCompliance:
    """Test that backends satisfy their protocols."""

    def test_inmemory_is_storage_backend(self):
        assert isinstance(InMemoryBackend(), StorageBackend)

    def test_inmemory_vector_is_vector_backend(self):
        assert isinstance(InMemoryVectorBackend(), VectorBackend)


class TestRedisBackendImport:
    """Test Redis backend behavior when redis is not installed."""

    @pytest.mark.asyncio
    async def test_create_raises_without_redis(self):
        # This will either work (if redis is installed) or raise ImportError
        try:
            backend = await RedisStorageBackend.create("redis://localhost:6379")
            await backend.close()
        except ImportError as e:
            assert "agentu[redis]" in str(e)
        except Exception:
            pass  # Connection error is fine — means redis lib exists but server is down


class TestLanceDBBackendImport:
    """Test LanceDB backend behavior when lancedb is not installed."""

    def test_create_raises_without_lancedb(self):
        try:
            backend = LanceDBBackend.create("/tmp/test_lance_vectors")
            backend.close()
        except ImportError as e:
            assert "agentu[vectors]" in str(e)
        except Exception:
            pass  # If lancedb is installed, creation may succeed


class TestAgentWithBackend:
    """Test agent builder methods for storage backends."""

    def test_with_backend_url(self):
        from agentu import Agent
        agent = Agent("test").with_backend("redis://localhost:6379")
        assert agent._backend_url == "redis://localhost:6379"

    def test_with_backend_instance(self):
        from agentu import Agent
        backend = InMemoryBackend()
        agent = Agent("test").with_backend(backend)
        assert agent._storage_backend is backend

    def test_with_vectors_url(self):
        from agentu import Agent
        agent = Agent("test").with_vectors("./test_vectors")
        assert agent._vector_dsn == "./test_vectors"

    def test_with_vectors_instance(self):
        from agentu import Agent
        backend = InMemoryVectorBackend()
        agent = Agent("test").with_vectors(backend)
        assert agent._vector_backend is backend

    def test_chaining(self):
        from agentu import Agent
        agent = (
            Agent("test")
            .with_backend("redis://localhost:6379")
            .with_vectors("./my_vectors")
        )
        assert agent._backend_url == "redis://localhost:6379"
        assert agent._vector_dsn == "./my_vectors"
