"""Tests for semantic cache index."""

import pytest
import pytest_asyncio

from agentu.cache_semantic import SemanticIndex
from agentu.cache_embeddings import FakeEmbedding


class TestSemanticIndex:
    @pytest_asyncio.fixture
    async def index(self, tmp_path):
        provider = FakeEmbedding(dimension=32)
        return SemanticIndex(
            embedding_provider=provider,
            db_path=str(tmp_path / "semantic.db"),
            threshold=0.95,
        )

    @pytest.mark.asyncio
    async def test_no_match_on_empty(self, index):
        result = await index.search("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_exact_text_matches(self, index):
        await index.add("hello world", "cache_key_1")
        result = await index.search("hello world")
        assert result == "cache_key_1"

    @pytest.mark.asyncio
    async def test_different_text_no_match(self, index):
        await index.add("hello world", "cache_key_1")
        result = await index.search("something completely different and unrelated")
        assert result is None

    @pytest.mark.asyncio
    async def test_remove(self, index):
        await index.add("hello world", "cache_key_1")
        await index.remove("cache_key_1")
        result = await index.search("hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear(self, index):
        await index.add("hello", "k1")
        await index.add("world", "k2")
        await index.clear()
        assert await index.search("hello") is None

    @pytest.mark.asyncio
    async def test_stats(self, index):
        await index.add("hello", "k1")
        stats = await index.stats()
        assert stats["embedding_count"] == 1
