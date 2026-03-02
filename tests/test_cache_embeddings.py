"""Tests for cache embedding providers."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agentu.cache_embeddings import (
    EmbeddingProvider, LocalEmbedding, APIEmbedding, FakeEmbedding, cosine_similarity
)


class TestFakeEmbedding:
    """FakeEmbedding is for testing - deterministic vectors."""

    @pytest.mark.asyncio
    async def test_embed_returns_list(self):
        provider = FakeEmbedding(dimension=8)
        result = await provider.embed("hello")
        assert isinstance(result, list)
        assert len(result) == 8

    @pytest.mark.asyncio
    async def test_same_input_same_output(self):
        provider = FakeEmbedding(dimension=8)
        a = await provider.embed("hello")
        b = await provider.embed("hello")
        assert a == b

    @pytest.mark.asyncio
    async def test_different_input_different_output(self):
        provider = FakeEmbedding(dimension=8)
        a = await provider.embed("hello")
        b = await provider.embed("goodbye")
        assert a != b

    def test_dimension(self):
        provider = FakeEmbedding(dimension=16)
        assert provider.dimension() == 16


class TestLocalEmbedding:
    @pytest.mark.asyncio
    async def test_graceful_when_not_installed(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            provider = LocalEmbedding()
            assert provider.available() is False


class TestAPIEmbedding:
    @pytest.mark.asyncio
    async def test_embed_calls_api(self):
        provider = APIEmbedding(api_base="http://localhost:11434/v1", model="nomic-embed-text")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.post", return_value=mock_resp):
            result = await provider.embed("hello")
            assert result == [0.1, 0.2, 0.3]


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 1]) == 0.0
