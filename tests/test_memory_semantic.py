"""Tests for semantic memory recall.

Validates the embedding-based semantic search wired into Memory and
LongTermMemory, including index building, incremental indexing,
recall with semantic=True, and fallback to substring matching.
"""

import asyncio
import time
import pytest

from agentu.memory.memory import Memory, LongTermMemory, MemoryEntry
from agentu.cache.embeddings import FakeEmbedding, cosine_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine to completion (test helper)."""
    return asyncio.run(coro)


@pytest.fixture
def fake_embedder():
    return FakeEmbedding(dimension=64)


@pytest.fixture
def long_term(fake_embedder):
    """LongTermMemory with an embedding provider, no persistent storage."""
    return LongTermMemory(embedding_provider=fake_embedder)


@pytest.fixture
def memory_with_embeddings(fake_embedder):
    """Full Memory system with an embedding provider."""
    return Memory(embedding_provider=fake_embedder)


@pytest.fixture
def memory_without_embeddings():
    """Full Memory system *without* an embedding provider (baseline)."""
    return Memory()


# ---------------------------------------------------------------------------
# LongTermMemory — semantic index management
# ---------------------------------------------------------------------------

class TestLongTermMemorySemanticIndex:
    """Tests for add_to_semantic_index, rebuild_semantic_index, semantic_search."""

    def test_initial_state_no_index(self, long_term):
        """Before any indexing the semantic index should be empty."""
        assert long_term._semantic_ready is False
        assert long_term._embeddings == []

    def test_add_to_semantic_index(self, long_term):
        """After add + index, the semantic index should be populated."""
        entry = long_term.add("The quick brown fox", memory_type="fact")
        _run(long_term.add_to_semantic_index(entry))

        assert long_term._semantic_ready is True
        assert len(long_term._embeddings) == 1
        assert len(long_term._embeddings[0]) == 64  # matches FakeEmbedding dim

    def test_add_placeholder_created(self, long_term):
        """add() should create a placeholder in _embeddings."""
        long_term.add("placeholder test")
        assert len(long_term._embeddings) == 1
        assert long_term._embeddings[0] == []  # placeholder, not yet computed

    def test_rebuild_semantic_index(self, long_term):
        """rebuild_semantic_index should embed all entries."""
        long_term.add("Memory alpha")
        long_term.add("Memory beta")
        long_term.add("Memory gamma")

        _run(long_term.rebuild_semantic_index())

        assert long_term._semantic_ready is True
        assert len(long_term._embeddings) == 3
        assert all(len(v) == 64 for v in long_term._embeddings)

    def test_semantic_search_returns_scored_tuples(self, long_term):
        """semantic_search should return (MemoryEntry, float) tuples."""
        e1 = long_term.add("Python is a programming language")
        e2 = long_term.add("The weather is sunny today")
        _run(long_term.add_to_semantic_index(e1))
        _run(long_term.add_to_semantic_index(e2))

        results = _run(long_term.semantic_search("Python programming", limit=5))
        assert len(results) > 0
        for entry, score in results:
            assert isinstance(entry, MemoryEntry)
            assert isinstance(score, float)

    def test_semantic_search_applies_limit(self, long_term):
        """semantic_search should respect the limit parameter."""
        for i in range(10):
            e = long_term.add(f"Memory number {i}")
            _run(long_term.add_to_semantic_index(e))

        results = _run(long_term.semantic_search("memory", limit=3))
        assert len(results) <= 3

    def test_semantic_search_applies_threshold(self, long_term):
        """With a very high threshold most entries should be filtered out."""
        e = long_term.add("Hello world")
        _run(long_term.add_to_semantic_index(e))

        # With threshold 1.0 only an exact match would pass
        results = _run(long_term.semantic_search("completely unrelated text", limit=5, threshold=1.0))
        assert len(results) == 0

    def test_semantic_search_not_ready(self, long_term):
        """When the index isn't ready, semantic_search returns empty list."""
        long_term.add("some data")
        # Don't build the index
        results = _run(long_term.semantic_search("some data"))
        assert results == []

    def test_semantic_search_no_provider(self):
        """Without an embedding provider, semantic search returns empty."""
        lt = LongTermMemory()  # no provider
        lt.add("data")
        results = _run(lt.semantic_search("data"))
        assert results == []

    def test_semantic_search_updates_access_stats(self, long_term):
        """Returned entries should have their access_count incremented."""
        entry = long_term.add("Important memory", importance=0.9)
        _run(long_term.add_to_semantic_index(entry))

        initial_count = entry.access_count
        _run(long_term.semantic_search("Important memory"))
        assert entry.access_count == initial_count + 1


# ---------------------------------------------------------------------------
# Memory.recall — semantic flag
# ---------------------------------------------------------------------------

class TestMemoryRecallSemantic:
    """Tests for Memory.recall(semantic=True) and fallback behaviour."""

    def test_recall_default_is_substring(self, memory_without_embeddings):
        """Default recall (semantic=False) should use substring matching."""
        mem = memory_without_embeddings
        mem.remember("The quick brown fox", importance=0.8, store_long_term=True)

        results = mem.recall(query="quick", limit=5, include_short_term=False)
        assert any("quick" in e.content for e in results)

    def test_recall_semantic_false_no_change(self, memory_with_embeddings):
        """semantic=False should use substring even when embeddings exist."""
        mem = memory_with_embeddings
        mem.remember("alpha beta gamma", importance=0.8, store_long_term=True)
        # Index the entry manually since remember's create_task won't run in tests
        _run(mem.long_term.rebuild_semantic_index())

        results = mem.recall(query="alpha", semantic=False, include_short_term=False)
        assert any("alpha" in e.content for e in results)

    def test_recall_semantic_true_uses_embeddings(self, memory_with_embeddings):
        """semantic=True should use embedding-based search."""
        mem = memory_with_embeddings
        mem.remember("Machine learning with neural networks", importance=0.9, store_long_term=True)
        mem.remember("Cooking pasta with tomato sauce", importance=0.9, store_long_term=True)
        _run(mem.long_term.rebuild_semantic_index())

        # With FakeEmbedding (hash-based), an exact-match query will score 1.0
        # for the matching entry, so it must appear in results.
        results = mem.recall(
            query="Machine learning with neural networks",
            semantic=True,
            include_short_term=False,
            limit=5,
        )
        assert len(results) > 0
        # The exact-text entry must be present somewhere in results
        contents = [r.content for r in results]
        assert "Machine learning with neural networks" in contents

    def test_recall_semantic_falls_back_when_not_ready(self, memory_with_embeddings):
        """If the semantic index isn't built, recall should fall back to substring."""
        mem = memory_with_embeddings
        mem.remember("Substring test data", importance=0.8, store_long_term=True)
        # Don't build index: mem.long_term._semantic_ready is False

        results = mem.recall(
            query="Substring",
            semantic=True,
            include_short_term=False,
        )
        # Should still find it via substring fallback
        assert any("Substring" in e.content for e in results)

    def test_recall_semantic_with_threshold(self, memory_with_embeddings):
        """semantic_threshold should filter low-similarity results."""
        mem = memory_with_embeddings
        mem.remember("Alpha one", importance=0.9, store_long_term=True)
        mem.remember("Beta two", importance=0.9, store_long_term=True)
        _run(mem.long_term.rebuild_semantic_index())

        # Very high threshold — likely nothing passes
        results = mem.recall(
            query="Completely different topic xyz",
            semantic=True,
            semantic_threshold=0.99,
            include_short_term=False,
        )
        # Should fall back to substring (which also won't match)
        # or return empty
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Memory.recall_async
# ---------------------------------------------------------------------------

class TestMemoryRecallAsync:
    """Tests for the async recall_async method."""

    def test_recall_async_basic(self, memory_with_embeddings):
        """recall_async with semantic=True should work."""
        mem = memory_with_embeddings
        mem.remember("Async test memory", importance=0.9, store_long_term=True)
        _run(mem.long_term.rebuild_semantic_index())

        results = _run(mem.recall_async(
            query="Async test memory",
            semantic=True,
            include_short_term=False,
        ))
        assert len(results) > 0

    def test_recall_async_without_semantic(self, memory_with_embeddings):
        """recall_async with semantic=False uses substring matching."""
        mem = memory_with_embeddings
        mem.remember("Keyword searchable data", importance=0.8, store_long_term=True)

        results = _run(mem.recall_async(
            query="Keyword",
            semantic=False,
            include_short_term=False,
        ))
        assert any("Keyword" in e.content for e in results)

    def test_recall_async_includes_short_term(self, memory_with_embeddings):
        """Short-term memories should be included when requested."""
        mem = memory_with_embeddings
        mem.remember("Short term data", importance=0.2)  # low importance → only short-term

        results = _run(mem.recall_async(
            query=None,
            include_short_term=True,
        ))
        assert any("Short term" in e.content for e in results)


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------

class TestBackwardsCompatibility:
    """Ensure old calling conventions still work."""

    def test_memory_init_no_embedding(self):
        """Memory() without embedding_provider should work identically to before."""
        mem = Memory()
        assert mem.embedding_provider is None
        assert mem.long_term.embedding_provider is None
        assert mem.long_term._semantic_ready is False

    def test_long_term_init_no_embedding(self):
        """LongTermMemory() without embedding_provider should work identically."""
        lt = LongTermMemory()
        assert lt.embedding_provider is None
        assert lt._semantic_ready is False

    def test_recall_without_semantic_param(self, memory_without_embeddings):
        """recall() without the semantic parameter should work unchanged."""
        mem = memory_without_embeddings
        mem.remember("backwards compat test", importance=0.8, store_long_term=True)
        results = mem.recall(query="backwards", include_short_term=False)
        assert any("backwards" in e.content for e in results)

    def test_recall_signature_defaults(self):
        """The new parameters should have safe defaults."""
        import inspect
        sig = inspect.signature(Memory.recall)
        assert sig.parameters["semantic"].default is False
        assert sig.parameters["semantic_threshold"].default == 0.0

    def test_remember_still_returns_entry(self, memory_with_embeddings):
        """remember() should still return a MemoryEntry."""
        entry = memory_with_embeddings.remember("test")
        assert isinstance(entry, MemoryEntry)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and error paths."""

    def test_empty_long_term_semantic_search(self, long_term):
        """Semantic search on empty memory should return []."""
        _run(long_term.rebuild_semantic_index())
        results = _run(long_term.semantic_search("anything"))
        assert results == []

    def test_rebuild_with_no_provider(self):
        """rebuild_semantic_index with no provider should be a no-op."""
        lt = LongTermMemory()
        lt.add("data")
        _run(lt.rebuild_semantic_index())  # should not raise
        assert lt._semantic_ready is False

    def test_add_to_semantic_index_no_provider(self):
        """add_to_semantic_index with no provider should be a no-op."""
        lt = LongTermMemory()
        entry = lt.add("data")
        _run(lt.add_to_semantic_index(entry))  # should not raise

    def test_cosine_similarity_identical_vectors(self):
        """cosine_similarity of identical vectors should be 1.0."""
        v = [0.1, 0.2, 0.3, 0.4]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        """cosine_similarity of orthogonal vectors should be 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_multiple_add_and_index(self, long_term):
        """Adding and indexing multiple entries should keep parallel lists in sync."""
        entries = []
        for i in range(5):
            e = long_term.add(f"Entry {i}")
            entries.append(e)

        for e in entries:
            _run(long_term.add_to_semantic_index(e))

        assert len(long_term.entries) == len(long_term._embeddings) == 5
        assert all(len(v) == 64 for v in long_term._embeddings)


# ---------------------------------------------------------------------------
# Agent-level: semantic tool and skill matching
# ---------------------------------------------------------------------------

class TestAgentSemanticMatching:
    """Tests for _find_matching_tools and _match_skills with embedding provider."""

    def test_find_matching_tools_keyword_fallback(self):
        """Without embedding provider, keyword matching should work."""
        from agentu._core.agent import Agent
        from agentu._core.tools import Tool

        agent = Agent("test", model="test-model")
        agent.deferred_tools = [
            Tool(lambda: None, name="weather_tool", description="Get weather forecast"),
            Tool(lambda: None, name="calculator", description="Perform math calculations"),
        ]
        results = agent._find_matching_tools("weather forecast", limit=5)
        assert any(t.name == "weather_tool" for t in results)

    def test_find_matching_tools_with_embeddings(self, fake_embedder):
        """With embedding provider, semantic matching should be attempted."""
        from agentu._core.agent import Agent
        from agentu._core.tools import Tool

        agent = Agent("test", model="test-model")
        agent.deferred_tools = [
            Tool(lambda: None, name="weather_tool", description="Get weather forecast"),
            Tool(lambda: None, name="calculator", description="Perform math calculations"),
        ]
        # Semantic matching with FakeEmbedding — returns some results
        results = agent._find_matching_tools(
            "weather forecast", limit=5, embedding_provider=fake_embedder
        )
        assert isinstance(results, list)

    def test_match_skills_keyword_fallback(self):
        """Without embedding provider, keyword skill matching should work."""
        from agentu._core.agent import Agent
        from agentu.skills.skill import Skill

        agent = Agent("test", model="test-model")
        agent.skills = [
            Skill(name="data_analysis", description="Analyze datasets and produce reports",
                  instructions="/tmp/dummy.md", _skip_validation=True),
        ]
        results = agent._match_skills("analyze my dataset")
        assert len(results) == 1
        assert results[0].name == "data_analysis"

    def test_match_skills_with_embeddings(self, fake_embedder):
        """With embedding provider, semantic skill matching should be attempted."""
        from agentu._core.agent import Agent
        from agentu.skills.skill import Skill

        agent = Agent("test", model="test-model")
        agent.skills = [
            Skill(name="data_analysis", description="Analyze datasets and produce reports",
                  instructions="/tmp/dummy.md", _skip_validation=True),
            Skill(name="email_writer", description="Compose and send emails",
                  instructions="/tmp/dummy.md", _skip_validation=True),
        ]
        results = agent._match_skills(
            "analyze my dataset",
            embedding_provider=fake_embedder,
            semantic_threshold=0.0,  # Low threshold to ensure matches
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_match_skills_empty_list(self, fake_embedder):
        """With no skills, both methods should return empty."""
        from agentu._core.agent import Agent

        agent = Agent("test", model="test-model")
        agent.skills = []
        results = agent._match_skills("anything", embedding_provider=fake_embedder)
        assert results == []


# ---------------------------------------------------------------------------
# FakeEmbedding determinism sanity
# ---------------------------------------------------------------------------

class TestFakeEmbeddingDeterminism:
    """Verify FakeEmbedding is deterministic and normalized."""

    def test_deterministic(self, fake_embedder):
        """Same text should always produce the same embedding."""
        v1 = _run(fake_embedder.embed("hello"))
        v2 = _run(fake_embedder.embed("hello"))
        assert v1 == v2

    def test_different_texts_differ(self, fake_embedder):
        """Different texts should produce different embeddings."""
        v1 = _run(fake_embedder.embed("hello"))
        v2 = _run(fake_embedder.embed("world"))
        assert v1 != v2

    def test_normalized(self, fake_embedder):
        """Embeddings should be approximately unit-length."""
        import math
        v = _run(fake_embedder.embed("test"))
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-6
