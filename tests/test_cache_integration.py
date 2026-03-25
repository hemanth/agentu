"""Tests for Agent.with_cache() integration."""

import pytest
from unittest.mock import patch

from agentu._core.agent import Agent
from agentu.cache.tiered import TieredCache


class TestWithCacheMethod:
    def test_with_cache_returns_self(self):
        agent = Agent("test", cache=False, enable_memory=False)
        result = agent.with_cache()
        assert result is agent

    def test_with_cache_basic_preset(self):
        agent = Agent("test", cache=False, enable_memory=False).with_cache(preset="basic")
        assert agent.cache is not None
        assert isinstance(agent.cache, TieredCache)

    def test_with_cache_no_preset_defaults_to_basic(self):
        agent = Agent("test", cache=False, enable_memory=False).with_cache()
        assert agent.cache is not None
        assert isinstance(agent.cache, TieredCache)

    def test_with_cache_custom_ttl(self):
        agent = Agent("test", cache=False, enable_memory=False).with_cache(ttl=7200)
        assert agent.cache is not None
        assert agent.cache.ttl == 7200

    def test_backward_compat_constructor_cache(self):
        agent = Agent("test", cache=True, cache_ttl=1800, enable_memory=False)
        assert agent.cache is not None
        assert isinstance(agent.cache, TieredCache)
        assert agent.cache.ttl == 1800

    def test_with_cache_enables_cache(self):
        agent = Agent("test", cache=False, enable_memory=False)
        assert agent.cache_enabled is False
        agent.with_cache()
        assert agent.cache_enabled is True


class TestWithCachePresets:
    def test_smart_preset_without_sentence_transformers(self):
        # sentence-transformers likely not installed in test env
        # semantic_index should be None (graceful degradation)
        agent = Agent("test", cache=False, enable_memory=False).with_cache(preset="smart")
        assert agent.cache is not None

    def test_offline_preset_has_sync(self):
        agent = Agent("test", cache=False, enable_memory=False).with_cache(preset="offline")
        assert agent._cache_sync is not None

    def test_distributed_preset(self):
        agent = Agent("test", cache=False, enable_memory=False).with_cache(
            preset="distributed",
            redis_url="redis://localhost:59999"
        )
        assert agent.cache is not None
