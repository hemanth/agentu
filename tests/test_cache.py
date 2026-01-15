"""Tests for LLM response caching."""

import pytest
import tempfile
import time
from pathlib import Path

from agentu.cache import LLMCache, CacheStats


class TestCacheStats:
    """Test CacheStats dataclass."""
    
    def test_initial_stats(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.size == 0
        assert stats.hit_rate == 0.0
    
    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=7, misses=3)
        assert stats.size == 10
        assert stats.hit_rate == 0.7
    
    def test_to_dict(self):
        stats = CacheStats(hits=5, misses=5)
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["size"] == 10
        assert d["hit_rate"] == 0.5


class TestLLMCache:
    """Test LLMCache class."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        """Create a cache with temp database."""
        db_path = str(tmp_path / "test_cache.db")
        return LLMCache(db_path=db_path, ttl=3600)
    
    def test_cache_miss_on_empty(self, cache):
        """Test that empty cache returns None."""
        result = cache.get("prompt", "model")
        assert result is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0
    
    def test_cache_set_and_get(self, cache):
        """Test basic set and get."""
        cache.set("test prompt", "gpt-4", "test response")
        
        result = cache.get("test prompt", "gpt-4")
        assert result == "test response"
        assert cache.stats.hits == 1
    
    def test_cache_key_includes_model(self, cache):
        """Test that different models have different cache keys."""
        cache.set("prompt", "model-a", "response-a")
        cache.set("prompt", "model-b", "response-b")
        
        assert cache.get("prompt", "model-a") == "response-a"
        assert cache.get("prompt", "model-b") == "response-b"
    
    def test_cache_key_includes_temperature(self, cache):
        """Test that different temperatures have different cache keys."""
        cache.set("prompt", "model", "response-hot", temperature=1.0)
        cache.set("prompt", "model", "response-cold", temperature=0.0)
        
        assert cache.get("prompt", "model", temperature=1.0) == "response-hot"
        assert cache.get("prompt", "model", temperature=0.0) == "response-cold"
    
    def test_cache_expiration(self, tmp_path):
        """Test that expired entries are not returned."""
        db_path = str(tmp_path / "expire_test.db")
        # Very short TTL
        cache = LLMCache(db_path=db_path, ttl=1)
        
        cache.set("prompt", "model", "response")
        
        # Should hit before expiration
        assert cache.get("prompt", "model") == "response"
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should miss after expiration
        assert cache.get("prompt", "model") is None
    
    def test_cache_clear(self, cache):
        """Test clearing the cache."""
        cache.set("prompt1", "model", "response1")
        cache.set("prompt2", "model", "response2")
        
        cache.clear()
        
        # Stats should be reset after clear
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
        
        # Entries should be gone (these will increment misses)
        assert cache.get("prompt1", "model") is None
        assert cache.get("prompt2", "model") is None
    
    def test_cache_invalidate(self, cache):
        """Test invalidating a specific entry."""
        cache.set("prompt1", "model", "response1")
        cache.set("prompt2", "model", "response2")
        
        cache.invalidate("prompt1", "model")
        
        assert cache.get("prompt1", "model") is None
        assert cache.get("prompt2", "model") == "response2"
    
    def test_get_stats(self, cache):
        """Test getting extended stats."""
        cache.set("p1", "m", "r1")
        cache.set("p2", "m", "r2")
        cache.get("p1", "m")  # hit
        cache.get("missing", "m")  # miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["active_entries"] == 2
    
    def test_max_entries_cleanup(self, tmp_path):
        """Test that old entries are removed when max is exceeded."""
        db_path = str(tmp_path / "max_test.db")
        cache = LLMCache(db_path=db_path, ttl=3600, max_entries=5)
        
        # Add more than max entries
        for i in range(10):
            cache.set(f"prompt{i}", "model", f"response{i}")
            time.sleep(0.01)  # Ensure different timestamps
        
        stats = cache.get_stats()
        assert stats["active_entries"] <= 5


class TestLLMCacheDefaults:
    """Test default cache location."""
    
    def test_default_path_creation(self):
        """Test that default path is created."""
        cache = LLMCache(ttl=1)
        expected_path = Path.home() / ".agentu" / "cache.db"
        assert Path(cache.db_path) == expected_path
        # Cleanup
        cache.clear()
