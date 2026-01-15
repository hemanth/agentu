"""LLM Response Caching for agentu.

Provides transparent caching of LLM responses with:
- Exact match using prompt hash (SHA256)
- Configurable TTL (time-to-live)
- SQLite storage
- Cache statistics
"""

import sqlite3
import hashlib
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache operations."""
    hits: int = 0
    misses: int = 0
    
    @property
    def size(self) -> int:
        return self.hits + self.misses
    
    @property
    def hit_rate(self) -> float:
        if self.size == 0:
            return 0.0
        return self.hits / self.size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": self.size,
            "hit_rate": round(self.hit_rate, 3)
        }


class LLMCache:
    """
    SQLite-based LLM response cache with TTL.
    
    Example:
        >>> cache = LLMCache(ttl=3600)  # 1 hour TTL
        >>> cache.get("prompt", "model")  # None
        >>> cache.set("prompt", "model", "response")
        >>> cache.get("prompt", "model")  # "response"
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        ttl: int = 3600,  # 1 hour default
        max_entries: int = 10000
    ):
        """
        Initialize the cache.
        
        Args:
            db_path: Path to SQLite database (default: ~/.agentu/cache.db)
            ttl: Time-to-live in seconds (default: 3600 = 1 hour)
            max_entries: Maximum cache entries before cleanup (default: 10000)
        """
        self.ttl = ttl
        self.max_entries = max_entries
        self.stats = CacheStats()
        
        if db_path is None:
            cache_dir = Path.home() / ".agentu"
            cache_dir.mkdir(exist_ok=True)
            db_path = str(cache_dir / "cache.db")
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                prompt_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                metadata TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON llm_cache(expires_at)
        """)
        conn.commit()
        conn.close()
    
    def _make_key(self, prompt: str, model: str, **kwargs) -> str:
        """Create a cache key from prompt and model."""
        # Include relevant kwargs in the key (temperature, etc.)
        key_data = {
            "prompt": prompt,
            "model": model,
            "temperature": kwargs.get("temperature"),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, **kwargs) -> Optional[str]:
        """
        Get cached response if available and not expired.
        
        Args:
            prompt: The prompt that was sent
            model: The model name
            **kwargs: Additional parameters that affect the response
            
        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._make_key(prompt, model, **kwargs)
        now = time.time()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT response FROM llm_cache
            WHERE cache_key = ? AND expires_at > ?
        """, (cache_key, now))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            self.stats.hits += 1
            logger.debug(f"Cache hit for key {cache_key[:8]}...")
            return row[0]
        
        self.stats.misses += 1
        logger.debug(f"Cache miss for key {cache_key[:8]}...")
        return None
    
    def set(
        self,
        prompt: str,
        model: str,
        response: str,
        metadata: Optional[Dict] = None,
        **kwargs
    ):
        """
        Store a response in the cache.
        
        Args:
            prompt: The prompt that was sent
            model: The model name
            response: The response to cache
            metadata: Optional metadata to store
            **kwargs: Additional parameters that affect caching
        """
        cache_key = self._make_key(prompt, model, **kwargs)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        now = time.time()
        expires_at = now + self.ttl
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO llm_cache
            (cache_key, prompt_hash, model, response, created_at, expires_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            cache_key,
            prompt_hash,
            model,
            response,
            now,
            expires_at,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Cached response for key {cache_key[:8]}...")
        
        # Cleanup if needed
        self._maybe_cleanup()
    
    def _maybe_cleanup(self):
        """Remove expired entries and enforce max_entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Remove expired entries
        cursor.execute("DELETE FROM llm_cache WHERE expires_at < ?", (time.time(),))
        
        # Check count
        cursor.execute("SELECT COUNT(*) FROM llm_cache")
        count = cursor.fetchone()[0]
        
        # Remove oldest if over limit
        if count > self.max_entries:
            excess = count - self.max_entries
            cursor.execute("""
                DELETE FROM llm_cache WHERE cache_key IN (
                    SELECT cache_key FROM llm_cache
                    ORDER BY created_at ASC
                    LIMIT ?
                )
            """, (excess,))
        
        conn.commit()
        conn.close()
    
    def clear(self):
        """Clear all cached entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM llm_cache")
        conn.commit()
        conn.close()
        self.stats = CacheStats()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM llm_cache WHERE expires_at > ?", (time.time(),))
        active_entries = cursor.fetchone()[0]
        conn.close()
        
        stats = self.stats.to_dict()
        stats["active_entries"] = active_entries
        return stats
    
    def invalidate(self, prompt: str, model: str, **kwargs):
        """Invalidate a specific cache entry."""
        cache_key = self._make_key(prompt, model, **kwargs)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM llm_cache WHERE cache_key = ?", (cache_key,))
        conn.commit()
        conn.close()
