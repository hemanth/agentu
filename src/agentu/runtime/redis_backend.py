"""Redis-backed session storage for agentu.

Provides ``RedisSessionStore`` — an async session store that persists
session data as JSON in Redis with configurable TTL and key prefixing.

Redis is **optional**.  If the ``redis`` package is not installed, importing
this module will succeed but instantiating ``RedisSessionStore`` will raise
a clear ``ImportError`` with installation instructions.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --- optional redis import ---------------------------------------------------
try:
    import redis.asyncio as aioredis
    from redis.asyncio import ConnectionPool as RedisConnectionPool

    HAS_REDIS = True
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore[assignment]
    RedisConnectionPool = None  # type: ignore[assignment, misc]
    HAS_REDIS = False


_KEY_PREFIX = "agentu:session:"
_DEFAULT_TTL = 3600  # 1 hour


def _require_redis() -> None:
    """Raise a helpful error when redis is missing."""
    if not HAS_REDIS:
        raise ImportError(
            "The 'redis' package is required for RedisSessionStore. "
            "Install it with: pip install 'agentu[redis]'"
        )


class RedisSessionStore:
    """Async Redis-backed session storage.

    Stores session data as JSON strings under the key
    ``agentu:session:{session_id}`` with a configurable TTL.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
        ttl: Time-to-live for session keys in seconds (default: 3600).
        key_prefix: Custom key prefix (default: ``agentu:session:``).

    Raises:
        ImportError: If the ``redis`` package is not installed.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl: int = _DEFAULT_TTL,
        key_prefix: str = _KEY_PREFIX,
    ) -> None:
        _require_redis()
        self.ttl = ttl
        self.key_prefix = key_prefix
        self._pool: "RedisConnectionPool" = aioredis.ConnectionPool.from_url(
            redis_url, decode_responses=True
        )
        self._redis: "aioredis.Redis" = aioredis.Redis(connection_pool=self._pool)
        logger.info("RedisSessionStore connected to %s", redis_url)

    # -- key helpers ----------------------------------------------------------

    def _key(self, session_id: str) -> str:
        """Build the full Redis key for a session."""
        return f"{self.key_prefix}{session_id}"

    # -- public API -----------------------------------------------------------

    async def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Persist session data to Redis.

        Args:
            session_id: Unique session identifier.
            data: Arbitrary JSON-serialisable session data.
        """
        key = self._key(session_id)
        payload = json.dumps(data, default=str)
        await self._redis.set(key, payload, ex=self.ttl)
        logger.debug("Saved session %s (ttl=%ds)", session_id, self.ttl)

    async def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data from Redis.

        Args:
            session_id: Unique session identifier.

        Returns:
            Session data dict, or ``None`` if the key does not exist or has
            expired.
        """
        key = self._key(session_id)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from Redis.

        Args:
            session_id: Session to remove.

        Returns:
            ``True`` if the key existed and was deleted.
        """
        key = self._key(session_id)
        deleted = await self._redis.delete(key)
        return deleted > 0

    async def list_sessions(self) -> List[str]:
        """List all session IDs currently stored in Redis.

        Uses ``SCAN`` to avoid blocking on large keyspaces.

        Returns:
            List of session IDs (without the key prefix).
        """
        prefix_len = len(self.key_prefix)
        session_ids: List[str] = []
        async for key in self._redis.scan_iter(match=f"{self.key_prefix}*"):
            # key is already a str because of decode_responses=True
            session_ids.append(key[prefix_len:])
        return session_ids

    async def exists(self, session_id: str) -> bool:
        """Check whether a session exists in Redis.

        Args:
            session_id: Session to look up.

        Returns:
            ``True`` if the key exists and has not expired.
        """
        key = self._key(session_id)
        return bool(await self._redis.exists(key))

    async def close(self) -> None:
        """Close the Redis connection pool."""
        await self._redis.aclose()
        logger.debug("RedisSessionStore connection closed")
