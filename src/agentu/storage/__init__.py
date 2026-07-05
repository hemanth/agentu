"""Pluggable storage backend abstraction for agentu.

Provides a unified interface for key-value and vector storage,
allowing components (sessions, checkpoints, memory, findings)
to swap between SQLite, Redis, or custom backends.

Usage:
    # Default — SQLite, zero-config
    agent = Agent("bot")

    # Redis — horizontal scaling
    agent = Agent("bot").with_backend("redis://localhost:6379")

    # Vectors — pgvector for production
    agent = Agent("bot").with_vectors("postgresql://localhost/agentu")
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


# ── Key-Value Storage ─────────────────────────────────────────────

@runtime_checkable
class StorageBackend(Protocol):
    """Interface for key-value storage backends.

    Used by: sessions, checkpoints, memory, schedule findings.
    """

    async def get(self, key: str) -> Optional[bytes]:
        """Get a value by key."""
        ...

    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        """Set a value with optional TTL in seconds."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a key. Returns True if existed."""
        ...

    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys matching a prefix."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...


class InMemoryBackend:
    """In-memory storage backend. Fast, no dependencies, lost on restart.

    Suitable for development, testing, and short-lived agents.
    """

    def __init__(self):
        self._store: Dict[str, Tuple[bytes, Optional[float]]] = {}

    async def get(self, key: str) -> Optional[bytes]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        expires_at = time.time() + ttl if ttl else None
        self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    async def list_keys(self, prefix: str = "") -> List[str]:
        now = time.time()
        return [
            k for k, (_, exp) in self._store.items()
            if k.startswith(prefix) and (exp is None or now < exp)
        ]

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None


class RedisStorageBackend:
    """Redis-backed storage backend for horizontal scaling.

    Requires: pip install 'agentu[redis]'

    Usage:
        backend = await RedisStorageBackend.create("redis://localhost:6379")
        await backend.set("key", b"value", ttl=3600)
    """

    def __init__(self, client, prefix: str = "agentu:"):
        self._client = client
        self._prefix = prefix

    @classmethod
    async def create(
        cls,
        redis_url: str,
        prefix: str = "agentu:",
        max_connections: int = 10,
    ) -> "RedisStorageBackend":
        """Create a Redis backend from a URL."""
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "Redis support requires the 'redis' package. "
                "Install it with: pip install 'agentu[redis]'"
            )

        pool = aioredis.ConnectionPool.from_url(
            redis_url, max_connections=max_connections
        )
        client = aioredis.Redis(connection_pool=pool)
        return cls(client, prefix)

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Optional[bytes]:
        result = await self._client.get(self._key(key))
        return result

    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        if ttl:
            await self._client.setex(self._key(key), ttl, value)
        else:
            await self._client.set(self._key(key), value)

    async def delete(self, key: str) -> bool:
        return bool(await self._client.delete(self._key(key)))

    async def list_keys(self, prefix: str = "") -> List[str]:
        pattern = f"{self._prefix}{prefix}*"
        keys = []
        async for key in self._client.scan_iter(match=pattern, count=100):
            # Strip the backend prefix to return clean keys
            k = key.decode() if isinstance(key, bytes) else key
            if k.startswith(self._prefix):
                k = k[len(self._prefix):]
            keys.append(k)
        return keys

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(self._key(key)))

    async def close(self):
        """Close the Redis connection pool."""
        await self._client.aclose()


# ── Vector Storage ────────────────────────────────────────────────

@runtime_checkable
class VectorBackend(Protocol):
    """Interface for vector similarity search backends.

    Used by: semantic memory recall, semantic cache, skill matching.
    """

    async def upsert(
        self,
        key: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a vector with metadata."""
        ...

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Search for similar vectors.

        Returns list of (key, score, metadata) tuples sorted by descending similarity.
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete a vector by key."""
        ...

    async def count(self) -> int:
        """Return the number of stored vectors."""
        ...


class InMemoryVectorBackend:
    """In-memory vector storage with brute-force cosine similarity.

    Suitable for development and small collections (<10K vectors).
    """

    def __init__(self):
        self._vectors: Dict[str, Tuple[List[float], Optional[Dict[str, Any]]]] = {}

    async def upsert(
        self,
        key: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._vectors[key] = (embedding, metadata)

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        if not self._vectors:
            return []

        results = []
        for key, (embedding, metadata) in self._vectors.items():
            score = _cosine_similarity(query_embedding, embedding)
            if score >= threshold:
                results.append((key, score, metadata))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def delete(self, key: str) -> bool:
        return self._vectors.pop(key, None) is not None

    async def count(self) -> int:
        return len(self._vectors)


class PgVectorBackend:
    """PostgreSQL + pgvector backend for production vector search.

    Requires: pip install 'agentu[pgvector]'

    Usage:
        backend = await PgVectorBackend.create(
            "postgresql://localhost/agentu",
            dimension=384,
        )
        await backend.upsert("doc1", embedding, {"source": "wiki"})
        results = await backend.search(query_vec, limit=5)

    Note: This is a stub interface for 2.0. Full implementation
    will ship in 2.1 with proper connection pooling, index creation,
    and batch operations.
    """

    def __init__(self, pool, table: str = "agentu_vectors", dimension: int = 384):
        self._pool = pool
        self._table = table
        self._dimension = dimension

    @classmethod
    async def create(
        cls,
        dsn: str,
        table: str = "agentu_vectors",
        dimension: int = 384,
    ) -> "PgVectorBackend":
        """Create a pgvector backend from a PostgreSQL DSN."""
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "pgvector support requires 'asyncpg' and 'pgvector'. "
                "Install with: pip install 'agentu[pgvector]'"
            )

        pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)

        # Create table and extension if not exists
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    key TEXT PRIMARY KEY,
                    embedding vector({dimension}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Create HNSW index for fast ANN search
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_embedding
                ON {table} USING hnsw (embedding vector_cosine_ops)
            """)

        return cls(pool, table, dimension)

    async def upsert(
        self,
        key: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        vec_str = f"[{','.join(str(v) for v in embedding)}]"
        meta_json = json.dumps(metadata or {})
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table} (key, embedding, metadata)
                VALUES ($1, $2::vector, $3::jsonb)
                ON CONFLICT (key) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """, key, vec_str, meta_json)

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        vec_str = f"[{','.join(str(v) for v in query_embedding)}]"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT key,
                       1 - (embedding <=> $1::vector) AS similarity,
                       metadata
                FROM {self._table}
                WHERE 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """, vec_str, threshold, limit)

        return [
            (row["key"], float(row["similarity"]), json.loads(row["metadata"]))
            for row in rows
        ]

    async def delete(self, key: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self._table} WHERE key = $1", key
            )
            return result == "DELETE 1"

    async def count(self) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(f"SELECT COUNT(*) FROM {self._table}")

    async def close(self):
        """Close the connection pool."""
        await self._pool.close()


# ── Helpers ───────────────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
