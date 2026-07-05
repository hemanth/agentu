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


class LanceDBBackend:
    """LanceDB-backed vector storage for production semantic search.

    Embedded, serverless, zero-config — like SQLite for vectors.

    Requires: pip install 'agentu[vectors]'

    Usage:
        backend = LanceDBBackend("./vectors")
        await backend.upsert("doc1", embedding, {"source": "wiki"})
        results = await backend.search(query_vec, limit=5)

        # Or use a cloud URI
        backend = LanceDBBackend("s3://bucket/vectors")
    """

    def __init__(self, db, table_name: str = "agentu_vectors"):
        self._db = db
        self._table_name = table_name
        self._table = None

    @classmethod
    def create(
        cls,
        uri: str = "./vectors",
        table_name: str = "agentu_vectors",
    ) -> "LanceDBBackend":
        """Create a LanceDB backend.

        Args:
            uri: Path to local directory or cloud URI (s3://, gs://, az://).
            table_name: Name of the vector table.
        """
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "LanceDB support requires the 'lancedb' package. "
                "Install it with: pip install 'agentu[vectors]'"
            )

        db = lancedb.connect(uri)
        return cls(db, table_name)

    def _get_or_create_table(self, embedding: List[float]):
        """Lazily create the table on first upsert."""
        if self._table is not None:
            return self._table

        try:
            self._table = self._db.open_table(self._table_name)
        except Exception:
            logger.debug("LanceDB table '%s' not found, creating", self._table_name, exc_info=True)
            data = [{
                "key": "__init__",
                "vector": embedding,
                "metadata": "{}",
            }]
            self._table = self._db.create_table(self._table_name, data=data)
            # Remove the init row
            self._table.delete('key = "__init__"')

        return self._table

    async def upsert(
        self,
        key: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        table = self._get_or_create_table(embedding)
        row = {
            "key": key,
            "vector": embedding,
            "metadata": json.dumps(metadata or {}),
        }
        try:
            (
                table.merge_insert("key")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )
        except Exception:
            # Fallback: delete + add (older lancedb versions)
            try:
                table.delete(f'key = "{key}"')
            except Exception:
                logger.debug("LanceDB delete fallback also failed", exc_info=True)
            table.add([row])

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        try:
            table = self._get_or_create_table(query_embedding)
        except Exception:
            logger.debug("LanceDB table access failed for search", exc_info=True)
            return []

        try:
            results = (
                table.search(query_embedding)
                .metric("cosine")
                .limit(limit)
                .to_list()
            )
        except Exception:
            logger.debug("LanceDB search query failed", exc_info=True)
            return []

        output = []
        for row in results:
            # LanceDB returns _distance (lower = more similar for cosine)
            distance = row.get("_distance", 1.0)
            similarity = 1.0 - distance
            if similarity >= threshold:
                meta_str = row.get("metadata", "{}")
                meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
                output.append((row["key"], similarity, meta))

        return output

    async def delete(self, key: str) -> bool:
        try:
            table = self._get_or_create_table([0.0])
            table.delete(f'key = "{key}"')
            return True
        except Exception:
            logger.debug("LanceDB delete failed", exc_info=True)
            return False

    async def count(self) -> int:
        try:
            table = self._get_or_create_table([0.0])
            return table.count_rows()
        except Exception:
            logger.debug("LanceDB count failed", exc_info=True)
            return 0

    def close(self):
        """Close the LanceDB connection."""
        self._db = None
        self._table = None


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
