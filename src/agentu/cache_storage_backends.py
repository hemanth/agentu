"""Pluggable cache storage backends for agentu."""

import time
import json
import sqlite3
import hashlib
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class CacheStorageBackend(Protocol):
    async def get(self, key: str) -> Optional[dict]: ...
    async def set(self, key: str, value: dict, ttl: Optional[int] = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...
    async def stats(self) -> Dict[str, Any]: ...


class MemoryBackend:
    """In-memory LRU cache backend."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._expiry: Dict[str, float] = {}

    async def get(self, key: str) -> Optional[dict]:
        if key not in self._store:
            return None
        if key in self._expiry and time.time() > self._expiry[key]:
            del self._store[key]
            del self._expiry[key]
            return None
        self._store.move_to_end(key)
        return dict(self._store[key])

    async def set(self, key: str, value: dict, ttl: Optional[int] = None) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if ttl is not None:
            self._expiry[key] = time.time() + ttl
        elif key in self._expiry:
            del self._expiry[key]
        while len(self._store) > self.max_size:
            oldest_key, _ = self._store.popitem(last=False)
            self._expiry.pop(oldest_key, None)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._expiry.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()
        self._expiry.clear()

    async def stats(self) -> Dict[str, Any]:
        return {"backend": "memory", "entries": len(self._store), "max_size": self.max_size}


class SQLiteBackend:
    """SQLite-based persistent cache backend."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            from pathlib import Path
            cache_dir = Path.home() / ".agentu"
            cache_dir.mkdir(exist_ok=True)
            db_path = str(cache_dir / "cache.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_store(expires_at)")
            conn.commit()
        finally:
            conn.close()

    async def get(self, key: str) -> Optional[dict]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT value FROM cache_store WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
                (key, time.time())
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
        finally:
            conn.close()

    async def set(self, key: str, value: dict, ttl: Optional[int] = None) -> None:
        now = time.time()
        expires_at = now + ttl if ttl is not None else None
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache_store (key, value, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (key, json.dumps(value), now, expires_at)
            )
            conn.commit()
        finally:
            conn.close()

    async def delete(self, key: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM cache_store WHERE key = ?", (key,))
            conn.commit()
        finally:
            conn.close()

    async def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM cache_store")
            conn.commit()
        finally:
            conn.close()

    async def stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM cache_store WHERE expires_at IS NULL OR expires_at > ?",
                (time.time(),)
            )
            count = cursor.fetchone()[0]
            return {"backend": "sqlite", "entries": count, "db_path": self.db_path}
        finally:
            conn.close()


class RedisBackend:
    """Redis cache backend. Requires `pip install redis`."""

    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "agentu:cache:"):
        self.url = url
        self.prefix = prefix
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self.url, decode_responses=True)
                self._client.ping()
            except Exception:
                self._client = None
        return self._client

    async def get(self, key: str) -> Optional[dict]:
        client = self._get_client()
        if client is None:
            return None
        try:
            val = client.get(self.prefix + key)
            return json.loads(val) if val else None
        except Exception:
            logger.warning("Redis get failed", exc_info=True)
            return None

    async def set(self, key: str, value: dict, ttl: Optional[int] = None) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            data = json.dumps(value)
            if ttl is not None:
                client.setex(self.prefix + key, ttl, data)
            else:
                client.set(self.prefix + key, data)
        except Exception:
            logger.warning("Redis set failed", exc_info=True)

    async def delete(self, key: str) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            client.delete(self.prefix + key)
        except Exception:
            logger.warning("Redis delete failed", exc_info=True)

    async def clear(self) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            keys = client.keys(self.prefix + "*")
            if keys:
                client.delete(*keys)
        except Exception:
            logger.warning("Redis clear failed", exc_info=True)

    async def stats(self) -> Dict[str, Any]:
        client = self._get_client()
        if client is None:
            return {"backend": "redis", "available": False}
        try:
            keys = client.keys(self.prefix + "*")
            return {"backend": "redis", "available": True, "entries": len(keys)}
        except Exception:
            return {"backend": "redis", "available": False}


class FilesystemBackend:
    """Filesystem-based cache backend for large responses and offline use."""

    def __init__(self, cache_dir: Optional[str] = None):
        from pathlib import Path
        if cache_dir is None:
            cache_dir = str(Path.home() / ".agentu" / "cache_fs")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str):
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.json"

    async def get(self, key: str) -> Optional[dict]:
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if data.get("expires_at") and time.time() > data["expires_at"]:
                path.unlink(missing_ok=True)
                return None
            return data["value"]
        except Exception:
            return None

    async def set(self, key: str, value: dict, ttl: Optional[int] = None) -> None:
        path = self._key_path(key)
        now = time.time()
        data = {
            "value": value,
            "created_at": now,
            "expires_at": now + ttl if ttl is not None else None,
        }
        try:
            path.write_text(json.dumps(data))
        except Exception:
            logger.warning("Filesystem cache set failed", exc_info=True)

    async def delete(self, key: str) -> None:
        path = self._key_path(key)
        path.unlink(missing_ok=True)

    async def clear(self) -> None:
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def stats(self) -> Dict[str, Any]:
        entries = len(list(self.cache_dir.glob("*.json")))
        return {"backend": "filesystem", "entries": entries, "cache_dir": str(self.cache_dir)}
