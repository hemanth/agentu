"""Pluggable cache storage backends for agentu."""

import time
import json
import sqlite3
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
