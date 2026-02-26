# Smart Caching System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semantic caching with tiered storage, configurable embeddings, and background sync to agentu's cache system.

**Architecture:** Layered cache with exact-match fast path, optional semantic index, pluggable storage backends (Memory, SQLite, Redis, Filesystem), and background sync daemon. Backward compatible with existing `cache=True` constructor API. New `with_cache(preset=...)` fluent method.

**Tech Stack:** Python 3.9+, aiohttp, sqlite3, sentence-transformers (optional), redis (optional)

**Design doc:** `docs/plans/2026-02-25-smart-caching-design.md`

---

### Task 1: CacheStorageBackend Protocol and MemoryBackend

**Files:**
- Create: `src/agentu/cache_storage.py`
- Test: `tests/test_cache_storage.py`

**Step 1: Write failing tests**

```python
# tests/test_cache_storage.py
"""Tests for pluggable cache storage backends."""

import pytest
import pytest_asyncio

from agentu.cache_storage_backends import CacheStorageBackend, MemoryBackend


class TestMemoryBackend:
    @pytest_asyncio.fixture
    async def backend(self):
        return MemoryBackend(max_size=5)

    @pytest.mark.asyncio
    async def test_get_miss(self, backend):
        result = await backend.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, backend):
        await backend.set("key1", {"response": "hello"})
        result = await backend.get("key1")
        assert result == {"response": "hello"}

    @pytest.mark.asyncio
    async def test_delete(self, backend):
        await backend.set("key1", {"response": "hello"})
        await backend.delete("key1")
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear(self, backend):
        await backend.set("k1", {"r": "1"})
        await backend.set("k2", {"r": "2"})
        await backend.clear()
        assert await backend.get("k1") is None
        assert await backend.get("k2") is None

    @pytest.mark.asyncio
    async def test_stats(self, backend):
        await backend.set("k1", {"r": "1"})
        stats = await backend.stats()
        assert stats["entries"] == 1
        assert stats["backend"] == "memory"

    @pytest.mark.asyncio
    async def test_lru_eviction(self, backend):
        for i in range(6):
            await backend.set(f"key{i}", {"r": str(i)})
        # key0 should be evicted (max_size=5)
        assert await backend.get("key0") is None
        assert await backend.get("key5") == {"r": "5"}

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, backend):
        await backend.set("key1", {"r": "1"}, ttl=0)
        import asyncio
        await asyncio.sleep(0.1)
        assert await backend.get("key1") is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentu.cache_storage_backends'`

**Step 3: Write minimal implementation**

```python
# src/agentu/cache_storage_backends.py
"""Pluggable cache storage backends for agentu."""

import time
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
        return self._store[key]

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache_storage.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agentu/cache_storage_backends.py tests/test_cache_storage.py
git commit -m "feat(cache): add CacheStorageBackend protocol and MemoryBackend"
```

---

### Task 2: SQLiteBackend

**Files:**
- Modify: `src/agentu/cache_storage_backends.py`
- Test: `tests/test_cache_storage.py` (append)

**Step 1: Write failing tests**

Append to `tests/test_cache_storage.py`:

```python
from agentu.cache_storage_backends import SQLiteBackend


class TestSQLiteBackend:
    @pytest_asyncio.fixture
    async def backend(self, tmp_path):
        return SQLiteBackend(db_path=str(tmp_path / "test.db"))

    @pytest.mark.asyncio
    async def test_get_miss(self, backend):
        assert await backend.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, backend):
        await backend.set("key1", {"response": "hello"})
        result = await backend.get("key1")
        assert result == {"response": "hello"}

    @pytest.mark.asyncio
    async def test_delete(self, backend):
        await backend.set("key1", {"response": "hello"})
        await backend.delete("key1")
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear(self, backend):
        await backend.set("k1", {"r": "1"})
        await backend.set("k2", {"r": "2"})
        await backend.clear()
        assert await backend.get("k1") is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, backend):
        import asyncio
        await backend.set("key1", {"r": "1"}, ttl=0)
        await asyncio.sleep(0.1)
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path):
        db = str(tmp_path / "persist.db")
        b1 = SQLiteBackend(db_path=db)
        await b1.set("key1", {"r": "1"})
        b2 = SQLiteBackend(db_path=db)
        assert await b2.get("key1") == {"r": "1"}

    @pytest.mark.asyncio
    async def test_stats(self, backend):
        await backend.set("k1", {"r": "1"})
        stats = await backend.stats()
        assert stats["backend"] == "sqlite"
        assert stats["entries"] == 1
```

**Step 2: Run tests to verify new tests fail**

Run: `pytest tests/test_cache_storage.py::TestSQLiteBackend -v`
Expected: FAIL with `ImportError`

**Step 3: Write implementation**

Add to `src/agentu/cache_storage_backends.py`:

```python
import sqlite3
import json


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
        conn.close()

    async def get(self, key: str) -> Optional[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT value FROM cache_store WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
            (key, time.time())
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None

    async def set(self, key: str, value: dict, ttl: Optional[int] = None) -> None:
        now = time.time()
        expires_at = now + ttl if ttl is not None else None
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO cache_store (key, value, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (key, json.dumps(value), now, expires_at)
        )
        conn.commit()
        conn.close()

    async def delete(self, key: str) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM cache_store WHERE key = ?", (key,))
        conn.commit()
        conn.close()

    async def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM cache_store")
        conn.commit()
        conn.close()

    async def stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM cache_store WHERE expires_at IS NULL OR expires_at > ?",
            (time.time(),)
        )
        count = cursor.fetchone()[0]
        conn.close()
        return {"backend": "sqlite", "entries": count, "db_path": self.db_path}
```

**Step 4: Run tests**

Run: `pytest tests/test_cache_storage.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agentu/cache_storage_backends.py tests/test_cache_storage.py
git commit -m "feat(cache): add SQLiteBackend storage"
```

---

### Task 3: RedisBackend and FilesystemBackend

**Files:**
- Modify: `src/agentu/cache_storage_backends.py`
- Test: `tests/test_cache_storage.py` (append)

**Step 1: Write failing tests**

Append to `tests/test_cache_storage.py`:

```python
from agentu.cache_storage_backends import RedisBackend, FilesystemBackend


class TestRedisBackend:
    """Tests use a mock redis — no real Redis needed."""

    @pytest.mark.asyncio
    async def test_unavailable_gracefully(self):
        backend = RedisBackend(url="redis://localhost:59999")
        # Should not raise, just return None
        result = await backend.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_stats_when_unavailable(self):
        backend = RedisBackend(url="redis://localhost:59999")
        stats = await backend.stats()
        assert stats["backend"] == "redis"
        assert stats["available"] is False


class TestFilesystemBackend:
    @pytest_asyncio.fixture
    async def backend(self, tmp_path):
        return FilesystemBackend(cache_dir=str(tmp_path / "fs_cache"))

    @pytest.mark.asyncio
    async def test_get_miss(self, backend):
        assert await backend.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, backend):
        await backend.set("key1", {"response": "hello"})
        result = await backend.get("key1")
        assert result == {"response": "hello"}

    @pytest.mark.asyncio
    async def test_delete(self, backend):
        await backend.set("key1", {"r": "1"})
        await backend.delete("key1")
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear(self, backend):
        await backend.set("k1", {"r": "1"})
        await backend.set("k2", {"r": "2"})
        await backend.clear()
        assert await backend.get("k1") is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, backend):
        import asyncio
        await backend.set("key1", {"r": "1"}, ttl=0)
        await asyncio.sleep(0.1)
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_stats(self, backend):
        await backend.set("k1", {"r": "1"})
        stats = await backend.stats()
        assert stats["backend"] == "filesystem"
        assert stats["entries"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache_storage.py::TestRedisBackend tests/test_cache_storage.py::TestFilesystemBackend -v`
Expected: FAIL with `ImportError`

**Step 3: Write implementation**

Add to `src/agentu/cache_storage_backends.py`:

```python
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

    def _key_path(self, key: str) -> 'Path':
        from pathlib import Path
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
        data = {
            "value": value,
            "created_at": time.time(),
            "expires_at": time.time() + ttl if ttl is not None else None,
        }
        path.write_text(json.dumps(data))

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
```

Also add `import hashlib` to the top of the file.

**Step 4: Run tests**

Run: `pytest tests/test_cache_storage.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agentu/cache_storage_backends.py tests/test_cache_storage.py
git commit -m "feat(cache): add RedisBackend and FilesystemBackend"
```

---

### Task 4: EmbeddingProvider Protocol and Implementations

**Files:**
- Create: `src/agentu/cache_embeddings.py`
- Create: `tests/test_cache_embeddings.py`

**Step 1: Write failing tests**

```python
# tests/test_cache_embeddings.py
"""Tests for cache embedding providers."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agentu.cache_embeddings import (
    EmbeddingProvider, LocalEmbedding, APIEmbedding, FakeEmbedding
)


class TestFakeEmbedding:
    """FakeEmbedding is for testing — deterministic vectors."""

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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/agentu/cache_embeddings.py
"""Configurable embedding providers for semantic cache matching."""

import hashlib
import logging
import math
from typing import List, Optional, Protocol, runtime_checkable

import aiohttp

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> List[float]: ...
    def dimension(self) -> int: ...


class FakeEmbedding:
    """Deterministic embeddings for testing. NOT for production."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    async def embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode()).hexdigest()
        values = []
        for i in range(0, min(len(h), self._dimension * 2), 2):
            values.append(int(h[i:i+2], 16) / 255.0)
        while len(values) < self._dimension:
            values.append(0.0)
        # Normalize
        norm = math.sqrt(sum(v * v for v in values))
        if norm > 0:
            values = [v / norm for v in values]
        return values[:self._dimension]

    def dimension(self) -> int:
        return self._dimension


class LocalEmbedding:
    """Local embeddings via sentence-transformers. Requires `pip install sentence-transformers`."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384  # default for MiniLM

    def available(self) -> bool:
        try:
            import sentence_transformers
            return True
        except (ImportError, TypeError):
            return False

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> List[float]:
        if not self.available():
            raise RuntimeError("sentence-transformers not installed")
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def dimension(self) -> int:
        return self._dimension


class APIEmbedding:
    """Embeddings via OpenAI-compatible API (Ollama, OpenAI, etc.)."""

    def __init__(self, api_base: str = "http://localhost:11434/v1",
                 model: str = "nomic-embed-text", api_key: Optional[str] = None):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._dimension = 768  # common default

    async def embed(self, text: str) -> List[float]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"input": text, "model": self.model}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/embeddings", json=payload, headers=headers
            ) as resp:
                data = await resp.json()
                embedding = data["data"][0]["embedding"]
                self._dimension = len(embedding)
                return embedding

    def dimension(self) -> int:
        return self._dimension


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

**Step 4: Run tests**

Run: `pytest tests/test_cache_embeddings.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agentu/cache_embeddings.py tests/test_cache_embeddings.py
git commit -m "feat(cache): add embedding providers (Local, API, Fake)"
```

---

### Task 5: SemanticIndex

**Files:**
- Create: `src/agentu/cache_semantic.py`
- Create: `tests/test_cache_semantic.py`

**Step 1: Write failing tests**

```python
# tests/test_cache_semantic.py
"""Tests for semantic cache index."""

import pytest
import pytest_asyncio

from agentu.cache_semantic import SemanticIndex
from agentu.cache_embeddings import FakeEmbedding


class TestSemanticIndex:
    @pytest_asyncio.fixture
    async def index(self, tmp_path):
        provider = FakeEmbedding(dimension=32)
        return SemanticIndex(
            embedding_provider=provider,
            db_path=str(tmp_path / "semantic.db"),
            threshold=0.95,
        )

    @pytest.mark.asyncio
    async def test_no_match_on_empty(self, index):
        result = await index.search("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_exact_text_matches(self, index):
        await index.add("hello world", "cache_key_1")
        result = await index.search("hello world")
        assert result == "cache_key_1"

    @pytest.mark.asyncio
    async def test_different_text_no_match(self, index):
        await index.add("hello world", "cache_key_1")
        result = await index.search("something completely different and unrelated")
        assert result is None

    @pytest.mark.asyncio
    async def test_remove(self, index):
        await index.add("hello world", "cache_key_1")
        await index.remove("cache_key_1")
        result = await index.search("hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear(self, index):
        await index.add("hello", "k1")
        await index.add("world", "k2")
        await index.clear()
        assert await index.search("hello") is None

    @pytest.mark.asyncio
    async def test_stats(self, index):
        await index.add("hello", "k1")
        stats = await index.stats()
        assert stats["embedding_count"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache_semantic.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/agentu/cache_semantic.py
"""Semantic index for cache similarity matching."""

import sqlite3
import json
import logging
from typing import Optional, Dict, Any, List

from .cache_embeddings import EmbeddingProvider, cosine_similarity

logger = logging.getLogger(__name__)


class SemanticIndex:
    """Stores embeddings alongside cache keys for semantic lookup."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db_path: Optional[str] = None,
        threshold: float = 0.95,
    ):
        self.provider = embedding_provider
        self.threshold = threshold

        if db_path is None:
            from pathlib import Path
            cache_dir = Path.home() / ".agentu"
            cache_dir.mkdir(exist_ok=True)
            db_path = str(cache_dir / "semantic_index.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                text_hash TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    async def add(self, text: str, cache_key: str) -> None:
        embedding = await self.provider.embed(text)
        import hashlib
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (cache_key, text_hash, embedding) VALUES (?, ?, ?)",
            (cache_key, text_hash, json.dumps(embedding))
        )
        conn.commit()
        conn.close()

    async def search(self, text: str) -> Optional[str]:
        query_embedding = await self.provider.embed(text)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT cache_key, embedding FROM embeddings")
        rows = cursor.fetchall()
        conn.close()

        best_key = None
        best_score = -1.0

        for cache_key, emb_json in rows:
            stored_embedding = json.loads(emb_json)
            score = cosine_similarity(query_embedding, stored_embedding)
            if score >= self.threshold and score > best_score:
                best_score = score
                best_key = cache_key

        return best_key

    async def remove(self, cache_key: str) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM embeddings WHERE cache_key = ?", (cache_key,))
        conn.commit()
        conn.close()

    async def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()

    async def stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        conn.close()
        return {
            "embedding_count": count,
            "threshold": self.threshold,
            "provider": type(self.provider).__name__,
        }
```

**Step 4: Run tests**

Run: `pytest tests/test_cache_semantic.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agentu/cache_semantic.py tests/test_cache_semantic.py
git commit -m "feat(cache): add SemanticIndex for similarity matching"
```

---

### Task 6: Background Sync

**Files:**
- Create: `src/agentu/cache_sync.py`
- Create: `tests/test_cache_sync.py`

**Step 1: Write failing tests**

```python
# tests/test_cache_sync.py
"""Tests for cache background sync."""

import pytest
import pytest_asyncio
import asyncio

from agentu.cache_sync import CacheSync
from agentu.cache_storage_backends import SQLiteBackend


class TestCacheSync:
    @pytest_asyncio.fixture
    async def setup(self, tmp_path):
        source = SQLiteBackend(db_path=str(tmp_path / "source.db"))
        sync_path = str(tmp_path / "sync")
        sync = CacheSync(source_backend=source, sync_path=sync_path, sync_interval=1)
        return source, sync, sync_path

    @pytest.mark.asyncio
    async def test_snapshot_created(self, setup):
        source, sync, sync_path = setup
        await source.set("k1", {"r": "1"})
        sync.mark_dirty()
        await sync.sync_once()
        from pathlib import Path
        snapshot = Path(sync_path) / "cache_snapshot.db"
        assert snapshot.exists()

    @pytest.mark.asyncio
    async def test_restore_from_snapshot(self, setup, tmp_path):
        source, sync, sync_path = setup
        await source.set("k1", {"r": "1"})
        sync.mark_dirty()
        await sync.sync_once()

        # New backend, restore from snapshot
        new_backend = SQLiteBackend(db_path=str(tmp_path / "restored.db"))
        restored = await sync.restore(new_backend)
        assert restored is True
        result = await new_backend.get("k1")
        assert result == {"r": "1"}

    @pytest.mark.asyncio
    async def test_no_sync_when_clean(self, setup):
        source, sync, sync_path = setup
        await source.set("k1", {"r": "1"})
        # Don't mark dirty
        await sync.sync_once()
        from pathlib import Path
        snapshot = Path(sync_path) / "cache_snapshot.db"
        assert not snapshot.exists()

    @pytest.mark.asyncio
    async def test_start_stop(self, setup):
        source, sync, sync_path = setup
        task = sync.start()
        assert sync.running is True
        await sync.stop()
        assert sync.running is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache_sync.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/agentu/cache_sync.py
"""Background cache sync daemon."""

import asyncio
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from .cache_storage_backends import SQLiteBackend

logger = logging.getLogger(__name__)


class CacheSync:
    """Periodically syncs cache to a snapshot location."""

    def __init__(
        self,
        source_backend: SQLiteBackend,
        sync_path: str,
        sync_interval: int = 300,
    ):
        self.source = source_backend
        self.sync_path = Path(sync_path)
        self.sync_path.mkdir(parents=True, exist_ok=True)
        self.sync_interval = sync_interval
        self._dirty = False
        self._task: Optional[asyncio.Task] = None
        self.running = False

    def mark_dirty(self):
        self._dirty = True

    @property
    def snapshot_path(self) -> Path:
        return self.sync_path / "cache_snapshot.db"

    async def sync_once(self) -> bool:
        if not self._dirty:
            return False
        try:
            shutil.copy2(self.source.db_path, str(self.snapshot_path))
            self._dirty = False
            logger.info("Cache synced to %s", self.snapshot_path)
            return True
        except Exception:
            logger.error("Cache sync failed", exc_info=True)
            return False

    async def restore(self, target_backend: SQLiteBackend) -> bool:
        if not self.snapshot_path.exists():
            return False
        try:
            src_conn = sqlite3.connect(str(self.snapshot_path))
            cursor = src_conn.execute("SELECT key, value, created_at, expires_at FROM cache_store")
            rows = cursor.fetchall()
            src_conn.close()

            dst_conn = sqlite3.connect(target_backend.db_path)
            for key, value, created_at, expires_at in rows:
                dst_conn.execute(
                    "INSERT OR REPLACE INTO cache_store (key, value, created_at, expires_at) VALUES (?, ?, ?, ?)",
                    (key, value, created_at, expires_at)
                )
            dst_conn.commit()
            dst_conn.close()
            logger.info("Restored %d entries from snapshot", len(rows))
            return True
        except Exception:
            logger.error("Cache restore failed", exc_info=True)
            return False

    def start(self) -> asyncio.Task:
        self.running = True
        self._task = asyncio.create_task(self._loop())
        return self._task

    async def stop(self):
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Final sync
        await self.sync_once()

    async def _loop(self):
        try:
            while self.running:
                await asyncio.sleep(self.sync_interval)
                await self.sync_once()
        except asyncio.CancelledError:
            pass
```

**Step 4: Run tests**

Run: `pytest tests/test_cache_sync.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agentu/cache_sync.py tests/test_cache_sync.py
git commit -m "feat(cache): add background sync daemon"
```

---

### Task 7: TieredCache — Unified Cache Orchestrator

**Files:**
- Create: `src/agentu/cache_tiered.py`
- Create: `tests/test_cache_tiered.py`

**Step 1: Write failing tests**

```python
# tests/test_cache_tiered.py
"""Tests for TieredCache orchestrator."""

import pytest
import pytest_asyncio

from agentu.cache_tiered import TieredCache
from agentu.cache_storage_backends import MemoryBackend, SQLiteBackend
from agentu.cache_embeddings import FakeEmbedding
from agentu.cache_semantic import SemanticIndex


class TestTieredCacheExactMatch:
    @pytest_asyncio.fixture
    async def cache(self, tmp_path):
        backends = [MemoryBackend(max_size=100), SQLiteBackend(db_path=str(tmp_path / "t.db"))]
        return TieredCache(backends=backends, ttl=3600)

    @pytest.mark.asyncio
    async def test_miss_on_empty(self, cache):
        result = await cache.get("prompt", "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        await cache.set("prompt", "model", "response")
        result = await cache.get("prompt", "model")
        assert result == "response"

    @pytest.mark.asyncio
    async def test_tier_promotion(self, tmp_path):
        mem = MemoryBackend(max_size=100)
        sql = SQLiteBackend(db_path=str(tmp_path / "promo.db"))
        cache = TieredCache(backends=[mem, sql], ttl=3600)

        # Write directly to sqlite only
        key = cache._make_key("prompt", "model")
        await sql.set(key, {"response": "hello"}, ttl=3600)

        # Get should promote to memory
        result = await cache.get("prompt", "model")
        assert result == "hello"
        mem_result = await mem.get(key)
        assert mem_result is not None

    @pytest.mark.asyncio
    async def test_stats(self, cache):
        await cache.set("p", "m", "r")
        await cache.get("p", "m")  # hit
        await cache.get("miss", "m")  # miss
        stats = await cache.get_stats()
        assert stats["exact_hits"] == 1
        assert stats["misses"] == 1
        assert "tier_hits" in stats


class TestTieredCacheWithSemantic:
    @pytest_asyncio.fixture
    async def cache(self, tmp_path):
        backends = [MemoryBackend(max_size=100)]
        provider = FakeEmbedding(dimension=32)
        index = SemanticIndex(
            embedding_provider=provider,
            db_path=str(tmp_path / "sem.db"),
            threshold=0.95,
        )
        return TieredCache(backends=backends, ttl=3600, semantic_index=index)

    @pytest.mark.asyncio
    async def test_exact_match_preferred_over_semantic(self, cache):
        await cache.set("hello world", "model", "exact response")
        result = await cache.get("hello world", "model")
        assert result == "exact response"
        stats = await cache.get_stats()
        assert stats["exact_hits"] == 1
        assert stats["semantic_hits"] == 0

    @pytest.mark.asyncio
    async def test_semantic_match_on_exact_miss(self, cache):
        await cache.set("hello world", "model", "cached response")
        # Same text should match semantically
        result = await cache.get("hello world", "model")
        assert result == "cached response"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache_tiered.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/agentu/cache_tiered.py
"""TieredCache — unified cache orchestrator with optional semantic matching."""

import hashlib
import json
import logging
from typing import Optional, Dict, Any, List

from .cache_storage_backends import CacheStorageBackend
from .cache_semantic import SemanticIndex

logger = logging.getLogger(__name__)


class TieredCache:
    """Orchestrates tiered storage backends with optional semantic index."""

    def __init__(
        self,
        backends: List[CacheStorageBackend],
        ttl: int = 3600,
        semantic_index: Optional[SemanticIndex] = None,
    ):
        self.backends = backends
        self.ttl = ttl
        self.semantic_index = semantic_index
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._tier_hits: Dict[str, int] = {}
        self._dirty = False

    def _make_key(self, prompt: str, model: str, **kwargs) -> str:
        key_data = {"prompt": prompt, "model": model, "temperature": kwargs.get("temperature")}
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    async def get(self, prompt: str, model: str, **kwargs) -> Optional[str]:
        cache_key = self._make_key(prompt, model, **kwargs)

        # Try exact match across tiers
        for i, backend in enumerate(self.backends):
            result = await backend.get(cache_key)
            if result is not None:
                self._exact_hits += 1
                backend_stats = await backend.stats()
                tier_name = backend_stats.get("backend", f"tier_{i}")
                self._tier_hits[tier_name] = self._tier_hits.get(tier_name, 0) + 1
                # Promote to higher tiers
                for j in range(i):
                    await self.backends[j].set(cache_key, result, ttl=self.ttl)
                return result.get("response")

        # Try semantic match
        if self.semantic_index is not None:
            matched_key = await self.semantic_index.search(prompt)
            if matched_key is not None:
                for backend in self.backends:
                    result = await backend.get(matched_key)
                    if result is not None:
                        self._semantic_hits += 1
                        return result.get("response")

        self._misses += 1
        return None

    async def set(self, prompt: str, model: str, response: str, **kwargs) -> None:
        cache_key = self._make_key(prompt, model, **kwargs)
        value = {"response": response, "model": model, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()}

        for backend in self.backends:
            await backend.set(cache_key, value, ttl=self.ttl)

        if self.semantic_index is not None:
            await self.semantic_index.add(prompt, cache_key)

        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def clear_dirty(self):
        self._dirty = False

    async def clear(self) -> None:
        for backend in self.backends:
            await backend.clear()
        if self.semantic_index:
            await self.semantic_index.clear()
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._tier_hits.clear()

    async def invalidate(self, prompt: str, model: str, **kwargs) -> None:
        cache_key = self._make_key(prompt, model, **kwargs)
        for backend in self.backends:
            await backend.delete(cache_key)
        if self.semantic_index:
            await self.semantic_index.remove(cache_key)

    async def get_stats(self) -> Dict[str, Any]:
        total = self._exact_hits + self._semantic_hits + self._misses
        stats = {
            "exact_hits": self._exact_hits,
            "semantic_hits": self._semantic_hits,
            "misses": self._misses,
            "hit_rate": round((self._exact_hits + self._semantic_hits) / total, 3) if total > 0 else 0.0,
            "tier_hits": dict(self._tier_hits),
        }
        if self.semantic_index:
            sem_stats = await self.semantic_index.stats()
            stats["embedding_count"] = sem_stats["embedding_count"]
        return stats
```

**Step 4: Run tests**

Run: `pytest tests/test_cache_tiered.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agentu/cache_tiered.py tests/test_cache_tiered.py
git commit -m "feat(cache): add TieredCache orchestrator"
```

---

### Task 8: Agent Integration — `with_cache()` Method and Presets

**Files:**
- Modify: `src/agentu/agent.py:85-147` (constructor + add method)
- Modify: `src/agentu/agent.py:576-618` (infer cache usage)
- Modify: `src/agentu/agent.py:724-725` (stream cache usage)
- Create: `tests/test_cache_integration.py`

**Step 1: Write failing tests**

```python
# tests/test_cache_integration.py
"""Tests for Agent.with_cache() integration."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from agentu.agent import Agent
from agentu.cache_tiered import TieredCache


class TestWithCacheMethod:
    def test_with_cache_returns_self(self):
        agent = Agent("test", cache=False)
        result = agent.with_cache()
        assert result is agent

    def test_with_cache_basic_preset(self):
        agent = Agent("test", cache=False).with_cache(preset="basic")
        assert agent.cache is not None
        assert isinstance(agent.cache, TieredCache)

    def test_with_cache_smart_preset(self):
        agent = Agent("test", cache=False).with_cache(preset="smart")
        assert agent.cache is not None
        assert agent.cache.semantic_index is not None

    def test_with_cache_no_preset_defaults_to_basic(self):
        agent = Agent("test", cache=False).with_cache()
        assert agent.cache is not None
        assert isinstance(agent.cache, TieredCache)

    def test_with_cache_custom_ttl(self):
        agent = Agent("test", cache=False).with_cache(ttl=7200)
        assert agent.cache is not None
        assert agent.cache.ttl == 7200

    def test_backward_compat_constructor_cache(self):
        agent = Agent("test", cache=True, cache_ttl=1800)
        assert agent.cache is not None


class TestWithCachePresets:
    def test_offline_preset_has_sync(self):
        agent = Agent("test", cache=False).with_cache(preset="offline")
        assert agent._cache_sync is not None

    def test_distributed_preset(self):
        # Redis won't connect, but the preset should configure without error
        agent = Agent("test", cache=False).with_cache(
            preset="distributed",
            redis_url="redis://localhost:59999"
        )
        assert agent.cache is not None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache_integration.py -v`
Expected: FAIL — `Agent` has no `with_cache` method

**Step 3: Write implementation**

Add to `src/agentu/agent.py` after line 147 (after `self._pending_mcp_config` assignment):

```python
    # Cache sync reference
    self._cache_sync = None
```

Add new method after `with_tools` (around line 260):

```python
def with_cache(self, preset: Optional[str] = None, ttl: int = 3600,
               similarity_threshold: float = 0.95, embedding_provider: str = "local",
               embedding_model: Optional[str] = None, redis_url: str = "redis://localhost:6379",
               sync_enabled: bool = False, sync_path: Optional[str] = None,
               sync_interval: int = 300) -> 'Agent':
    """Configure smart caching with presets.

    Args:
        preset: "basic", "smart", "offline", or "distributed"
        ttl: Cache TTL in seconds
        similarity_threshold: Cosine similarity threshold for semantic matching
        embedding_provider: "local" or "api"
        embedding_model: Model name for embeddings
        redis_url: Redis URL for distributed preset
        sync_enabled: Enable background sync
        sync_path: Path for sync snapshots
        sync_interval: Seconds between syncs
    """
    from .cache_storage_backends import MemoryBackend, SQLiteBackend, RedisBackend, FilesystemBackend
    from .cache_tiered import TieredCache

    if preset is None:
        preset = "basic"

    # Build backends based on preset
    backends = []
    semantic_index = None

    if preset == "basic":
        backends = [MemoryBackend(), SQLiteBackend()]
    elif preset == "smart":
        backends = [MemoryBackend(), SQLiteBackend()]
        semantic_index = self._build_semantic_index(
            embedding_provider, embedding_model, similarity_threshold
        )
    elif preset == "offline":
        backends = [MemoryBackend(), SQLiteBackend(), FilesystemBackend()]
        semantic_index = self._build_semantic_index(
            embedding_provider, embedding_model, similarity_threshold
        )
        sync_enabled = True
    elif preset == "distributed":
        backends = [MemoryBackend(), RedisBackend(url=redis_url)]
        semantic_index = self._build_semantic_index(
            "api", embedding_model, similarity_threshold
        )

    self.cache = TieredCache(backends=backends, ttl=ttl, semantic_index=semantic_index)
    self.cache_enabled = True

    if sync_enabled:
        from .cache_sync import CacheSync
        sqlite_backend = next((b for b in backends if isinstance(b, SQLiteBackend)), None)
        if sqlite_backend:
            if sync_path is None:
                from pathlib import Path
                sync_path = str(Path.home() / ".agentu" / "cache_sync")
            self._cache_sync = CacheSync(
                source_backend=sqlite_backend,
                sync_path=sync_path,
                sync_interval=sync_interval,
            )

    return self

def _build_semantic_index(self, provider_type: str, model: Optional[str],
                          threshold: float):
    from .cache_semantic import SemanticIndex
    from .cache_embeddings import LocalEmbedding, APIEmbedding

    if provider_type == "local":
        provider = LocalEmbedding(model_name=model or "all-MiniLM-L6-v2")
        if not provider.available():
            logger.warning("sentence-transformers not installed, semantic caching disabled")
            return None
    else:
        provider = APIEmbedding(
            api_base=self.api_base, model=model or "nomic-embed-text", api_key=self.api_key
        )

    return SemanticIndex(embedding_provider=provider, threshold=threshold)
```

Update the cache usage in `infer()` (around line 576-618) to work with TieredCache:

Replace the existing cache get/set calls to use async `await self.cache.get(...)` and `await self.cache.set(...)` — the TieredCache has the same signature as the old `LLMCache` but async.

Update `stream()` cache usage (around line 724-725) similarly.

Update backward-compat constructor (line 147):

```python
# Initialize cache if enabled (backward compatibility)
self.cache_enabled = cache
if cache:
    from .cache_tiered import TieredCache
    from .cache_storage_backends import MemoryBackend, SQLiteBackend
    self.cache = TieredCache(
        backends=[MemoryBackend(), SQLiteBackend()],
        ttl=cache_ttl,
    )
else:
    self.cache = None
```

**Step 4: Run tests**

Run: `pytest tests/test_cache_integration.py tests/test_cache.py -v`
Expected: All PASS (both new and existing tests)

**Step 5: Commit**

```bash
git add src/agentu/agent.py tests/test_cache_integration.py
git commit -m "feat(cache): add with_cache() method with presets"
```

---

### Task 9: Exports, Optional Dependencies, and Final Verification

**Files:**
- Modify: `src/agentu/__init__.py`
- Modify: `pyproject.toml`

**Step 1: Update exports**

Add to `src/agentu/__init__.py`:

```python
from .cache_storage_backends import CacheStorageBackend, MemoryBackend, SQLiteBackend, RedisBackend, FilesystemBackend
from .cache_embeddings import EmbeddingProvider, LocalEmbedding, APIEmbedding, cosine_similarity
from .cache_semantic import SemanticIndex
from .cache_tiered import TieredCache
from .cache_sync import CacheSync
```

Add these to `__all__` list.

**Step 2: Add optional dependencies to `pyproject.toml`**

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "httpx>=0.24.0",
]
semantic = [
    "sentence-transformers>=2.2.0",
]
redis = [
    "redis>=4.0.0",
]
cache-all = [
    "sentence-transformers>=2.2.0",
    "redis>=4.0.0",
]
```

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 4: Run existing cache tests specifically**

Run: `pytest tests/test_cache.py -v`
Expected: All PASS (backward compatibility verified)

**Step 5: Commit**

```bash
git add src/agentu/__init__.py pyproject.toml
git commit -m "feat(cache): add exports and optional dependencies for smart caching"
```

---

## Summary

| Task | What it builds | New files |
|------|---------------|-----------|
| 1 | Protocol + MemoryBackend | `cache_storage_backends.py`, `test_cache_storage.py` |
| 2 | SQLiteBackend | (appends to above) |
| 3 | RedisBackend + FilesystemBackend | (appends to above) |
| 4 | Embedding providers | `cache_embeddings.py`, `test_cache_embeddings.py` |
| 5 | SemanticIndex | `cache_semantic.py`, `test_cache_semantic.py` |
| 6 | Background sync | `cache_sync.py`, `test_cache_sync.py` |
| 7 | TieredCache orchestrator | `cache_tiered.py`, `test_cache_tiered.py` |
| 8 | Agent.with_cache() + presets | modifies `agent.py`, `test_cache_integration.py` |
| 9 | Exports + optional deps | modifies `__init__.py`, `pyproject.toml` |
