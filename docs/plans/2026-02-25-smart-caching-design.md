# Smart Caching System Design

## Problem

agentu's current caching is exact-match only (SHA256 hash of prompt + model + temperature). This misses cache opportunities when prompts are semantically similar, offers no tiered storage, and provides no offline support.

## Goals

- Reduce LLM API costs by matching semantically similar prompts
- Reduce latency via in-memory tier for hot data
- Support offline/limited connectivity via background cache sync
- Preserve full backward compatibility with existing `with_cache()` API

## Approach: Layered Cache with Semantic Index

Keep the current `cache.py` as the foundation, add a semantic index layer on top with pluggable storage backends.

## Architecture

### Lookup Flow

```
cache.get(prompt, model, temperature)
    |
    +-> Exact match (SHA256) in Memory tier -> HIT -> return
    |
    +-> Exact match in SQLite tier -> HIT -> promote to Memory -> return
    |
    +-> Semantic search (if enabled, threshold >= 0.95) -> HIT -> return
    |
    +-> MISS -> call LLM -> store in all tiers -> compute embedding -> index
```

Exact match is always tried first (zero latency cost). Semantic search only runs on exact-match miss.

### New Modules

- `src/agentu/cache_semantic.py` - Semantic index and similarity matching
- `src/agentu/cache_storage.py` - Pluggable storage backends (Memory, SQLite, Redis, Filesystem)
- `src/agentu/cache_sync.py` - Background sync daemon
- `src/agentu/cache_embeddings.py` - Configurable embedding providers

### Existing Module Changes

- `src/agentu/cache.py` - Refactor to use new storage backends, add semantic lookup as optional layer

## Storage Backends

Each backend implements a common protocol:

```python
class CacheStorageBackend(Protocol):
    async def get(self, key: str) -> Optional[dict]: ...
    async def set(self, key: str, value: dict, ttl: Optional[int] = None): ...
    async def delete(self, key: str): ...
    async def clear(self): ...
    async def stats(self) -> dict: ...
```

| Backend | Best For | Eviction | Persistence |
|---------|----------|----------|-------------|
| MemoryBackend | Hot data, fast lookups | LRU with configurable max size (default 1000) | No |
| SQLiteBackend | Current default, local persistence | TTL-based (existing behavior) | Yes |
| RedisBackend | Shared cache across processes | TTL + Redis eviction policies | Yes |
| FilesystemBackend | Large responses, offline use | TTL + disk space limits | Yes |

Tiered lookup checks backends in order. On a hit in a lower tier, the entry is promoted to higher tiers.

Redis and Filesystem only activate if the relevant dependency is installed (`redis` or `aiofiles`).

## Embedding Providers

Two built-in providers, users can add custom:

```python
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    def dimension(self) -> int: ...
```

| Provider | How it works | Pros | Cons |
|----------|-------------|------|------|
| LocalEmbedding | Uses `sentence-transformers` (all-MiniLM-L6-v2 by default) | Free, fast, offline, ~80MB | Requires torch/numpy |
| APIEmbedding | Uses Ollama or OpenAI-compatible embedding endpoint | No heavy deps, consistent with LLM provider | Adds latency + cost per lookup |

Embeddings are stored in a SQLite table alongside cache keys. Similarity search uses cosine similarity computed in Python. For the conservative threshold (0.95+), brute-force scan is sufficient at typical cache sizes (<100K entries).

Graceful degradation: if `sentence-transformers` isn't installed and provider is "local", semantic matching is silently disabled and a warning is logged.

## Background Sync

- Async background task runs on a configurable interval (default: 5 minutes)
- Exports cache entries (keys, values, embeddings, metadata) to a snapshot file
- On agent startup, imports any existing snapshot to pre-warm the cache
- Sync format: SQLite database file (portable, atomic writes)
- Only writes if cache has changed since last sync (dirty flag)
- Conflict resolution: last-write-wins
- Gracefully stops on agent shutdown via `async with agent:` context manager
- Emits events (`cache_sync_start`, `cache_sync_complete`, `cache_sync_error`) through the observer system

## API Surface

### Backward Compatible

```python
# Current behavior - unchanged
agent = Agent("bot").with_cache(ttl=3600)
```

### Preset-Based

```python
agent = Agent("bot").with_cache(preset="smart")        # memory + sqlite + local semantic
agent = Agent("bot").with_cache(preset="distributed")   # memory + redis + api semantic
agent = Agent("bot").with_cache(preset="offline")       # memory + sqlite + filesystem + local semantic + sync
```

### Presets

| Preset | Backends | Semantic | Sync | Use Case |
|--------|----------|----------|------|----------|
| "basic" | memory + sqlite | off | off | Current behavior (default) |
| "smart" | memory + sqlite | local | off | Cost reduction + latency |
| "offline" | memory + sqlite + filesystem | local | on | Works without connectivity |
| "distributed" | memory + redis | api | off | Multi-process / shared cache |

### Override

```python
agent = Agent("bot").with_cache(
    preset="smart",
    similarity_threshold=0.90,
    ttl=7200,
)
```

### Extended Stats

```python
stats = await agent.cache_stats()
# {
#   "exact_hits": 150,
#   "semantic_hits": 23,
#   "misses": 40,
#   "hit_rate": 0.81,
#   "tier_hits": {"memory": 120, "sqlite": 50, "redis": 3},
#   "last_sync": "2026-02-24T10:30:00Z",
#   "embedding_count": 213,
# }
```

### Optional Dependencies

- `pip install agentu` - exact-match + SQLite (current behavior)
- `pip install agentu[semantic]` - adds `sentence-transformers` for local embeddings
- `pip install agentu[redis]` - adds `redis` for Redis backend
- `pip install agentu[cache-all]` - all caching extras

## Error Handling

Cache failures never break the agent. Every failure degrades gracefully.

| Failure | Behavior |
|---------|----------|
| Redis unavailable | Skip tier, log warning, continue with remaining backends |
| Embedding provider fails | Fall back to exact-match only, log warning |
| Sync write fails | Log error via observer, retry on next interval |
| Corrupt snapshot on startup | Ignore snapshot, log warning, start with empty cache |
| `sentence-transformers` not installed | Semantic matching silently disabled, exact-match works |

## Testing

- Unit tests per module - each new module gets its own test file
- Mock embeddings for fast tests - fake embedding provider returns deterministic vectors, no torch in CI
- Integration test - end-to-end test exercising full lookup flow (exact miss -> semantic hit -> tier promotion)
- Backward compatibility - existing `test_cache.py` tests must pass unchanged
