"""MemoryMixin – memory-related methods extracted from Agent."""

import asyncio
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


class MemoryMixin:
    """Mixin providing memory operations for the Agent class.

    Methods here assume they are mixed into an Agent instance that has
    ``memory_enabled``, ``memory``, and ``observer`` attributes.

    When the agent has a ``_vector_backend`` configured (via
    ``with_vectors()``), ``remember()`` and ``recall(semantic=True)``
    will use it for embedding-based storage and retrieval.
    """

    def _get_vector_backend_sync(self):
        """Resolve vector backend, lazily creating from DSN if needed.

        Returns the backend or None if not configured.
        LanceDBBackend.create() is synchronous, so this works in any
        context (sync or async).
        """
        # Direct backend (set via with_vectors(backend_obj))
        backend = getattr(self, '_vector_backend', None)
        if backend is not None:
            return backend
        # DSN configured but not yet created (with_vectors("./path"))
        dsn = getattr(self, '_vector_dsn', None)
        if dsn is not None:
            try:
                from ..storage import LanceDBBackend
                backend = LanceDBBackend.create(dsn)
                # Cache so we don't re-create on every call
                self._vector_backend = backend
                return backend
            except Exception:
                logger.debug("Failed to lazily create vector backend", exc_info=True)
        return None

    def remember(self, content: str, memory_type: str = 'conversation',
                metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5,
                store_long_term: bool = False) -> None:
        """Store information in memory.

        Args:
            content: The content to remember
            memory_type: Type of memory ('conversation', 'fact', 'task', 'observation')
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            store_long_term: If True, store directly in long-term memory
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        entry = self.memory.remember(content, memory_type, metadata, importance, store_long_term)

        # Also store in vector backend for semantic search if available
        vector_backend = self._get_vector_backend_sync()
        if vector_backend is not None and (store_long_term or importance >= 0.7):
            self._store_to_vector_backend(content, entry, memory_type, metadata)

    def _store_to_vector_backend(self, content: str, entry: Any,
                                  memory_type: str,
                                  metadata: Optional[Dict[str, Any]]) -> None:
        """Store a memory entry in the vector backend (fire-and-forget)."""
        embedding_provider = getattr(self.memory, 'embedding_provider', None)
        if embedding_provider is None:
            logger.debug("No embedding provider for vector backend storage")
            return

        # Resolve backend once (already cached by _get_vector_backend_sync)
        backend = self._get_vector_backend_sync()
        if backend is None:
            return

        async def _store():
            try:
                embedding = await embedding_provider.embed(content)
                meta = dict(metadata or {})
                meta['memory_type'] = memory_type
                meta['content'] = content
                key = f"mem:{getattr(entry, 'id', id(entry))}"
                await backend.store(key, embedding, meta)
            except Exception:
                logger.debug("Vector backend store failed", exc_info=True)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_store())
        except RuntimeError:
            asyncio.run(_store())

    def recall(self, query: Optional[str] = None, memory_type: Optional[str] = None,
              limit: int = 5, include_short_term: bool = True,
              semantic: bool = False, semantic_threshold: float = 0.0) -> list:
        """Recall memories.

        Args:
            query: Search query (if None, returns recent memories)
            memory_type: Filter by memory type
            limit: Maximum number of results
            include_short_term: Whether to include short-term memories
            semantic: If True, use embedding-based similarity search
                instead of substring matching (requires embedding provider
                or a vector backend configured via ``with_vectors()``)
            semantic_threshold: Minimum similarity score for semantic recall

        Returns:
            List of MemoryEntry objects
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return []

        # Try vector backend for semantic recall if available and
        # the in-memory semantic index isn't ready
        vector_backend = self._get_vector_backend_sync()
        if (semantic and query and vector_backend is not None
                and not getattr(self.memory.long_term, '_semantic_ready', False)):
            vector_results = self._recall_from_vector_backend(
                query, limit, semantic_threshold
            )
            if vector_results:
                return vector_results

        return self.memory.recall(
            query, memory_type, limit, include_short_term,
            semantic=semantic, semantic_threshold=semantic_threshold,
        )

    def _recall_from_vector_backend(self, query: str, limit: int,
                                     threshold: float) -> list:
        """Search the vector backend for semantically similar memories."""
        embedding_provider = getattr(self.memory, 'embedding_provider', None)
        if embedding_provider is None:
            return []

        # Resolve backend once (already cached)
        backend = self._get_vector_backend_sync()
        if backend is None:
            return []

        async def _search():
            query_embedding = await embedding_provider.embed(query)
            results = await backend.search(
                query_embedding, limit=limit
            )
            return results

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    results = pool.submit(asyncio.run, _search()).result()
            else:
                results = asyncio.run(_search())
        except Exception:
            logger.debug("Vector backend search failed", exc_info=True)
            return []

        # Convert vector backend results to MemoryEntry-like objects
        from ..memory.memory import MemoryEntry
        entries = []
        for key, score, meta in results:
            if score < threshold:
                continue
            content = meta.get('content', '')
            entry = MemoryEntry(
                content=content,
                memory_type=meta.get('memory_type', 'fact'),
                metadata=meta,
                importance=score,
            )
            entries.append(entry)
        return entries

    def get_memory_context(self, max_entries: int = 5) -> str:
        """Get formatted context from memories.

        Args:
            max_entries: Maximum number of memory entries to include

        Returns:
            Formatted string with memory context
        """
        if not self.memory_enabled:
            return ""

        return self.memory.get_context(max_entries)

    def consolidate_memory(self, importance_threshold: float = 0.6) -> None:
        """Consolidate short-term memories to long-term storage.

        Args:
            importance_threshold: Minimum importance to consolidate
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.consolidate_to_long_term(importance_threshold)

    def clear_short_term_memory(self) -> None:
        """Clear short-term memory."""
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.clear_short_term()

    def save_memory(self) -> None:
        """Save memory to persistent storage."""
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.save()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        if not self.memory_enabled:
            return {'memory_enabled': False}

        stats = self.memory.stats()
        stats['memory_enabled'] = True
        return stats
