"""StorageMixin – storage backend builder methods extracted from Agent."""

import logging
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


class StorageMixin:
    """Mixin providing storage backend configuration for the Agent class.

    Methods here manage the key-value (Redis) and vector (LanceDB)
    storage backends. The backends are lazily created on first access
    and consumed by:
      - ``MemoryMixin`` for semantic search (vector backend)
      - ``Session.checkpoint()`` / ``SessionManager.resume()`` (storage backend)
      - ``serve()`` for Redis-backed sessions/tasks (backend URL)
    """

    def with_backend(
        self,
        backend: Union[str, Any],
    ) -> 'Agent':
        """Set the key-value storage backend for sessions, checkpoints, and memory.

        Args:
            backend: Either a Redis URL string (``redis://host:6379/0``)
                or a :class:`StorageBackend` instance.

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("bot").with_backend("redis://localhost:6379")
            >>> # or with a custom backend
            >>> agent = Agent("bot").with_backend(MyCustomBackend())
        """
        if isinstance(backend, str):
            # Treat as Redis URL — lazy-create on first use
            self._backend_url = backend
            self._storage_backend = None  # Created async on first access
        else:
            self._storage_backend = backend
            self._backend_url = None
        return self

    def with_vectors(
        self,
        backend: Union[str, Any],
        dimension: int = 384,
    ) -> 'Agent':
        """Set the vector storage backend for semantic search.

        Args:
            backend: Either a local path (``./vectors``) or cloud URI
                for LanceDB, or a :class:`VectorBackend` instance.
            dimension: Embedding dimension (default: 384 for MiniLM).

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("bot").with_vectors("./vectors")
            >>> # or with a custom backend
            >>> from agentu.storage import InMemoryVectorBackend
            >>> agent = Agent("bot").with_vectors(InMemoryVectorBackend())
        """
        if isinstance(backend, str):
            self._vector_dsn = backend
            self._vector_dimension = dimension
            self._vector_backend = None  # Created on first access
        else:
            self._vector_backend = backend
            self._vector_dsn = None
        return self

    async def get_storage_backend(self):
        """Get or lazily create the key-value storage backend.

        Returns the configured StorageBackend. If with_backend() was called
        with a Redis URL, the backend is created on first access.
        Returns None if no backend is configured (uses component defaults).
        """
        if self._storage_backend is not None:
            return self._storage_backend
        if self._backend_url:
            from ..storage import RedisStorageBackend
            self._storage_backend = await RedisStorageBackend.create(self._backend_url)
            return self._storage_backend
        return None

    async def get_vector_backend(self):
        """Get or lazily create the vector storage backend.

        Returns the configured VectorBackend. If with_vectors() was called
        with a path/URI, a LanceDBBackend is created on first access.
        Returns None if no backend is configured (uses in-memory vectors).
        """
        if self._vector_backend is not None:
            return self._vector_backend
        if self._vector_dsn:
            from ..storage import LanceDBBackend
            self._vector_backend = LanceDBBackend.create(self._vector_dsn)
            return self._vector_backend
        return None
