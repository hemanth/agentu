"""Async task queue for background agent processing.

Provides ``TaskQueue`` — a lightweight task manager that allows long-running
agent calls to be submitted, polled, and cancelled.  The queue supports:

* Configurable max-concurrent-tasks via an ``asyncio.Semaphore``
* Automatic result expiry (default: 1 hour)
* In-memory storage by default; Redis-backed when a Redis URL is provided
* Task states: ``submitted → working → completed / failed / cancelled``
"""

import asyncio
import json
import logging
import time
import uuid
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# --- optional redis import ---------------------------------------------------
try:
    import redis.asyncio as aioredis

    HAS_REDIS = True
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore[assignment]
    HAS_REDIS = False


class TaskStatus(str, Enum):
    """Lifecycle states for a queued task."""

    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskInfo:
    """Metadata and result payload for a single task.

    Attributes:
        task_id: Unique identifier.
        status: Current lifecycle state.
        result: Output value when completed (or error message on failure).
        created_at: Epoch timestamp when the task was submitted.
        completed_at: Epoch timestamp when the task finished (or ``None``).
    """

    __slots__ = ("task_id", "status", "result", "created_at", "completed_at")

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.status = TaskStatus.SUBMITTED
        self.result: Any = None
        self.created_at: float = time.time()
        self.completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status.value,
            "created_at": self.created_at,
        }
        if self.completed_at is not None:
            d["completed_at"] = self.completed_at
        if self.result is not None:
            d["result"] = self.result
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskInfo":
        info = cls(task_id=data["task_id"])
        info.status = TaskStatus(data["status"])
        info.result = data.get("result")
        info.created_at = data.get("created_at", time.time())
        info.completed_at = data.get("completed_at")
        return info


_REDIS_TASK_PREFIX = "agentu:task:"
_DEFAULT_TASK_TTL = 3600  # 1 hour


class TaskQueue:
    """Background task manager with optional Redis persistence.

    Args:
        max_concurrent: Maximum number of tasks running simultaneously.
        task_ttl: Seconds before completed/failed results are evicted.
        redis_url: When provided, task metadata is stored in Redis instead
            of in-process memory.

    Raises:
        ImportError: If ``redis_url`` is given but the ``redis`` package is
            not installed.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        task_ttl: int = _DEFAULT_TASK_TTL,
        redis_url: Optional[str] = None,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.task_ttl = task_ttl
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: Dict[str, TaskInfo] = {}
        self._asyncio_tasks: Dict[str, asyncio.Task] = {}

        # Optional Redis backend
        self._redis: Optional["aioredis.Redis"] = None
        if redis_url is not None:
            if not HAS_REDIS:
                raise ImportError(
                    "The 'redis' package is required for Redis-backed TaskQueue. "
                    "Install it with: pip install 'agentu[redis]'"
                )
            self._redis = aioredis.from_url(redis_url, decode_responses=True)

    # -- storage helpers -------------------------------------------------------

    async def _save_task(self, info: TaskInfo) -> None:
        """Persist task info (in-memory or Redis)."""
        self._tasks[info.task_id] = info
        if self._redis is not None:
            key = f"{_REDIS_TASK_PREFIX}{info.task_id}"
            await self._redis.set(key, info.to_json(), ex=self.task_ttl)

    async def _load_task(self, task_id: str) -> Optional[TaskInfo]:
        """Load task info (in-memory or Redis)."""
        # Check in-memory first
        info = self._tasks.get(task_id)
        if info is not None:
            return info

        # Fallback to Redis
        if self._redis is not None:
            key = f"{_REDIS_TASK_PREFIX}{task_id}"
            raw = await self._redis.get(key)
            if raw is not None:
                info = TaskInfo.from_dict(json.loads(raw))
                self._tasks[task_id] = info
                return info

        return None

    # -- public API ------------------------------------------------------------

    async def submit(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        task_id: Optional[str] = None,
    ) -> TaskInfo:
        """Submit an async callable for background execution.

        The *coro_factory* is a zero-argument callable that returns an
        awaitable (coroutine).  It will be scheduled and run once the
        concurrency semaphore permits.

        Args:
            coro_factory: ``lambda: agent.infer(user_input)``
            task_id: Optional custom ID; auto-generated if omitted.

        Returns:
            ``TaskInfo`` with status ``submitted``.
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        info = TaskInfo(task_id=task_id)
        await self._save_task(info)

        asyncio_task = asyncio.create_task(self._run(info, coro_factory))
        self._asyncio_tasks[task_id] = asyncio_task
        logger.info("Task %s submitted", task_id)
        return info

    async def _run(
        self,
        info: TaskInfo,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        """Internal runner: acquires semaphore, executes, stores result."""
        async with self._semaphore:
            # Task may have been cancelled while waiting for semaphore
            if info.status == TaskStatus.CANCELLED:
                return

            info.status = TaskStatus.WORKING
            await self._save_task(info)
            logger.debug("Task %s working", info.task_id)

            try:
                result = await coro_factory()
                info.status = TaskStatus.COMPLETED
                info.result = result
            except asyncio.CancelledError:
                info.status = TaskStatus.CANCELLED
            except Exception as exc:
                info.status = TaskStatus.FAILED
                info.result = str(exc)
                logger.error("Task %s failed: %s", info.task_id, exc)
            finally:
                info.completed_at = time.time()
                await self._save_task(info)
                # Schedule cleanup after TTL
                asyncio.get_event_loop().call_later(
                    self.task_ttl, self._evict, info.task_id
                )

    def _evict(self, task_id: str) -> None:
        """Remove a completed task from in-memory storage after TTL."""
        self._tasks.pop(task_id, None)
        self._asyncio_tasks.pop(task_id, None)
        logger.debug("Evicted task %s (TTL expired)", task_id)

    async def get(self, task_id: str) -> Optional[TaskInfo]:
        """Retrieve the current state of a task.

        Args:
            task_id: Task identifier returned by ``submit()``.

        Returns:
            ``TaskInfo`` or ``None`` if the task was never submitted or
            has been evicted.
        """
        return await self._load_task(task_id)

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running or submitted task.

        Args:
            task_id: Task to cancel.

        Returns:
            ``True`` if the task was found and a cancellation was attempted.
        """
        info = await self._load_task(task_id)
        if info is None:
            return False

        # Only cancel if still running or submitted
        if info.status in (TaskStatus.SUBMITTED, TaskStatus.WORKING):
            info.status = TaskStatus.CANCELLED
            info.completed_at = time.time()
            await self._save_task(info)

            asyncio_task = self._asyncio_tasks.get(task_id)
            if asyncio_task and not asyncio_task.done():
                asyncio_task.cancel()

            logger.info("Task %s cancelled", task_id)
            return True
        return False

    async def list_tasks(self) -> List[Dict[str, Any]]:
        """List all known tasks (in-memory snapshot).

        Returns:
            List of task summary dicts.
        """
        return [info.to_dict() for info in self._tasks.values()]

    async def close(self) -> None:
        """Cancel all running tasks and close Redis if connected."""
        for task_id, asyncio_task in list(self._asyncio_tasks.items()):
            if not asyncio_task.done():
                asyncio_task.cancel()
        if self._redis is not None:
            await self._redis.aclose()
