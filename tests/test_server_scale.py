"""Tests for server scalability: TaskQueue, RedisSessionStore, create_server, workers."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agentu import Agent, Tool, AgentServer
from agentu.runtime.serve import create_server, serve
from agentu.runtime.tasks import TaskQueue, TaskInfo, TaskStatus


# ── fixtures ────────────────────────────────────────────────────────────────


def _calculator(x: float, y: float, operation: str) -> float:
    ops = {"add": x + y, "subtract": x - y, "multiply": x * y, "divide": x / y if y else 0}
    return ops.get(operation, 0)


@pytest.fixture
def agent():
    ag = Agent("scale_test", model="qwen3:latest", enable_memory=True)
    ag.with_tools([
        Tool(
            name="calculator",
            description="calc",
            function=_calculator,
            parameters={"x": "float", "y": "float", "operation": "str"},
        )
    ])
    return ag


@pytest.fixture
def server(agent):
    return AgentServer(agent, title="Scale Test API")


@pytest.fixture
def client(server):
    return TestClient(server.app)


# ── TaskQueue in-memory tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_submit_and_get():
    """Submit a task and retrieve the result."""
    tq = TaskQueue(max_concurrent=5, task_ttl=60)

    async def work():
        return {"answer": 42}

    info = await tq.submit(lambda: work())
    assert info.status == TaskStatus.SUBMITTED
    assert info.task_id

    # Wait for completion
    await asyncio.sleep(0.1)

    result = await tq.get(info.task_id)
    assert result is not None
    assert result.status == TaskStatus.COMPLETED
    assert result.result == {"answer": 42}
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_task_failure():
    """A failing task should transition to FAILED."""
    tq = TaskQueue(max_concurrent=5, task_ttl=60)

    async def failing():
        raise ValueError("boom")

    info = await tq.submit(lambda: failing())
    await asyncio.sleep(0.1)

    result = await tq.get(info.task_id)
    assert result is not None
    assert result.status == TaskStatus.FAILED
    assert "boom" in result.result


@pytest.mark.asyncio
async def test_task_cancel():
    """Cancelling a submitted/working task."""
    tq = TaskQueue(max_concurrent=5, task_ttl=60)

    async def slow():
        await asyncio.sleep(10)
        return "done"

    info = await tq.submit(lambda: slow())
    # Give it a moment to start
    await asyncio.sleep(0.05)

    cancelled = await tq.cancel(info.task_id)
    assert cancelled is True

    result = await tq.get(info.task_id)
    assert result is not None
    assert result.status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_task_cancel_nonexistent():
    """Cancelling a non-existent task returns False."""
    tq = TaskQueue()
    assert await tq.cancel("no-such-task") is False


@pytest.mark.asyncio
async def test_task_max_concurrent():
    """Tasks beyond the concurrency limit should queue up."""
    tq = TaskQueue(max_concurrent=2, task_ttl=60)
    started = []

    async def tracked(n):
        started.append(n)
        await asyncio.sleep(0.2)
        return n

    # Submit 4 tasks with max_concurrent=2
    infos = []
    for i in range(4):
        n = i
        infos.append(await tq.submit(lambda n=n: tracked(n)))

    # After a brief moment, at most 2 should be running
    await asyncio.sleep(0.05)
    assert len(started) <= 2

    # Wait for all to complete
    await asyncio.sleep(1.0)
    for info in infos:
        result = await tq.get(info.task_id)
        assert result is not None
        assert result.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_task_ttl_eviction():
    """Completed tasks should be evicted after TTL."""
    tq = TaskQueue(max_concurrent=5, task_ttl=1)

    async def work():
        return "result"

    info = await tq.submit(lambda: work())
    await asyncio.sleep(0.1)

    # Should be available right after completion
    assert await tq.get(info.task_id) is not None

    # Wait for TTL
    await asyncio.sleep(1.5)

    # Should be evicted from in-memory store
    assert tq._tasks.get(info.task_id) is None


@pytest.mark.asyncio
async def test_task_list():
    """list_tasks returns all known tasks."""
    tq = TaskQueue(max_concurrent=5, task_ttl=60)

    async def work():
        return "ok"

    await tq.submit(lambda: work())
    await tq.submit(lambda: work())
    await asyncio.sleep(0.1)

    tasks = await tq.list_tasks()
    assert len(tasks) == 2


@pytest.mark.asyncio
async def test_task_info_serialisation():
    """TaskInfo round-trips through to_dict / from_dict."""
    info = TaskInfo(task_id="abc-123")
    info.status = TaskStatus.COMPLETED
    info.result = {"x": 1}
    info.completed_at = time.time()

    d = info.to_dict()
    assert d["task_id"] == "abc-123"
    assert d["status"] == "completed"

    restored = TaskInfo.from_dict(d)
    assert restored.task_id == "abc-123"
    assert restored.status == TaskStatus.COMPLETED
    assert restored.result == {"x": 1}


# ── RedisSessionStore tests (mocked) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_redis_session_store_save_load():
    """RedisSessionStore save + load with mocked redis."""
    with patch("agentu.runtime.redis_backend.HAS_REDIS", True), \
         patch("agentu.runtime.redis_backend.aioredis") as mock_redis_mod:

        # Build mock Redis instance
        mock_redis = AsyncMock()
        mock_pool = MagicMock()
        mock_redis_mod.ConnectionPool.from_url.return_value = mock_pool
        mock_redis_mod.Redis.return_value = mock_redis

        from agentu.runtime.redis_backend import RedisSessionStore
        store = RedisSessionStore(redis_url="redis://localhost:6379/0", ttl=300)

        # save_session
        await store.save_session("s1", {"user": "alice"})
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert "agentu:session:s1" in call_args[0]

        # load_session
        mock_redis.get.return_value = '{"user": "alice"}'
        data = await store.load_session("s1")
        assert data == {"user": "alice"}

        # load_session miss
        mock_redis.get.return_value = None
        assert await store.load_session("nope") is None


@pytest.mark.asyncio
async def test_redis_session_store_delete():
    """RedisSessionStore delete."""
    with patch("agentu.runtime.redis_backend.HAS_REDIS", True), \
         patch("agentu.runtime.redis_backend.aioredis") as mock_redis_mod:

        mock_redis = AsyncMock()
        mock_redis_mod.ConnectionPool.from_url.return_value = MagicMock()
        mock_redis_mod.Redis.return_value = mock_redis

        from agentu.runtime.redis_backend import RedisSessionStore
        store = RedisSessionStore()

        mock_redis.delete.return_value = 1
        assert await store.delete_session("s1") is True

        mock_redis.delete.return_value = 0
        assert await store.delete_session("s2") is False


@pytest.mark.asyncio
async def test_redis_session_store_exists():
    """RedisSessionStore exists check."""
    with patch("agentu.runtime.redis_backend.HAS_REDIS", True), \
         patch("agentu.runtime.redis_backend.aioredis") as mock_redis_mod:

        mock_redis = AsyncMock()
        mock_redis_mod.ConnectionPool.from_url.return_value = MagicMock()
        mock_redis_mod.Redis.return_value = mock_redis

        from agentu.runtime.redis_backend import RedisSessionStore
        store = RedisSessionStore()

        mock_redis.exists.return_value = 1
        assert await store.exists("s1") is True

        mock_redis.exists.return_value = 0
        assert await store.exists("s2") is False


def test_redis_session_store_import_error():
    """RedisSessionStore raises ImportError when redis is not installed."""
    with patch("agentu.runtime.redis_backend.HAS_REDIS", False):
        from agentu.runtime.redis_backend import RedisSessionStore
        with pytest.raises(ImportError, match="redis"):
            RedisSessionStore()


# ── create_server tests ─────────────────────────────────────────────────────


def test_create_server_returns_fastapi_app(agent):
    """create_server() returns a FastAPI ASGI application."""
    app = create_server(agent, enable_cors=True)
    # FastAPI instances are callable ASGI apps
    assert callable(app)
    assert hasattr(app, "routes")


def test_create_server_has_task_routes(agent):
    """The app returned by create_server has /tasks endpoints."""
    app = create_server(agent)
    route_paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/tasks/{task_id}" in route_paths
    assert "/tasks" in route_paths


# ── serve() parameter tests ─────────────────────────────────────────────────


def test_serve_accepts_workers_param(agent):
    """serve() accepts workers= without error (we mock uvicorn.run)."""
    with patch("agentu.runtime.serve.uvicorn") as mock_uv:
        mock_uv.run = MagicMock()
        serve(agent, host="127.0.0.1", port=9999, workers=4)
        mock_uv.run.assert_called_once()
        call_kwargs = mock_uv.run.call_args
        assert call_kwargs[1].get("workers") == 4 or call_kwargs.kwargs.get("workers") == 4


def test_serve_accepts_redis_url_param(agent):
    """serve() accepts redis_url= without error (we mock uvicorn.run)."""
    with patch("agentu.runtime.serve.uvicorn") as mock_uv, \
         patch("agentu.runtime.redis_backend.HAS_REDIS", True), \
         patch("agentu.runtime.redis_backend.aioredis") as mock_redis_mod, \
         patch("agentu.runtime.tasks.HAS_REDIS", True), \
         patch("agentu.runtime.tasks.aioredis") as mock_tasks_redis:
        mock_uv.run = MagicMock()
        mock_redis_mod.ConnectionPool.from_url.return_value = MagicMock()
        mock_redis_mod.Redis.return_value = AsyncMock()
        mock_tasks_redis.from_url.return_value = AsyncMock()
        serve(agent, redis_url="redis://localhost:6379/0")
        mock_uv.run.assert_called_once()


# ── /tasks HTTP endpoint tests ──────────────────────────────────────────────


def test_get_task_not_found(client):
    """GET /tasks/{id} returns 404 for unknown task."""
    response = client.get("/tasks/nonexistent-id")
    assert response.status_code == 404


def test_delete_task_not_found(client):
    """DELETE /tasks/{id} returns 404 for unknown task."""
    response = client.delete("/tasks/nonexistent-id")
    assert response.status_code == 404


def test_list_tasks_empty(client):
    """GET /tasks returns empty list initially."""
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["tasks"] == []


# ── async /process endpoint test ────────────────────────────────────────────


def test_process_background_returns_task_id(agent):
    """POST /process?background=true returns a task_id."""
    # Mock agent.infer to avoid needing a real LLM
    async def fake_infer(text):
        return {"tool_used": "none", "result": text}

    agent.infer = fake_infer
    server = AgentServer(agent, title="Async Test")
    client = TestClient(server.app)

    response = client.post(
        "/process?background=true",
        json={"input": "hello"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "submitted"

    # The task should be findable
    task_resp = client.get(f"/tasks/{data['task_id']}")
    assert task_resp.status_code == 200


def test_process_sync_still_works(agent):
    """POST /process (no background param) works synchronously."""
    async def fake_infer(text):
        return {"tool_used": "none", "result": text}

    agent.infer = fake_infer
    server = AgentServer(agent, title="Sync Test")
    client = TestClient(server.app)

    response = client.post("/process", json={"input": "hi"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["input"] == "hi"


# ── AgentServer redis_url parameter test ────────────────────────────────────


def test_agent_server_redis_url_param(agent):
    """AgentServer accepts redis_url and creates Redis session store."""
    with patch("agentu.runtime.redis_backend.HAS_REDIS", True), \
         patch("agentu.runtime.redis_backend.aioredis") as mock_redis_mod, \
         patch("agentu.runtime.tasks.HAS_REDIS", True), \
         patch("agentu.runtime.tasks.aioredis") as mock_tasks_redis:
        mock_redis_mod.ConnectionPool.from_url.return_value = MagicMock()
        mock_redis_mod.Redis.return_value = AsyncMock()
        mock_tasks_redis.from_url.return_value = AsyncMock()

        server = AgentServer(agent, redis_url="redis://localhost:6379/0")
        assert server._redis_session_store is not None
        assert server.task_queue._redis is not None


def test_agent_server_no_redis_by_default(agent):
    """AgentServer without redis_url uses in-memory backends."""
    server = AgentServer(agent)
    assert server._redis_session_store is None
    assert server.task_queue._redis is None
