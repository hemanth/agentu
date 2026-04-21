"""Tests for sandbox execution backends."""

import asyncio
import pytest

from agentu import Agent, Tool, ToolPermission
from agentu.runtime.sandbox import (
    SubprocessSandbox, SandboxLimits, SandboxResult, build_tool_code,
)


# ──────────────────────────────────────────────
# 1. SubprocessSandbox unit tests
# ──────────────────────────────────────────────

class TestSubprocessSandbox:

    @pytest.mark.asyncio
    async def test_basic_execution(self):
        sandbox = SubprocessSandbox()
        result = await sandbox.execute("print('hello')", SandboxLimits())
        assert result.success
        assert result.output.strip() == "hello"

    @pytest.mark.asyncio
    async def test_timeout(self):
        sandbox = SubprocessSandbox()
        result = await sandbox.execute(
            "import time; time.sleep(60)",
            SandboxLimits(timeout_seconds=1.0)
        )
        assert result.timed_out
        assert not result.success

    @pytest.mark.asyncio
    async def test_error_handling(self):
        sandbox = SubprocessSandbox()
        result = await sandbox.execute("raise ValueError('boom')", SandboxLimits())
        assert not result.success
        assert "boom" in (result.error or "")

    @pytest.mark.asyncio
    async def test_json_output(self):
        sandbox = SubprocessSandbox()
        result = await sandbox.execute(
            'import json; print(json.dumps({"result": 42, "error": None}))',
            SandboxLimits()
        )
        assert result.success
        import json
        assert json.loads(result.output.strip())["result"] == 42


# ──────────────────────────────────────────────
# 2. build_tool_code
# ──────────────────────────────────────────────

class TestBuildToolCode:

    @pytest.mark.asyncio
    async def test_generated_code_runs(self):
        def multiply(x: int, y: int) -> int:
            return x * y

        import inspect, textwrap
        source = textwrap.dedent(inspect.getsource(multiply))
        code = build_tool_code(source, "multiply", {"x": 6, "y": 7})

        sandbox = SubprocessSandbox()
        result = await sandbox.execute(code, SandboxLimits())
        assert result.success
        import json
        assert json.loads(result.output.strip())["result"] == 42


# ──────────────────────────────────────────────
# 3. Agent.with_sandbox(read_tools=, write_tools=)
# ──────────────────────────────────────────────

class TestAgentSandbox:

    def test_chaining(self):
        assert Agent("test").with_sandbox() is not None

    def test_read_tools_get_readonly(self):
        def search(q: str) -> str:
            return q

        agent = Agent("test").with_sandbox(read_tools=[search])
        tool = next(t for t in agent.tools if t.name == "search")
        assert tool.permission == ToolPermission.READONLY

    def test_write_tools_get_write(self):
        def save(data: str) -> str:
            return data

        agent = Agent("test").with_sandbox(write_tools=[save])
        tool = next(t for t in agent.tools if t.name == "save")
        assert tool.permission == ToolPermission.WRITE

    def test_both_read_and_write(self):
        def search(q: str) -> str:
            return q
        def save(data: str) -> str:
            return data

        agent = Agent("test").with_sandbox(read_tools=[search], write_tools=[save])
        names = [t.name for t in agent.tools]
        assert "search" in names
        assert "save" in names

    @pytest.mark.asyncio
    async def test_sandboxed_read_tool(self):
        def add(x: int, y: int) -> int:
            return x + y

        agent = Agent("test").with_sandbox(read_tools=[add], timeout=10)
        result = await agent.call("add", {"x": 5, "y": 3})
        assert result == 8

    @pytest.mark.asyncio
    async def test_sandboxed_write_tool(self):
        def concat(a: str, b: str) -> str:
            return a + b

        agent = Agent("test").with_sandbox(write_tools=[concat], timeout=10)
        result = await agent.call("concat", {"a": "hello", "b": "world"})
        assert result == "helloworld"

    @pytest.mark.asyncio
    async def test_timeout_kills_tool(self):
        def slow() -> str:
            import time
            time.sleep(60)
            return "done"

        agent = Agent("test").with_sandbox(write_tools=[slow], timeout=1)
        with pytest.raises(TimeoutError):
            await agent.call("slow", {})

    @pytest.mark.asyncio
    async def test_error_in_sandbox(self):
        def broken() -> str:
            raise ValueError("boom")

        agent = Agent("test").with_sandbox(write_tools=[broken], timeout=5)
        with pytest.raises(RuntimeError, match="failed in sandbox"):
            await agent.call("broken", {})

    @pytest.mark.asyncio
    async def test_sandbox_events_capture_details(self):
        """Observer should capture sandbox exit_code and stderr."""
        def add(x: int, y: int) -> int:
            return x + y

        agent = Agent("test").with_sandbox(read_tools=[add], timeout=10)
        await agent.call("add", {"x": 1, "y": 2})

        events = agent.observer.get_events()
        sandbox_events = [e for e in events if "sandbox_exit_code" in e]
        assert len(sandbox_events) >= 1
        assert sandbox_events[0]["sandbox_exit_code"] == 0
        assert sandbox_events[0]["sandbox_timed_out"] is False

    @pytest.mark.asyncio
    async def test_sandbox_events_capture_stderr_on_error(self):
        """Observer should capture stderr when tool errors."""
        def broken() -> str:
            raise ValueError("boom")

        agent = Agent("test").with_sandbox(write_tools=[broken], timeout=5)
        try:
            await agent.call("broken", {})
        except RuntimeError:
            pass

        events = agent.observer.get_events()
        sandbox_events = [e for e in events if "sandbox_stderr" in e]
        assert len(sandbox_events) >= 1
        assert sandbox_events[0]["sandbox_stderr"] is not None

    @pytest.mark.asyncio
    async def test_no_sandbox_by_default(self):
        def add(x: int, y: int) -> int:
            return x + y

        agent = Agent("test")
        agent.with_tools([Tool(add)])
        await agent.call("add", {"x": 1, "y": 2})

        events = agent.observer.get_events()
        tool_events = [e for e in events if e.get("event") == "tool_call"]
        assert tool_events[0]["sandboxed"] is False
