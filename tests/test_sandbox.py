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
    """Test the SubprocessSandbox execution backend."""

    @pytest.mark.asyncio
    async def test_basic_execution(self):
        """Simple code should run and return output."""
        sandbox = SubprocessSandbox()
        result = await sandbox.execute("print('hello')", SandboxLimits())
        assert result.success
        assert result.output.strip() == "hello"

    @pytest.mark.asyncio
    async def test_math_execution(self):
        """Computation should work."""
        sandbox = SubprocessSandbox()
        result = await sandbox.execute("print(2 + 2)", SandboxLimits())
        assert result.success
        assert result.output.strip() == "4"

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Long-running code should be killed after timeout."""
        sandbox = SubprocessSandbox()
        result = await sandbox.execute(
            "import time; time.sleep(60)",
            SandboxLimits(timeout_seconds=1.0)
        )
        assert result.timed_out
        assert not result.success

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Errors should be captured, not crash the sandbox."""
        sandbox = SubprocessSandbox()
        result = await sandbox.execute(
            "raise ValueError('boom')",
            SandboxLimits()
        )
        assert not result.success
        assert result.exit_code != 0
        assert "boom" in (result.error or "")

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        """Syntax errors should be handled."""
        sandbox = SubprocessSandbox()
        result = await sandbox.execute(
            "def broken(",
            SandboxLimits()
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_output_capture(self):
        """Multi-line output should be captured."""
        sandbox = SubprocessSandbox()
        result = await sandbox.execute(
            "for i in range(3): print(i)",
            SandboxLimits()
        )
        assert result.success
        assert result.output.strip() == "0\n1\n2"

    @pytest.mark.asyncio
    async def test_json_output(self):
        """JSON output from tool code should be parseable."""
        sandbox = SubprocessSandbox()
        result = await sandbox.execute(
            'import json; print(json.dumps({"result": 42, "error": None}))',
            SandboxLimits()
        )
        assert result.success
        import json
        parsed = json.loads(result.output.strip())
        assert parsed["result"] == 42


# ──────────────────────────────────────────────
# 2. build_tool_code
# ──────────────────────────────────────────────

class TestBuildToolCode:
    """Test tool code serialization."""

    def test_generates_valid_code(self):
        """Generated code should be syntactically valid Python."""
        def add(x: int, y: int) -> int:
            return x + y

        import inspect
        source = inspect.getsource(add)
        import textwrap
        source = textwrap.dedent(source)
        code = build_tool_code(source, "add", {"x": 3, "y": 4})

        assert "add" in code
        assert "params" in code
        assert "json.dumps" in code

    @pytest.mark.asyncio
    async def test_generated_code_runs(self):
        """Generated code should actually execute correctly."""
        def multiply(x: int, y: int) -> int:
            return x * y

        import inspect, textwrap
        source = textwrap.dedent(inspect.getsource(multiply))
        code = build_tool_code(source, "multiply", {"x": 6, "y": 7})

        sandbox = SubprocessSandbox()
        result = await sandbox.execute(code, SandboxLimits())

        assert result.success
        import json
        parsed = json.loads(result.output.strip())
        assert parsed["result"] == 42


# ──────────────────────────────────────────────
# 3. Agent + Sandbox integration
# ──────────────────────────────────────────────

class TestAgentSandbox:
    """Test Agent.with_sandbox() integration."""

    def test_with_sandbox_chaining(self):
        """with_sandbox should return self for chaining."""
        agent = Agent("test")
        result = agent.with_sandbox()
        assert result is agent
        assert agent._sandbox is not None

    def test_with_sandbox_default_backend(self):
        """Default backend should be SubprocessSandbox."""
        agent = Agent("test").with_sandbox()
        assert isinstance(agent._sandbox, SubprocessSandbox)

    def test_with_sandbox_custom_timeout(self):
        """Custom timeout should be set."""
        agent = Agent("test").with_sandbox(timeout=10.0)
        assert agent._sandbox_limits.timeout_seconds == 10.0

    @pytest.mark.asyncio
    async def test_sandboxed_tool_execution(self):
        """Tool should execute in sandbox and return result."""
        def add(x: int, y: int) -> int:
            return x + y

        agent = Agent("test").with_sandbox(timeout=10)
        agent.with_tools([Tool(add, permission=ToolPermission.READONLY)])

        result = await agent.call("add", {"x": 5, "y": 3})
        assert result == 8

    @pytest.mark.asyncio
    async def test_sandboxed_tool_timeout(self):
        """Tool that takes too long should raise TimeoutError."""
        def slow_tool() -> str:
            import time
            time.sleep(60)
            return "done"

        agent = Agent("test").with_sandbox(timeout=1)
        agent.with_tools([Tool(slow_tool)])

        with pytest.raises(TimeoutError, match="timed out"):
            await agent.call("slow_tool", {})

    @pytest.mark.asyncio
    async def test_sandboxed_tool_error(self):
        """Tool errors in sandbox should raise RuntimeError."""
        def broken_tool() -> str:
            raise ValueError("something broke")

        agent = Agent("test").with_sandbox(timeout=5)
        agent.with_tools([Tool(broken_tool)])

        with pytest.raises(RuntimeError, match="failed in sandbox"):
            await agent.call("broken_tool", {})

    @pytest.mark.asyncio
    async def test_sandbox_event_metadata(self):
        """Sandboxed calls should record sandboxed=True in events."""
        def add(x: int, y: int) -> int:
            return x + y

        agent = Agent("test").with_sandbox(timeout=10)
        agent.with_tools([Tool(add)])

        await agent.call("add", {"x": 1, "y": 2})

        events = agent.observer.get_events()
        tool_events = [e for e in events if e.get("event") == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0]["sandboxed"] is True

    @pytest.mark.asyncio
    async def test_no_sandbox_by_default(self):
        """Without with_sandbox(), tools run in-process (sandboxed=False)."""
        def add(x: int, y: int) -> int:
            return x + y

        agent = Agent("test")
        agent.with_tools([Tool(add)])
        await agent.call("add", {"x": 1, "y": 2})

        events = agent.observer.get_events()
        tool_events = [e for e in events if e.get("event") == "tool_call"]
        assert tool_events[0]["sandboxed"] is False
