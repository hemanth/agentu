"""SandboxMixin – sandbox execution methods extracted from Agent."""

import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Callable

from .tools import Tool, ToolPermission
from ..middleware.observe import EventType

logger = logging.getLogger(__name__)


class SandboxMixin:
    """Mixin providing sandbox execution for the Agent class.

    Methods here assume they are mixed into an Agent instance that has
    ``_sandbox``, ``_sandbox_limits``, ``tools``, ``observer``, and
    ``codemode`` attributes.
    """

    def with_sandbox(
        self,
        read_tools: Optional[List[Union[Tool, Callable]]] = None,
        write_tools: Optional[List[Union[Tool, Callable]]] = None,
        backend=None,
        timeout: float = 30.0,
        max_memory_mb: Optional[int] = 256,
        allow_network: bool = True,
        codemode: bool = False,
    ) -> 'Agent':
        """Set up a sandboxed execution environment.

        Tools in `read_tools` get READONLY permission (no side effects).
        Tools in `write_tools` get WRITE permission (has side effects).
        All tools run in an isolated subprocess.

        When codemode=True, the LLM writes Python code that calls tools via
        a typed API, instead of making individual JSON tool calls. The code
        runs in the sandbox with the same isolation and timeout guarantees.

        Args:
            read_tools: Tools with no side effects (READONLY)
            write_tools: Tools with side effects (WRITE)
            backend: A SandboxBackend instance. Defaults to SubprocessSandbox.
            timeout: Max seconds per tool call (default: 30)
            max_memory_mb: Memory limit in MB (default: 256, None=unlimited)
            allow_network: Allow outbound network access (default: True).
                Set to False to block HTTP via proxy env vars.
            codemode: If True, enable code mode for this sandbox (default: False)

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("bot").with_sandbox(
            ...     read_tools=[search, get_weather],
            ...     write_tools=[save_file, send_email],
            ...     timeout=10,
            ...     allow_network=False,
            ... )
        """
        from ..runtime.sandbox import SubprocessSandbox, SandboxLimits

        # Register read tools with READONLY permission
        if read_tools:
            for t in read_tools:
                if isinstance(t, Tool):
                    t.permission = ToolPermission.READONLY
                    self._add_tool_internal(t)
                else:
                    self._add_tool_internal(Tool(t, permission=ToolPermission.READONLY))

        # Register write tools with WRITE permission
        if write_tools:
            for t in write_tools:
                if isinstance(t, Tool):
                    t.permission = ToolPermission.WRITE
                    self._add_tool_internal(t)
                else:
                    self._add_tool_internal(Tool(t, permission=ToolPermission.WRITE))

        # Configure sandbox backend
        self._sandbox = backend or SubprocessSandbox()
        self._sandbox_limits = SandboxLimits(
            timeout_seconds=timeout,
            max_memory_mb=max_memory_mb,
            allow_network=allow_network,
        )
        if codemode:
            self.codemode = True
        logger.info(f"Sandbox enabled: {type(self._sandbox).__name__} (timeout={timeout}s, codemode={self.codemode})")
        return self

    async def _call_sandboxed(self, tool, parameters: Dict[str, Any]) -> Any:
        """Execute a tool in the sandbox backend.

        Serializes the tool function to source code, runs it in subprocess,
        and parses the JSON result.
        """
        import inspect as _inspect
        from ..runtime.sandbox import build_tool_code

        try:
            func_source = _inspect.getsource(tool.function)
        except (OSError, TypeError):
            # Can't get source (builtins, lambdas, etc.) -- fall back to in-process
            logger.warning(
                f"Tool '{tool.name}' cannot be sandboxed (no source). Running in-process."
            )
            result = tool.function(**parameters)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        # Dedent the source in case it's a nested function
        import textwrap
        func_source = textwrap.dedent(func_source)

        code = build_tool_code(func_source, tool.function.__name__, parameters)
        sandbox_result = await self._sandbox.execute(code, self._sandbox_limits)

        # Record sandbox execution details in observer
        self.observer.record(EventType.TOOL_CALL, {
            "tool_name": tool.name,
            "sandbox_exit_code": sandbox_result.exit_code,
            "sandbox_timed_out": sandbox_result.timed_out,
            "sandbox_stderr": sandbox_result.error,
        })

        if sandbox_result.timed_out:
            raise TimeoutError(
                f"Tool '{tool.name}' timed out after {self._sandbox_limits.timeout_seconds}s"
            )

        if not sandbox_result.success:
            raise RuntimeError(
                f"Tool '{tool.name}' failed in sandbox: {sandbox_result.error}"
            )

        # Parse JSON output from subprocess
        output = sandbox_result.output.strip()
        if output:
            try:
                parsed = json.loads(output)
                return parsed.get("result", output)
            except json.JSONDecodeError:
                return output
        return None
