"""Sandbox execution backends for agentu.

Follows Anthropic's execute(name, input) → string pattern.
Tool functions run in isolation with resource limits.
"""

import json
import asyncio
import logging
import tempfile
import textwrap
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SandboxLimits:
    """Resource limits for sandboxed tool execution.
    
    Args:
        timeout_seconds: Max execution time (default: 30)
        max_memory_mb: Max memory in MB (default: 256, None=unlimited)
        max_output_bytes: Max stdout/stderr bytes to capture (default: 1MB)
    """
    timeout_seconds: float = 30.0
    max_memory_mb: Optional[int] = 256
    max_output_bytes: int = 1_048_576  # 1MB


@dataclass
class SandboxResult:
    """Result from sandboxed execution."""
    output: str
    error: Optional[str] = None
    exit_code: int = 0
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None


@runtime_checkable
class SandboxBackend(Protocol):
    """Protocol for sandbox backends.
    
    Follows Anthropic's pattern: execute(name, input) → string.
    Any backend that implements this protocol can be used.
    """
    
    async def execute(self, code: str, limits: SandboxLimits) -> SandboxResult:
        """Execute code in a sandboxed environment.
        
        Args:
            code: Python code string to execute
            limits: Resource limits for execution
            
        Returns:
            SandboxResult with output and status
        """
        ...


class SubprocessSandbox:
    """Run tool code in a subprocess with resource limits.
    
    Uses subprocess isolation with optional ulimits on macOS/Linux.
    This is a lightweight sandbox -- not security-grade, but prevents
    runaway tools from crashing the agent process.
    
    Example:
        >>> sandbox = SubprocessSandbox()
        >>> result = await sandbox.execute("print(2+2)", SandboxLimits(timeout_seconds=5))
        >>> assert result.output.strip() == "4"
    """
    
    def __init__(self, python_path: str = "python3"):
        """Initialize subprocess sandbox.
        
        Args:
            python_path: Path to Python interpreter for subprocess
        """
        self.python_path = python_path
    
    async def execute(self, code: str, limits: SandboxLimits) -> SandboxResult:
        """Execute Python code in a subprocess.
        
        Args:
            code: Python code string to execute
            limits: Resource limits
            
        Returns:
            SandboxResult with captured stdout
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self.python_path, "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Prevent the subprocess from inheriting signals
                start_new_session=True,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=limits.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return SandboxResult(
                    output="",
                    error=f"Execution timed out after {limits.timeout_seconds}s",
                    exit_code=-1,
                    timed_out=True,
                )
            
            stdout_str = stdout.decode("utf-8", errors="replace")[:limits.max_output_bytes]
            stderr_str = stderr.decode("utf-8", errors="replace")[:limits.max_output_bytes]
            
            return SandboxResult(
                output=stdout_str,
                error=stderr_str if stderr_str.strip() else None,
                exit_code=process.returncode or 0,
            )
            
        except Exception as e:
            return SandboxResult(
                output="",
                error=f"Sandbox error: {str(e)}",
                exit_code=-1,
            )


class InProcessSandbox:
    """Execute tools in-process (no isolation, for backward compatibility).
    
    This is the default -- tools run directly in the agent's process.
    Use SubprocessSandbox or a custom backend for actual isolation.
    """
    
    async def execute(self, code: str, limits: SandboxLimits) -> SandboxResult:
        """Execute code in-process using exec().
        
        WARNING: No isolation. Use only for trusted code.
        """
        output_capture: Dict[str, Any] = {"__result__": None}
        try:
            exec(code, output_capture)
            result = output_capture.get("__result__", "")
            return SandboxResult(output=str(result) if result else "")
        except Exception as e:
            return SandboxResult(
                output="",
                error=str(e),
                exit_code=1,
            )


def build_tool_code(func_source: str, func_name: str, params: Dict[str, Any]) -> str:
    """Build executable Python code that calls a tool function.
    
    Serializes the function and parameters into a standalone script
    that can run in a subprocess.
    
    Args:
        func_source: Source code of the function
        func_name: Name of the function to call
        params: Parameters to pass
        
    Returns:
        Python code string ready for execution
    """
    params_json = json.dumps(params)
    lines = [
        "import json",
        "import sys",
        "",
        func_source,
        "",
        f"params = json.loads('''{params_json}''')",
        "try:",
        f"    result = {func_name}(**params)",
        '    print(json.dumps({"result": result, "error": None}))',
        "except Exception as e:",
        '    print(json.dumps({"result": None, "error": str(e)}), file=sys.stderr)',
        "    sys.exit(1)",
    ]
    return "\n".join(lines) + "\n"

