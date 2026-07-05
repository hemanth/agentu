"""Sandbox execution backends for agentu.

Follows Anthropic's execute(name, input) → string pattern.
Tool functions run in isolation with resource limits.
"""

import json
import os
import platform
import asyncio
import logging
import tempfile
import textwrap
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SandboxLimits:
    """Resource limits for sandboxed tool execution.
    
    Args:
        timeout_seconds: Max execution time (default: 30)
        max_memory_mb: Max memory in MB (default: 256, None=unlimited)
        max_output_bytes: Max stdout/stderr bytes to capture (default: 1MB)
        allow_network: If False, block outbound HTTP(S) via proxy env vars
            (default: True for backwards compatibility)
        network_allowlist: When allow_network is False, list of domains/IPs
            that bypass the blocking proxy (set via NO_PROXY). Ignored when
            allow_network is True.
    """
    timeout_seconds: float = 30.0
    max_memory_mb: Optional[int] = 256
    max_output_bytes: int = 1_048_576  # 1MB
    allow_network: bool = True
    network_allowlist: Optional[List[str]] = None


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
            # Prepend memory limit preamble on POSIX systems
            exec_code = code
            if limits.max_memory_mb is not None and platform.system() != "Windows":
                max_bytes = limits.max_memory_mb * 1024 * 1024
                preamble = (
                    "import resource, platform as _plat\n"
                    "try:\n"
                    "    _rlimit = resource.RLIMIT_AS if hasattr(resource, 'RLIMIT_AS') "
                    "and _plat.system() != 'Darwin' else resource.RLIMIT_RSS\n"
                    f"    _soft, _hard = resource.getrlimit(_rlimit)\n"
                    f"    _desired = {max_bytes}\n"
                    "    _new_hard = _hard if _hard != resource.RLIM_INFINITY else _desired\n"
                    "    _new_soft = min(_desired, _new_hard)\n"
                    "    resource.setrlimit(_rlimit, (_new_soft, _new_hard))\n"
                    "except (ValueError, OSError):\n"
                    "    pass\n"
                )
                exec_code = preamble + code

            # Build subprocess environment with optional egress controls
            env = self._build_env(limits)

            process = await asyncio.create_subprocess_exec(
                self.python_path, "-c", exec_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Prevent the subprocess from inheriting signals
                start_new_session=True,
                env=env,
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


    @staticmethod
    def _build_env(limits: SandboxLimits) -> Optional[Dict[str, str]]:
        """Build subprocess environment dict with optional egress controls.

        When ``limits.allow_network`` is *True* (the default), no
        modifications are made and ``None`` is returned so the subprocess
        inherits the parent environment as-is.

        When ``limits.allow_network`` is *False*, the returned env dict
        sets ``HTTP_PROXY`` / ``HTTPS_PROXY`` to a dummy address so that
        well-behaved HTTP clients will fail rather than reaching the
        network.  If ``limits.network_allowlist`` is provided, those
        domains are placed in ``NO_PROXY`` so they still work.

        Returns:
            Modified environment dict, or ``None`` to inherit as-is.
        """
        if limits.allow_network:
            return None  # inherit parent env unchanged

        env = os.environ.copy()
        blocked_proxy = "http://blocked"
        env["HTTP_PROXY"] = blocked_proxy
        env["HTTPS_PROXY"] = blocked_proxy
        env["http_proxy"] = blocked_proxy
        env["https_proxy"] = blocked_proxy

        if limits.network_allowlist:
            no_proxy = ",".join(limits.network_allowlist)
        else:
            no_proxy = ""
        env["NO_PROXY"] = no_proxy
        env["no_proxy"] = no_proxy

        logger.info(
            "Sandbox egress blocked (NO_PROXY=%s)",
            no_proxy or "<none>",
        )
        return env


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

