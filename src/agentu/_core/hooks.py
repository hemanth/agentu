"""Pre-tool, post-tool, and on-stop hooks for agentu agents.

Hooks let callers intercept tool execution at three points:
- **pre_tool**: before a tool runs — can allow, deny, or modify params
- **post_tool**: after a tool runs — can transform the result
- **on_stop**: when the agent loop finishes — can transform the final response

Both sync and async callables are supported for all hooks.
"""

import asyncio
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class HookAction(Enum):
    """Action returned by a pre-tool hook."""

    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"


@dataclass
class HookResult:
    """Result of a pre-tool hook invocation.

    Attributes:
        action: Whether to allow, deny, or modify the tool call.
        reason: Human-readable reason (used when action is DENY).
        modified_params: Replacement parameters (used when action is MODIFY).
    """

    action: HookAction = HookAction.ALLOW
    reason: Optional[str] = None
    modified_params: Optional[Dict[str, Any]] = None


class PermissionApprovalRequired(Exception):
    """Raised when a tool call requires explicit caller approval.

    The caller should catch this, present it to the user, then re-invoke
    the tool if approved.

    Attributes:
        tool_name: Name of the tool that requires approval.
        parameters: Parameters the tool was called with.
        reason: Why approval is needed.
    """

    def __init__(self, tool_name: str, parameters: Dict[str, Any], reason: str):
        self.tool_name = tool_name
        self.parameters = parameters
        self.reason = reason
        super().__init__(reason)


# ---------------------------------------------------------------------------
# Utility: call a sync or async callable uniformly
# ---------------------------------------------------------------------------

async def _call_maybe_async(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Invoke *fn* and ``await`` it if it returns a coroutine."""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


# ---------------------------------------------------------------------------
# HookSet — container for all hooks registered on an agent
# ---------------------------------------------------------------------------

# Type aliases for readability
PreToolHook = Callable  # (tool_name, parameters, context) -> HookResult
PostToolHook = Callable  # (tool_name, parameters, result) -> result
OnStopHook = Callable  # (final_response) -> final_response


class HookSet:
    """Aggregates pre-tool, post-tool, and on-stop hooks.

    Multiple hooks of each kind can be registered.  Pre-tool hooks run in
    order: the first DENY or MODIFY wins.  Post-tool and on-stop hooks
    chain: each receives the output of the previous one.
    """

    def __init__(
        self,
        pre_tool: Optional[List[PreToolHook]] = None,
        post_tool: Optional[List[PostToolHook]] = None,
        on_stop: Optional[List[OnStopHook]] = None,
    ):
        self.pre_tool_hooks: List[PreToolHook] = list(pre_tool or [])
        self.post_tool_hooks: List[PostToolHook] = list(post_tool or [])
        self.on_stop_hooks: List[OnStopHook] = list(on_stop or [])

    # -- pre-tool ----------------------------------------------------------

    async def run_pre_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> HookResult:
        """Run all pre-tool hooks.  First DENY or MODIFY wins."""
        ctx = context or {}
        for hook in self.pre_tool_hooks:
            result = await _call_maybe_async(hook, tool_name, parameters, ctx)
            if not isinstance(result, HookResult):
                # Tolerate hooks that return None (= allow)
                continue
            if result.action in (HookAction.DENY, HookAction.MODIFY):
                return result
        return HookResult(action=HookAction.ALLOW)

    # -- post-tool ---------------------------------------------------------

    async def run_post_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
    ) -> Any:
        """Run all post-tool hooks, chaining the result."""
        for hook in self.post_tool_hooks:
            result = await _call_maybe_async(hook, tool_name, parameters, result)
        return result

    # -- on-stop -----------------------------------------------------------

    async def run_on_stop(self, final_response: Any) -> Any:
        """Run all on-stop hooks, chaining the response."""
        for hook in self.on_stop_hooks:
            final_response = await _call_maybe_async(hook, final_response)
        return final_response
