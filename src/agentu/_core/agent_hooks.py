"""HooksMixin – hook and permission methods extracted from Agent."""

import logging
from typing import Dict, Any, Optional, Union, List, Callable

from .hooks import (
    HookAction, HookResult, HookSet,
    PreToolHook, PostToolHook, OnStopHook,
)
from .tools import ToolPermission

logger = logging.getLogger(__name__)


class HooksMixin:
    """Mixin providing lifecycle hooks and permission management for the Agent class.

    Methods here assume they are mixed into an Agent instance that has
    ``_hooks``, ``_permission_mode``, ``_can_use_tool``, and
    ``_allow_dangerous`` attributes.
    """

    def with_permissions(
        self,
        mode: str = "auto",
        can_use_tool: Optional[Callable] = None,
        allow_dangerous: bool = False,
    ) -> 'Agent':
        """Configure tool permission policy.

        Permission modes:
            - ``"auto"``: All tools execute freely (current default behaviour).
            - ``"plan"``: WRITE and DANGEROUS tools are blocked. The agent can
              only call READONLY tools.
            - ``"ask-writes"``: WRITE tools raise
              :class:`PermissionApprovalRequired` so the caller can prompt
              the user before allowing execution.

        The ``plan`` mode is implemented as a built-in *pre_tool* hook that
        denies WRITE tools, so it composes naturally with custom hooks.

        Args:
            mode: Permission mode — ``"auto"``, ``"plan"``, or ``"ask-writes"``.
            can_use_tool: Optional async/sync callback
                ``(tool_name, params, context) -> 'allow' | 'deny' | 'ask'``.
                Overrides the default mode logic for individual calls.
            allow_dangerous: If True, allow DANGEROUS tools to execute.

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("bot").with_permissions(allow_dangerous=True)
            >>> agent = Agent("bot").with_permissions(mode="plan")
            >>> agent = Agent("bot").with_permissions(
            ...     mode="ask-writes",
            ...     can_use_tool=lambda name, params, ctx: "allow",
            ... )
        """
        if mode not in ("auto", "plan", "ask-writes"):
            raise ValueError(f"Invalid permission mode: {mode!r}. Use 'auto', 'plan', or 'ask-writes'.")
        self._permission_mode = mode
        self._can_use_tool = can_use_tool
        self._allow_dangerous = allow_dangerous

        # 'plan' mode installs a pre_tool hook that denies WRITE tools
        if mode == "plan":
            def _plan_mode_hook(tool_name: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> HookResult:
                tool_obj = context.get("tool")
                if tool_obj and tool_obj.permission in (ToolPermission.WRITE, ToolPermission.DANGEROUS):
                    return HookResult(
                        action=HookAction.DENY,
                        reason=f"Tool '{tool_name}' is blocked in plan mode (permission={tool_obj.permission.value}). "
                               f"Only READONLY tools are allowed.",
                    )
                return HookResult(action=HookAction.ALLOW)

            self.with_hooks(pre_tool=_plan_mode_hook)

        return self

    def with_hooks(
        self,
        pre_tool: Optional[Union[PreToolHook, List[PreToolHook]]] = None,
        post_tool: Optional[Union[PostToolHook, List[PostToolHook]]] = None,
        on_stop: Optional[Union[OnStopHook, List[OnStopHook]]] = None,
    ) -> 'Agent':
        """Register lifecycle hooks on this agent.

        Hooks can be sync or async callables.  Multiple calls to
        ``with_hooks`` are additive — hooks accumulate.

        Args:
            pre_tool: Called **before** each tool execution.
                Signature: ``(tool_name, parameters, context) -> HookResult``
                *context* contains ``{"tool": <Tool object>}``.
                Return ``HookResult(action=HookAction.DENY, reason=…)`` to
                block the call, or ``HookResult(action=HookAction.MODIFY,
                modified_params=…)`` to rewrite parameters.
            post_tool: Called **after** each tool execution.
                Signature: ``(tool_name, parameters, result) -> result``
                The returned value replaces the tool result.
            on_stop: Called when the agent loop finishes.
                Signature: ``(final_response) -> final_response``

        Returns:
            Self for method chaining

        Example:
            >>> def audit(tool_name, params, ctx):
            ...     print(f"Calling {tool_name}")
            ...     return HookResult()  # allow
            >>> agent = Agent("bot").with_hooks(pre_tool=audit)
        """
        if self._hooks is None:
            self._hooks = HookSet()

        # Normalise single callable into a list
        if pre_tool is not None:
            hooks = pre_tool if isinstance(pre_tool, list) else [pre_tool]
            self._hooks.pre_tool_hooks.extend(hooks)
        if post_tool is not None:
            hooks = post_tool if isinstance(post_tool, list) else [post_tool]
            self._hooks.post_tool_hooks.extend(hooks)
        if on_stop is not None:
            hooks = on_stop if isinstance(on_stop, list) else [on_stop]
            self._hooks.on_stop_hooks.extend(hooks)

        return self
