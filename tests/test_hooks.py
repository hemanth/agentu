"""Tests for hooks and permission modes."""

import asyncio
import pytest
from agentu import Agent, Tool, ToolPermission, HookAction, HookResult, HookSet, PermissionApprovalRequired
from agentu._core.hooks import _call_maybe_async


# ─── Helpers ───


def _make_agent(**kwargs) -> Agent:
    """Create an agent with a fixed model to avoid Ollama auto-detection."""
    return Agent("test", model="test-model", enable_memory=False, **kwargs)


def _readonly_tool():
    """A READONLY tool for testing."""
    def read_data(key: str) -> str:
        """Read data by key."""
        return f"data-{key}"
    return Tool(read_data, permission=ToolPermission.READONLY)


def _write_tool():
    """A WRITE tool for testing."""
    def save_data(key: str, value: str) -> str:
        """Save data."""
        return f"saved {key}={value}"
    return Tool(save_data, permission=ToolPermission.WRITE)


def _dangerous_tool():
    """A DANGEROUS tool for testing."""
    def delete_everything() -> str:
        """Delete everything."""
        return "deleted"
    return Tool(delete_everything, permission=ToolPermission.DANGEROUS)


# ═══════════════════════════════════════════════════════════════════════
# HookResult / HookAction basics
# ═══════════════════════════════════════════════════════════════════════


class TestHookDataclasses:
    def test_default_hook_result(self):
        r = HookResult()
        assert r.action == HookAction.ALLOW
        assert r.reason is None
        assert r.modified_params is None

    def test_deny_hook_result(self):
        r = HookResult(action=HookAction.DENY, reason="blocked")
        assert r.action == HookAction.DENY
        assert r.reason == "blocked"

    def test_modify_hook_result(self):
        r = HookResult(action=HookAction.MODIFY, modified_params={"x": 1})
        assert r.action == HookAction.MODIFY
        assert r.modified_params == {"x": 1}

    def test_hook_action_values(self):
        assert HookAction.ALLOW.value == "allow"
        assert HookAction.DENY.value == "deny"
        assert HookAction.MODIFY.value == "modify"


# ═══════════════════════════════════════════════════════════════════════
# _call_maybe_async
# ═══════════════════════════════════════════════════════════════════════


class TestCallMaybeAsync:
    @pytest.mark.asyncio
    async def test_sync_callable(self):
        def sync_fn(x):
            return x * 2
        result = await _call_maybe_async(sync_fn, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_async_callable(self):
        async def async_fn(x):
            return x * 3
        result = await _call_maybe_async(async_fn, 5)
        assert result == 15


# ═══════════════════════════════════════════════════════════════════════
# HookSet unit tests (no agent needed)
# ═══════════════════════════════════════════════════════════════════════


class TestHookSet:
    @pytest.mark.asyncio
    async def test_no_hooks_allows(self):
        hs = HookSet()
        result = await hs.run_pre_tool("any_tool", {}, {})
        assert result.action == HookAction.ALLOW

    @pytest.mark.asyncio
    async def test_pre_tool_deny(self):
        def deny_all(name, params, ctx):
            return HookResult(action=HookAction.DENY, reason="nope")
        hs = HookSet(pre_tool=[deny_all])
        result = await hs.run_pre_tool("tool", {}, {})
        assert result.action == HookAction.DENY
        assert result.reason == "nope"

    @pytest.mark.asyncio
    async def test_pre_tool_modify(self):
        def rewrite(name, params, ctx):
            return HookResult(action=HookAction.MODIFY, modified_params={"x": 99})
        hs = HookSet(pre_tool=[rewrite])
        result = await hs.run_pre_tool("tool", {"x": 1}, {})
        assert result.action == HookAction.MODIFY
        assert result.modified_params == {"x": 99}

    @pytest.mark.asyncio
    async def test_pre_tool_first_deny_wins(self):
        """When multiple hooks exist, first DENY wins."""
        calls = []
        def hook_a(name, params, ctx):
            calls.append("a")
            return HookResult(action=HookAction.DENY, reason="a denied")
        def hook_b(name, params, ctx):
            calls.append("b")
            return HookResult(action=HookAction.ALLOW)

        hs = HookSet(pre_tool=[hook_a, hook_b])
        result = await hs.run_pre_tool("tool", {}, {})
        assert result.action == HookAction.DENY
        # hook_b should never have been called
        assert calls == ["a"]

    @pytest.mark.asyncio
    async def test_pre_tool_allow_continues(self):
        """When a hook ALLOWs, next hook runs."""
        calls = []
        def hook_a(name, params, ctx):
            calls.append("a")
            return HookResult(action=HookAction.ALLOW)
        def hook_b(name, params, ctx):
            calls.append("b")
            return HookResult(action=HookAction.DENY, reason="b denied")

        hs = HookSet(pre_tool=[hook_a, hook_b])
        result = await hs.run_pre_tool("tool", {}, {})
        assert result.action == HookAction.DENY
        assert calls == ["a", "b"]

    @pytest.mark.asyncio
    async def test_pre_tool_tolerates_none_return(self):
        """Hooks that return None are treated as ALLOW."""
        def hook_no_return(name, params, ctx):
            pass  # returns None

        hs = HookSet(pre_tool=[hook_no_return])
        result = await hs.run_pre_tool("tool", {}, {})
        assert result.action == HookAction.ALLOW

    @pytest.mark.asyncio
    async def test_post_tool_chains(self):
        """Multiple post-tool hooks chain their results."""
        def double(name, params, result):
            return result * 2
        def add_ten(name, params, result):
            return result + 10

        hs = HookSet(post_tool=[double, add_ten])
        result = await hs.run_post_tool("tool", {}, 5)
        # 5 -> double -> 10 -> add_ten -> 20
        assert result == 20

    @pytest.mark.asyncio
    async def test_on_stop_chains(self):
        """Multiple on-stop hooks chain their results."""
        def add_key(response):
            response["extra"] = True
            return response

        hs = HookSet(on_stop=[add_key])
        result = await hs.run_on_stop({"result": "ok"})
        assert result == {"result": "ok", "extra": True}

    @pytest.mark.asyncio
    async def test_async_pre_tool_hook(self):
        """Async pre-tool hooks work."""
        async def async_deny(name, params, ctx):
            await asyncio.sleep(0)  # simulate async work
            return HookResult(action=HookAction.DENY, reason="async denied")

        hs = HookSet(pre_tool=[async_deny])
        result = await hs.run_pre_tool("tool", {}, {})
        assert result.action == HookAction.DENY

    @pytest.mark.asyncio
    async def test_async_post_tool_hook(self):
        """Async post-tool hooks work."""
        async def async_transform(name, params, result):
            await asyncio.sleep(0)
            return f"async-{result}"

        hs = HookSet(post_tool=[async_transform])
        result = await hs.run_post_tool("tool", {}, "hello")
        assert result == "async-hello"

    @pytest.mark.asyncio
    async def test_async_on_stop_hook(self):
        """Async on-stop hooks work."""
        async def async_stop(response):
            await asyncio.sleep(0)
            response["async"] = True
            return response

        hs = HookSet(on_stop=[async_stop])
        result = await hs.run_on_stop({"result": "done"})
        assert result["async"] is True


# ═══════════════════════════════════════════════════════════════════════
# Agent.with_hooks() builder
# ═══════════════════════════════════════════════════════════════════════


class TestWithHooks:
    def test_returns_self(self):
        agent = _make_agent()
        result = agent.with_hooks(pre_tool=lambda n, p, c: HookResult())
        assert result is agent

    def test_hooks_initialized_lazily(self):
        agent = _make_agent()
        assert agent._hooks is None
        agent.with_hooks(pre_tool=lambda n, p, c: HookResult())
        assert agent._hooks is not None
        assert len(agent._hooks.pre_tool_hooks) == 1

    def test_multiple_calls_additive(self):
        agent = _make_agent()
        agent.with_hooks(pre_tool=lambda n, p, c: HookResult())
        agent.with_hooks(pre_tool=lambda n, p, c: HookResult())
        assert len(agent._hooks.pre_tool_hooks) == 2

    def test_all_hook_types(self):
        agent = _make_agent()
        agent.with_hooks(
            pre_tool=lambda n, p, c: HookResult(),
            post_tool=lambda n, p, r: r,
            on_stop=lambda r: r,
        )
        assert len(agent._hooks.pre_tool_hooks) == 1
        assert len(agent._hooks.post_tool_hooks) == 1
        assert len(agent._hooks.on_stop_hooks) == 1

    def test_list_hooks(self):
        agent = _make_agent()
        agent.with_hooks(pre_tool=[
            lambda n, p, c: HookResult(),
            lambda n, p, c: HookResult(),
        ])
        assert len(agent._hooks.pre_tool_hooks) == 2

    def test_chaining(self):
        """Builder pattern chains."""
        agent = (
            _make_agent()
            .with_hooks(pre_tool=lambda n, p, c: HookResult())
            .with_hooks(post_tool=lambda n, p, r: r)
        )
        assert len(agent._hooks.pre_tool_hooks) == 1
        assert len(agent._hooks.post_tool_hooks) == 1


# ═══════════════════════════════════════════════════════════════════════
# Agent.with_permissions() builder
# ═══════════════════════════════════════════════════════════════════════


class TestWithPermissions:
    def test_returns_self(self):
        agent = _make_agent()
        result = agent.with_permissions(allow_dangerous=True)
        assert result is agent

    def test_default_mode_is_auto(self):
        agent = _make_agent()
        assert agent._permission_mode == "auto"

    def test_plan_mode_sets_mode(self):
        agent = _make_agent()
        agent.with_permissions(mode="plan")
        assert agent._permission_mode == "plan"

    def test_plan_mode_installs_hook(self):
        agent = _make_agent()
        agent.with_permissions(mode="plan")
        assert agent._hooks is not None
        assert len(agent._hooks.pre_tool_hooks) == 1

    def test_ask_writes_mode_sets_mode(self):
        agent = _make_agent()
        agent.with_permissions(mode="ask-writes")
        assert agent._permission_mode == "ask-writes"

    def test_invalid_mode_raises(self):
        agent = _make_agent()
        with pytest.raises(ValueError, match="Invalid permission mode"):
            agent.with_permissions(mode="invalid")

    def test_can_use_tool_callback_set(self):
        cb = lambda n, p, c: "allow"
        agent = _make_agent()
        agent.with_permissions(can_use_tool=cb)
        assert agent._can_use_tool is cb

    def test_backwards_compat_allow_dangerous_only(self):
        """Old-style call with just allow_dangerous still works."""
        agent = _make_agent()
        agent.with_permissions(allow_dangerous=True)
        assert agent._allow_dangerous is True
        assert agent._permission_mode == "auto"


# ═══════════════════════════════════════════════════════════════════════
# Agent.call() integration with hooks
# ═══════════════════════════════════════════════════════════════════════


class TestCallWithHooks:
    @pytest.mark.asyncio
    async def test_pre_tool_deny_returns_denial_string(self):
        """A denied tool call returns DENIED: <reason> to the model."""
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])

        def deny_hook(name, params, ctx):
            return HookResult(action=HookAction.DENY, reason="not allowed")

        agent.with_hooks(pre_tool=deny_hook)
        result = await agent.call("read_data", {"key": "test"})
        assert "DENIED" in result
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_pre_tool_modify_changes_params(self):
        """Modified params are used for tool execution."""
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])

        def modify_hook(name, params, ctx):
            return HookResult(
                action=HookAction.MODIFY,
                modified_params={"key": "modified"},
            )

        agent.with_hooks(pre_tool=modify_hook)
        result = await agent.call("read_data", {"key": "original"})
        assert result == "data-modified"

    @pytest.mark.asyncio
    async def test_pre_tool_allow_normal_execution(self):
        """ALLOW hook lets tool execute normally."""
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])

        def allow_hook(name, params, ctx):
            return HookResult(action=HookAction.ALLOW)

        agent.with_hooks(pre_tool=allow_hook)
        result = await agent.call("read_data", {"key": "test"})
        assert result == "data-test"

    @pytest.mark.asyncio
    async def test_post_tool_transforms_result(self):
        """Post-tool hook transforms the result."""
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])

        def uppercase_hook(name, params, result):
            return result.upper()

        agent.with_hooks(post_tool=uppercase_hook)
        result = await agent.call("read_data", {"key": "test"})
        assert result == "DATA-TEST"

    @pytest.mark.asyncio
    async def test_no_hooks_normal_execution(self):
        """Without hooks, tools execute normally (backwards-compat)."""
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])
        result = await agent.call("read_data", {"key": "test"})
        assert result == "data-test"

    @pytest.mark.asyncio
    async def test_async_hooks_work(self):
        """Async pre and post hooks work through agent.call()."""
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])

        async def async_pre(name, params, ctx):
            await asyncio.sleep(0)
            return HookResult(action=HookAction.MODIFY, modified_params={"key": "async"})

        async def async_post(name, params, result):
            await asyncio.sleep(0)
            return f"[{result}]"

        agent.with_hooks(pre_tool=async_pre, post_tool=async_post)
        result = await agent.call("read_data", {"key": "orig"})
        assert result == "[data-async]"

    @pytest.mark.asyncio
    async def test_hook_context_has_tool_object(self):
        """The context dict passed to pre_tool contains the Tool object."""
        captured = {}

        def capture_hook(name, params, ctx):
            captured.update(ctx)
            return HookResult()

        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])
        agent.with_hooks(pre_tool=capture_hook)
        await agent.call("read_data", {"key": "test"})

        assert "tool" in captured
        assert isinstance(captured["tool"], Tool)
        assert captured["tool"].name == "read_data"


# ═══════════════════════════════════════════════════════════════════════
# Agent.call() integration with permission modes
# ═══════════════════════════════════════════════════════════════════════


class TestCallWithPermissions:
    @pytest.mark.asyncio
    async def test_dangerous_blocked_by_default(self):
        agent = _make_agent()
        agent.with_tools(tools=[_dangerous_tool()])
        with pytest.raises(PermissionError, match="DANGEROUS"):
            await agent.call("delete_everything", {})

    @pytest.mark.asyncio
    async def test_dangerous_allowed_with_flag(self):
        agent = _make_agent()
        agent.with_tools(tools=[_dangerous_tool()])
        agent.with_permissions(allow_dangerous=True)
        result = await agent.call("delete_everything", {})
        assert result == "deleted"

    @pytest.mark.asyncio
    async def test_plan_mode_blocks_write(self):
        agent = _make_agent()
        agent.with_tools(tools=[_write_tool()])
        agent.with_permissions(mode="plan")
        result = await agent.call("save_data", {"key": "k", "value": "v"})
        assert "DENIED" in result
        assert "plan mode" in result

    @pytest.mark.asyncio
    async def test_plan_mode_allows_readonly(self):
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])
        agent.with_permissions(mode="plan")
        result = await agent.call("read_data", {"key": "test"})
        assert result == "data-test"

    @pytest.mark.asyncio
    async def test_ask_writes_raises_for_write_tools(self):
        agent = _make_agent()
        agent.with_tools(tools=[_write_tool()])
        agent.with_permissions(mode="ask-writes")
        with pytest.raises(PermissionApprovalRequired) as exc_info:
            await agent.call("save_data", {"key": "k", "value": "v"})
        assert exc_info.value.tool_name == "save_data"
        assert exc_info.value.parameters == {"key": "k", "value": "v"}

    @pytest.mark.asyncio
    async def test_ask_writes_allows_readonly(self):
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])
        agent.with_permissions(mode="ask-writes")
        result = await agent.call("read_data", {"key": "test"})
        assert result == "data-test"

    @pytest.mark.asyncio
    async def test_can_use_tool_deny(self):
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])
        agent.with_permissions(can_use_tool=lambda n, p, c: "deny")
        with pytest.raises(PermissionError, match="denied by can_use_tool"):
            await agent.call("read_data", {"key": "test"})

    @pytest.mark.asyncio
    async def test_can_use_tool_ask(self):
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])
        agent.with_permissions(can_use_tool=lambda n, p, c: "ask")
        with pytest.raises(PermissionApprovalRequired):
            await agent.call("read_data", {"key": "test"})

    @pytest.mark.asyncio
    async def test_can_use_tool_allow(self):
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])
        agent.with_permissions(can_use_tool=lambda n, p, c: "allow")
        result = await agent.call("read_data", {"key": "test"})
        assert result == "data-test"

    @pytest.mark.asyncio
    async def test_async_can_use_tool(self):
        async def async_cb(name, params, ctx):
            await asyncio.sleep(0)
            return "deny"

        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])
        agent.with_permissions(can_use_tool=async_cb)
        with pytest.raises(PermissionError):
            await agent.call("read_data", {"key": "test"})


# ═══════════════════════════════════════════════════════════════════════
# PermissionApprovalRequired exception
# ═══════════════════════════════════════════════════════════════════════


class TestPermissionApprovalRequired:
    def test_attributes(self):
        exc = PermissionApprovalRequired(
            tool_name="write_file",
            parameters={"path": "/tmp/x"},
            reason="needs approval",
        )
        assert exc.tool_name == "write_file"
        assert exc.parameters == {"path": "/tmp/x"}
        assert str(exc) == "needs approval"

    def test_is_exception(self):
        exc = PermissionApprovalRequired("t", {}, "r")
        assert isinstance(exc, Exception)


# ═══════════════════════════════════════════════════════════════════════
# Backwards compatibility
# ═══════════════════════════════════════════════════════════════════════


class TestBackwardsCompatibility:
    def test_no_hooks_by_default(self):
        agent = _make_agent()
        assert agent._hooks is None

    def test_default_permission_mode_is_auto(self):
        agent = _make_agent()
        assert agent._permission_mode == "auto"
        assert agent._can_use_tool is None

    @pytest.mark.asyncio
    async def test_call_works_without_hooks_or_permissions(self):
        """Default agent.call() works exactly like before."""
        agent = _make_agent()
        agent.with_tools(tools=[_write_tool()])
        result = await agent.call("save_data", {"key": "k", "value": "v"})
        assert result == "saved k=v"

    def test_with_permissions_old_api(self):
        """Old API: with_permissions(allow_dangerous=True)."""
        agent = _make_agent()
        ret = agent.with_permissions(allow_dangerous=True)
        assert ret is agent
        assert agent._allow_dangerous is True

    def test_chaining_hooks_and_permissions(self):
        """with_hooks and with_permissions chain properly."""
        agent = (
            _make_agent()
            .with_hooks(pre_tool=lambda n, p, c: HookResult())
            .with_permissions(allow_dangerous=True)
        )
        assert agent._hooks is not None
        assert agent._allow_dangerous is True


# ═══════════════════════════════════════════════════════════════════════
# Composition: hooks + permissions together
# ═══════════════════════════════════════════════════════════════════════


class TestComposition:
    @pytest.mark.asyncio
    async def test_plan_mode_with_custom_hook(self):
        """Plan mode + custom hook both run; plan mode denies WRITE first."""
        call_log = []

        def log_hook(name, params, ctx):
            call_log.append(name)
            return HookResult()  # allow

        agent = _make_agent()
        agent.with_tools(tools=[_write_tool(), _readonly_tool()])
        agent.with_hooks(pre_tool=log_hook)
        agent.with_permissions(mode="plan")

        # WRITE tool: plan hook fires first (denies) before log_hook runs
        result = await agent.call("save_data", {"key": "k", "value": "v"})
        assert "DENIED" in result

        # READONLY tool: both hooks fire, plan allows, log allows
        call_log.clear()
        result = await agent.call("read_data", {"key": "test"})
        assert result == "data-test"
        assert "read_data" in call_log

    @pytest.mark.asyncio
    async def test_pre_and_post_hooks_together(self):
        """Pre modifies params, post transforms result."""
        agent = _make_agent()
        agent.with_tools(tools=[_readonly_tool()])

        def modify_pre(name, params, ctx):
            return HookResult(action=HookAction.MODIFY, modified_params={"key": "hooked"})

        def transform_post(name, params, result):
            return f"POST({result})"

        agent.with_hooks(pre_tool=modify_pre, post_tool=transform_post)
        result = await agent.call("read_data", {"key": "original"})
        assert result == "POST(data-hooked)"
