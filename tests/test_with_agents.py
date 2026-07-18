"""Tests for with_agents() — agents as callable tools."""
import pytest
from agentu import Agent


def _make_agent(name, **kwargs):
    return Agent(name, model='test-model', auto_discover_rules=False, **kwargs)


class TestWithAgents:
    def test_returns_self(self):
        child = _make_agent("child")
        parent = _make_agent("parent")
        result = parent.with_agents([child])
        assert result is parent

    def test_adds_call_tool(self):
        child = _make_agent("researcher")
        parent = _make_agent("planner").with_agents([child])
        tool_names = {t.name for t in parent.tools}
        assert "call_researcher" in tool_names

    def test_multiple_agents(self):
        r = _make_agent("researcher")
        a = _make_agent("analyst")
        w = _make_agent("writer")
        parent = _make_agent("planner").with_agents([r, a, w])
        tool_names = {t.name for t in parent.tools}
        assert "call_researcher" in tool_names
        assert "call_analyst" in tool_names
        assert "call_writer" in tool_names

    def test_child_tools_in_description(self):
        def search(q: str) -> str:
            """Search the web."""
            return q
        child = _make_agent("researcher").with_tools([search])
        parent = _make_agent("planner").with_agents([child])
        tool = next(t for t in parent.tools if t.name == "call_researcher")
        assert "search" in tool.description.lower() or "search" in (tool.function.__doc__ or "").lower()

    def test_tracks_child_agents(self):
        r = _make_agent("researcher")
        a = _make_agent("analyst")
        parent = _make_agent("planner").with_agents([r, a])
        assert hasattr(parent, '_child_agents')
        assert len(parent._child_agents) == 2

    def test_chainable(self):
        r = _make_agent("researcher")
        w = _make_agent("writer")
        parent = (
            _make_agent("planner")
            .with_agents([r])
            .with_agents([w])
        )
        tool_names = {t.name for t in parent.tools}
        assert "call_researcher" in tool_names
        assert "call_writer" in tool_names
        assert len(parent._child_agents) == 2

    def test_chainable_with_other_builders(self):
        child = _make_agent("researcher")
        parent = (
            _make_agent("planner", enable_memory=True)
            .with_agents([child])
        )
        tool_names = {t.name for t in parent.tools}
        assert "call_researcher" in tool_names
        assert parent.memory_enabled

    @pytest.mark.asyncio
    async def test_tool_calls_child_infer(self):
        from unittest.mock import AsyncMock
        child = _make_agent("researcher")
        child.infer = AsyncMock(return_value="found 3 papers")
        parent = _make_agent("planner").with_agents([child])
        tool = next(t for t in parent.tools if t.name == "call_researcher")
        result = await tool.function(task="Find papers on AI")
        assert "3 papers" in result
        child.infer.assert_called_once_with("Find papers on AI")

    @pytest.mark.asyncio
    async def test_multiple_calls(self):
        from unittest.mock import AsyncMock
        r = _make_agent("researcher")
        w = _make_agent("writer")
        r.infer = AsyncMock(return_value="data gathered")
        w.infer = AsyncMock(return_value="report written")
        parent = _make_agent("planner").with_agents([r, w])

        r_tool = next(t for t in parent.tools if t.name == "call_researcher")
        w_tool = next(t for t in parent.tools if t.name == "call_writer")

        r_result = await r_tool.function(task="Research AI")
        w_result = await w_tool.function(task=f"Write report: {r_result}")

        assert "gathered" in r_result
        assert "written" in w_result

    def test_empty_agents_list(self):
        parent = _make_agent("planner").with_agents([])
        assert len(parent.tools) == 0

    def test_agent_with_no_tools(self):
        child = _make_agent("thinker")  # no tools
        parent = _make_agent("planner").with_agents([child])
        tool_names = {t.name for t in parent.tools}
        assert "call_thinker" in tool_names
