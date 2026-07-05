"""Tests for multi-turn tool iteration in infer()."""

import asyncio
import pytest
from agentu._core.agent import Agent


class TestMultiTurnInfer:
    """Test the multi-turn tool-calling loop in infer()."""

    def _make_agent(self, **kwargs):
        """Create a test agent with mocked LLM."""
        return Agent("test", api_base="http://localhost:11434/v1", **kwargs)

    @pytest.mark.asyncio
    async def test_single_tool_then_stop(self):
        """Model calls one tool, then returns no tool to stop."""
        agent = self._make_agent(max_turns=5)

        def calc(x, y):
            return x + y

        agent.with_tools([calc])

        call_count = 0

        async def mock_evaluate(context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "selected_tool": "calc",
                    "parameters": {"x": 2, "y": 3},
                    "reasoning": "Adding numbers"
                }
            else:
                return {
                    "selected_tool": None,
                    "parameters": {},
                    "reasoning": "Task complete"
                }

        agent.evaluate_tool_use = mock_evaluate

        result = await agent.infer("Add 2 and 3")
        assert result["tool_used"] == "calc"
        assert result["result"] == 5

    @pytest.mark.asyncio
    async def test_multi_turn_two_tools(self):
        """Model calls two tools in sequence, then stops."""
        agent = self._make_agent(max_turns=5)

        def step1():
            return "step1_done"

        def step2():
            return "step2_done"

        agent.with_tools([step1, step2])

        call_count = 0

        async def mock_evaluate(context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "selected_tool": "step1",
                    "parameters": {},
                    "reasoning": "Doing step 1"
                }
            elif call_count == 2:
                return {
                    "selected_tool": "step2",
                    "parameters": {},
                    "reasoning": "Doing step 2"
                }
            else:
                return {
                    "selected_tool": None,
                    "parameters": {},
                    "reasoning": "All done"
                }

        agent.evaluate_tool_use = mock_evaluate

        result = await agent.infer("Do both steps")
        # Model called two tools, then stopped — 3 evaluate calls total
        assert call_count == 3
        # Last turn result should be the final response
        assert result["tool_used"] == "step2"
        assert result["result"] == "step2_done"

    @pytest.mark.asyncio
    async def test_text_response_completion(self):
        """Model returns text_response to indicate it's done."""
        agent = self._make_agent(max_turns=5)

        def tool1():
            return "result1"

        agent.with_tools([tool1])

        call_count = 0

        async def mock_evaluate(context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "selected_tool": "tool1",
                    "parameters": {},
                    "reasoning": "Using tool"
                }
            else:
                return {
                    "text_response": "I've completed the task using tool1.",
                    "selected_tool": None,
                    "parameters": {},
                }

        agent.evaluate_tool_use = mock_evaluate

        result = await agent.infer("Do something")
        assert "text_response" in result
        assert result["text_response"] == "I've completed the task using tool1."
        assert len(result["history"]) == 1

    @pytest.mark.asyncio
    async def test_max_turns_reached(self):
        """When max_turns is exhausted, return error with history."""
        agent = self._make_agent(max_turns=2)

        def tool1():
            return "result"

        agent.with_tools([tool1])

        async def mock_evaluate(context):
            return {
                "selected_tool": "tool1",
                "parameters": {},
                "reasoning": "Keep going"
            }

        agent.evaluate_tool_use = mock_evaluate

        result = await agent.infer("Infinite loop task")
        assert "error" in result
        assert "Max turns (2) reached" in result["error"]
        assert len(result["history"]) == 2
        # Should include last result
        assert result.get("tool_used") == "tool1"
        assert result.get("result") == "result"

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self):
        """Model requests multiple tool calls at once via additional_tools."""
        agent = self._make_agent(max_turns=5)

        results_log = []

        async def fetch_a():
            results_log.append("a")
            return "data_a"

        async def fetch_b():
            results_log.append("b")
            return "data_b"

        agent.with_tools([fetch_a, fetch_b])

        call_count = 0

        async def mock_evaluate(context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "selected_tool": "fetch_a",
                    "parameters": {},
                    "additional_tools": [
                        {"tool": "fetch_b", "parameters": {}}
                    ],
                    "reasoning": "Fetching both in parallel"
                }
            else:
                return {
                    "selected_tool": None,
                    "parameters": {},
                    "reasoning": "Done"
                }

        agent.evaluate_tool_use = mock_evaluate

        result = await agent.infer("Fetch both datasets")
        # Both tools should have been called
        assert "a" in results_log
        assert "b" in results_log
        assert call_count == 2  # one parallel call turn + one stop turn

    @pytest.mark.asyncio
    async def test_error_in_tool_continues_loop(self):
        """Tool errors don't stop the loop — model sees error and reacts."""
        agent = self._make_agent(max_turns=5)

        def failing_tool():
            raise ValueError("Something went wrong")

        def recovery_tool():
            return "recovered"

        agent.with_tools([failing_tool, recovery_tool])

        call_count = 0

        async def mock_evaluate(context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "selected_tool": "failing_tool",
                    "parameters": {},
                    "reasoning": "Try this"
                }
            elif call_count == 2:
                return {
                    "selected_tool": "recovery_tool",
                    "parameters": {},
                    "reasoning": "First one failed, try recovery"
                }
            else:
                return {
                    "selected_tool": None,
                    "parameters": {},
                    "reasoning": "Done"
                }

        agent.evaluate_tool_use = mock_evaluate

        result = await agent.infer("Try to do something")
        assert call_count == 3
        assert result["tool_used"] == "recovery_tool"
        assert result["result"] == "recovered"

    @pytest.mark.asyncio
    async def test_search_tools_continues(self):
        """search_tools always continues to next turn."""
        agent = self._make_agent(max_turns=5)

        def real_tool():
            return "done"

        agent.with_tools([real_tool])

        call_count = 0

        async def mock_evaluate(context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "selected_tool": "search_tools",
                    "parameters": {"query": "find stuff"},
                    "reasoning": "Searching"
                }
            elif call_count == 2:
                return {
                    "selected_tool": "real_tool",
                    "parameters": {},
                    "reasoning": "Found it"
                }
            else:
                return {
                    "selected_tool": None,
                    "parameters": {},
                    "reasoning": "Done"
                }

        # Mock call method to handle search_tools
        original_call = agent.call

        async def mock_call(tool_name, params):
            if tool_name == "search_tools":
                return "Found: real_tool"
            return await original_call(tool_name, params)

        agent.call = mock_call
        agent.evaluate_tool_use = mock_evaluate

        result = await agent.infer("Find and use a tool")
        assert call_count == 3
        assert result["tool_used"] == "real_tool"

    @pytest.mark.asyncio
    async def test_no_tool_found_first_turn(self):
        """If no tool is found on first turn, return error."""
        agent = self._make_agent(max_turns=5)

        def some_tool():
            return "result"

        agent.with_tools([some_tool])

        async def mock_evaluate(context):
            return {
                "selected_tool": None,
                "parameters": {},
                "reasoning": "No matching tool"
            }

        agent.evaluate_tool_use = mock_evaluate

        result = await agent.infer("Something unrelated")
        assert "error" in result
        assert result["error"] == "No appropriate tool found"
