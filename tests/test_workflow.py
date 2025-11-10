"""Tests for workflow composition with >> and & operators."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from agentu import Agent
from agentu.workflow import Step, SequentialStep, ParallelStep


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock(spec=Agent)
    agent.name = "test_agent"
    agent.infer = AsyncMock(return_value={"result": "test_result"})
    return agent


@pytest.fixture
def researcher():
    """Create a mock researcher agent."""
    agent = Mock(spec=Agent)
    agent.name = "researcher"
    agent.infer = AsyncMock(return_value={"result": "research findings"})
    return agent


@pytest.fixture
def analyst():
    """Create a mock analyst agent."""
    agent = Mock(spec=Agent)
    agent.name = "analyst"
    agent.infer = AsyncMock(return_value={"result": "analysis complete"})
    return agent


@pytest.fixture
def writer():
    """Create a mock writer agent."""
    agent = Mock(spec=Agent)
    agent.name = "writer"
    agent.infer = AsyncMock(return_value={"result": "report written"})
    return agent


class TestStep:
    """Test Step class and basic functionality."""

    @pytest.mark.asyncio
    async def test_step_creation(self, mock_agent):
        """Test creating a basic step."""
        step = Step(mock_agent, "test task")
        assert step.agent == mock_agent
        assert step.task == "test task"

    @pytest.mark.asyncio
    async def test_step_execution_no_context(self, mock_agent):
        """Test executing step without context."""
        step = Step(mock_agent, "test task")
        result = await step.run()

        mock_agent.infer.assert_called_once_with("test task")
        assert result == {"result": "test_result"}

    @pytest.mark.asyncio
    async def test_step_execution_with_context(self, mock_agent):
        """Test executing step with previous context."""
        step = Step(mock_agent, "analyze this")
        result = await step.run(context="previous data")

        # Should auto-inject context
        call_args = mock_agent.infer.call_args[0][0]
        assert "analyze this" in call_args
        assert "previous data" in call_args
        assert "Context from previous step" in call_args

    @pytest.mark.asyncio
    async def test_step_with_lambda(self, mock_agent):
        """Test step with lambda function."""
        step = Step(mock_agent, lambda ctx: f"Process: {ctx['data']}")
        result = await step.run(context={"data": "test data"})

        mock_agent.infer.assert_called_once_with("Process: test data")

    @pytest.mark.asyncio
    async def test_step_error_handling(self, mock_agent):
        """Test step handles errors gracefully."""
        mock_agent.infer.side_effect = Exception("Test error")
        step = Step(mock_agent, "test task")
        result = await step.run()

        assert "error" in result
        assert result["error"] == "Test error"


class TestSequentialOperator:
    """Test >> (sequential) operator."""

    @pytest.mark.asyncio
    async def test_sequential_two_steps(self, researcher, analyst):
        """Test chaining two steps sequentially."""
        workflow = Step(researcher, "Find trends") >> Step(analyst, "Analyze")

        assert isinstance(workflow, SequentialStep)
        result = await workflow.run()

        # researcher called first
        researcher.infer.assert_called_once()
        # analyst called second with researcher's result
        analyst.infer.assert_called_once()
        call_args = analyst.infer.call_args[0][0]
        assert "research findings" in call_args

    @pytest.mark.asyncio
    async def test_sequential_three_steps(self, researcher, analyst, writer):
        """Test chaining three steps."""
        workflow = (
            Step(researcher, "Find trends")
            >> Step(analyst, "Analyze")
            >> Step(writer, "Write report")
        )

        result = await workflow.run()

        # All called in order
        researcher.infer.assert_called_once()
        analyst.infer.assert_called_once()
        writer.infer.assert_called_once()

    @pytest.mark.asyncio
    async def test_sequential_with_lambda(self, researcher, analyst):
        """Test sequential with lambda for precise control."""
        workflow = (
            Step(researcher, "Find trends")
            >> Step(analyst, lambda prev: f"Analyze: {prev['result']}")
        )

        result = await workflow.run()

        analyst.infer.assert_called_once_with("Analyze: research findings")

    @pytest.mark.asyncio
    async def test_sequential_with_initial_context(self, researcher, analyst):
        """Test sequential with initial context."""
        workflow = Step(researcher, "Research") >> Step(analyst, "Analyze")
        result = await workflow.run(context="initial input")

        # First step gets initial context
        call_args = researcher.infer.call_args[0][0]
        assert "initial input" in call_args


class TestParallelOperator:
    """Test & (parallel) operator."""

    @pytest.mark.asyncio
    async def test_parallel_two_steps(self, researcher):
        """Test running two steps in parallel."""
        agent1 = Mock(spec=Agent)
        agent1.name = "agent1"
        agent1.infer = AsyncMock(return_value={"result": "result1"})

        agent2 = Mock(spec=Agent)
        agent2.name = "agent2"
        agent2.infer = AsyncMock(return_value={"result": "result2"})

        workflow = Step(agent1, "task1") & Step(agent2, "task2")

        assert isinstance(workflow, ParallelStep)
        results = await workflow.run()

        # Both called
        agent1.infer.assert_called_once_with("task1")
        agent2.infer.assert_called_once_with("task2")

        # Results is a list
        assert len(results) == 2
        assert results[0] == {"result": "result1"}
        assert results[1] == {"result": "result2"}

    @pytest.mark.asyncio
    async def test_parallel_three_steps(self, researcher):
        """Test running three steps in parallel."""
        agents = []
        for i in range(3):
            agent = Mock(spec=Agent)
            agent.name = f"agent{i}"
            agent.infer = AsyncMock(return_value={"result": f"result{i}"})
            agents.append(agent)

        workflow = Step(agents[0], "task0") & Step(agents[1], "task1") & Step(agents[2], "task2")

        results = await workflow.run()

        assert len(results) == 3
        for i, agent in enumerate(agents):
            agent.infer.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_with_error(self, researcher):
        """Test parallel execution with one step failing."""
        agent1 = Mock(spec=Agent)
        agent1.name = "agent1"
        agent1.infer = AsyncMock(return_value={"result": "success"})

        agent2 = Mock(spec=Agent)
        agent2.name = "agent2"
        agent2.infer = AsyncMock(side_effect=Exception("Failed"))

        workflow = Step(agent1, "task1") & Step(agent2, "task2")
        results = await workflow.run()

        # First result is success
        assert results[0] == {"result": "success"}

        # Second result is error dict
        assert "error" in results[1]
        assert "Failed" in results[1]["error"]


class TestCombinedOperators:
    """Test combining >> and & operators."""

    @pytest.mark.asyncio
    async def test_parallel_then_sequential(self, researcher, analyst):
        """Test fan-out (parallel) then fan-in (sequential)."""
        agent1 = Mock(spec=Agent)
        agent1.name = "agent1"
        agent1.infer = AsyncMock(return_value={"result": "r1"})

        agent2 = Mock(spec=Agent)
        agent2.name = "agent2"
        agent2.infer = AsyncMock(return_value={"result": "r2"})

        workflow = (
            Step(agent1, "task1") & Step(agent2, "task2")
        ) >> Step(analyst, "merge results")

        result = await workflow.run()

        # Parallel steps called
        agent1.infer.assert_called_once()
        agent2.infer.assert_called_once()

        # Analyst receives both results
        analyst.infer.assert_called_once()
        call_args = analyst.infer.call_args[0][0]
        assert "r1" in call_args or "r2" in call_args

    @pytest.mark.asyncio
    async def test_sequential_then_parallel(self, researcher, analyst):
        """Test sequential then parallel."""
        agent1 = Mock(spec=Agent)
        agent1.name = "agent1"
        agent1.infer = AsyncMock(return_value={"result": "initial"})

        agent2 = Mock(spec=Agent)
        agent2.name = "agent2"
        agent2.infer = AsyncMock(return_value={"result": "branch1"})

        agent3 = Mock(spec=Agent)
        agent3.name = "agent3"
        agent3.infer = AsyncMock(return_value={"result": "branch2"})

        workflow = Step(agent1, "initial") >> (
            Step(agent2, "branch1") & Step(agent3, "branch2")
        )

        results = await workflow.run()

        # Initial step called first
        agent1.infer.assert_called_once()

        # Parallel steps called with initial result
        agent2.infer.assert_called_once()
        agent3.infer.assert_called_once()

        # Results is a list from parallel
        assert isinstance(results, list)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_complex_workflow(self, researcher, analyst, writer):
        """Test complex multi-stage workflow."""
        # 3 parallel research -> 1 analysis -> 1 report
        r1 = Mock(spec=Agent)
        r1.name = "r1"
        r1.infer = AsyncMock(return_value={"result": "AI"})

        r2 = Mock(spec=Agent)
        r2.name = "r2"
        r2.infer = AsyncMock(return_value={"result": "ML"})

        r3 = Mock(spec=Agent)
        r3.name = "r3"
        r3.infer = AsyncMock(return_value={"result": "Crypto"})

        workflow = (
            Step(r1, "AI") & Step(r2, "ML") & Step(r3, "Crypto")
        ) >> Step(analyst, "Analyze") >> Step(writer, "Write")

        result = await workflow.run()

        # All research called
        r1.infer.assert_called_once()
        r2.infer.assert_called_once()
        r3.infer.assert_called_once()

        # Analysis and writing called
        analyst.infer.assert_called_once()
        writer.infer.assert_called_once()


class TestAgentCallable:
    """Test Agent __call__ method for workflow creation."""

    def test_agent_callable_returns_step(self):
        """Test that calling agent returns a Step."""
        agent = Agent("test")
        step = agent("test task")

        assert isinstance(step, Step)
        assert step.agent == agent
        assert step.task == "test task"

    @pytest.mark.asyncio
    async def test_agent_callable_with_operators(self):
        """Test agent callable works with operators."""
        agent1 = Agent("agent1")
        agent2 = Agent("agent2")

        # Mock infer methods
        agent1.infer = AsyncMock(return_value={"result": "step1"})
        agent2.infer = AsyncMock(return_value={"result": "step2"})

        # Use callable syntax
        workflow = agent1("task1") >> agent2("task2")

        result = await workflow.run()

        agent1.infer.assert_called_once()
        agent2.infer.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_callable_parallel(self):
        """Test agent callable with parallel operator."""
        agent1 = Agent("agent1")
        agent2 = Agent("agent2")

        agent1.infer = AsyncMock(return_value={"result": "r1"})
        agent2.infer = AsyncMock(return_value={"result": "r2"})

        workflow = agent1("task1") & agent2("task2")
        results = await workflow.run()

        assert len(results) == 2
