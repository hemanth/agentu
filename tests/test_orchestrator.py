"""Tests for multi-agent orchestration."""

import pytest
from agentu import Agent, Tool
from agentu.orchestrator import (
    Orchestrator,
    AgentRole,
    ExecutionMode,
    Task,
    AgentCapability,
    Message,
    make_agent
)


@pytest.fixture
def basic_orchestrator():
    """Create a basic orchestrator for testing."""
    return Orchestrator(name="TestOrchestrator", execution_mode=ExecutionMode.SEQUENTIAL)


@pytest.fixture
def sample_agents():
    """Create sample agents for testing."""
    researcher = make_agent(
        name="Researcher",
        role=AgentRole.RESEARCHER,
        model="llama2",
        enable_memory=False
    )

    analyst = make_agent(
        name="Analyst",
        role=AgentRole.ANALYST,
        model="llama2",
        enable_memory=False
    )

    return {
        'researcher': (researcher),
        'analyst': (analyst)
    }


class TestAgentCapability:
    """Test AgentCapability class."""

    def test_capability_creation(self):
        """Test creating agent capabilities."""
        cap = AgentCapability(
            role=AgentRole.RESEARCHER,
            skills=["research", "search", "analyze"],
            priority=7,
            description="Research specialist"
        )

        assert cap.role == AgentRole.RESEARCHER
        assert len(cap.skills) == 3
        assert cap.priority == 7
        assert "research" in cap.skills

    def test_matches_task_perfect_match(self):
        """Test task matching with perfect match."""
        cap = AgentCapability(
            role=AgentRole.RESEARCHER,
            skills=["research", "search", "analyze"]
        )

        score = cap.matches_task(["research", "search"])
        assert score == 1.0  # 2/2 = 1.0

    def test_matches_task_partial_match(self):
        """Test task matching with partial match."""
        cap = AgentCapability(
            role=AgentRole.RESEARCHER,
            skills=["research", "search", "analyze"]
        )

        score = cap.matches_task(["research", "coding"])
        assert score == 0.5  # 1/2 = 0.5

    def test_matches_task_no_match(self):
        """Test task matching with no match."""
        cap = AgentCapability(
            role=AgentRole.RESEARCHER,
            skills=["research", "search", "analyze"]
        )

        score = cap.matches_task(["coding", "debugging"])
        assert score == 0.0  # 0/2 = 0.0

    def test_matches_task_empty_requirements(self):
        """Test task matching with empty requirements."""
        cap = AgentCapability(
            role=AgentRole.RESEARCHER,
            skills=["research", "search"]
        )

        score = cap.matches_task([])
        assert score == 0.5  # Default for no requirements


class TestMessage:
    """Test Message class."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(
            sender="Agent1",
            receiver="Agent2",
            content="Hello",
            message_type="info"
        )

        assert msg.sender == "Agent1"
        assert msg.receiver == "Agent2"
        assert msg.content == "Hello"
        assert msg.message_type == "info"

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(
            sender="Agent1",
            receiver="Agent2",
            content="Test message",
            message_type="task"
        )

        msg_dict = msg.to_dict()

        assert msg_dict['sender'] == "Agent1"
        assert msg_dict['receiver'] == "Agent2"
        assert msg_dict['content'] == "Test message"
        assert msg_dict['message_type'] == "task"

    def test_broadcast_message(self):
        """Test creating a broadcast message."""
        msg = Message(
            sender="Manager",
            receiver=None,  # Broadcast
            content="Broadcast message"
        )

        assert msg.receiver is None


class TestTask:
    """Test Task class."""

    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            description="Analyze data",
            required_skills=["analyze", "statistics"],
            metadata={"priority": "high"}
        )

        assert task.description == "Analyze data"
        assert len(task.required_skills) == 2
        assert task.status == "pending"
        assert task.assigned_agent is None
        assert task.metadata["priority"] == "high"

    def test_task_status_update(self):
        """Test updating task status (immutable pattern)."""
        task = Task(description="Test task")

        assert task.status == "pending"

        # Tasks are immutable, use with_status method
        in_progress_task = task.with_status("in_progress")
        assert in_progress_task.status == "in_progress"

        completed_task = in_progress_task.with_status("completed")
        assert completed_task.status == "completed"

        # Original task unchanged
        assert task.status == "pending"


class TestOrchestrator:
    """Test Orchestrator class."""

    def test_orchestrator_creation(self):
        """Test creating an orchestrator."""
        orch = Orchestrator(
            name="TestOrch",
            execution_mode=ExecutionMode.PARALLEL
        )

        assert orch.name == "TestOrch"
        assert orch.execution_mode == ExecutionMode.PARALLEL
        assert len(orch.agents) == 0
        assert len(orch.message_queue) == 0

    def test_add_agent(self, basic_orchestrator, sample_agents):
        """Test adding an agent."""
        researcher = sample_agents['researcher']

        basic_orchestrator.add_agent(researcher)

        assert "Researcher" in basic_orchestrator.agents
        assert "Researcher" in basic_orchestrator.agent_capabilities
        assert basic_orchestrator.agent_capabilities["Researcher"].role == AgentRole.RESEARCHER

    def test_add_agents(self, basic_orchestrator, sample_agents):
        """Test adding multiple agents at once."""
        researcher = sample_agents['researcher']
        analyst = sample_agents['analyst']

        basic_orchestrator.add_agents([researcher, analyst])

        assert "Researcher" in basic_orchestrator.agents
        assert "Analyst" in basic_orchestrator.agents
        assert len(basic_orchestrator.agents) == 2

    def test_remove_agent(self, basic_orchestrator, sample_agents):
        """Test removing an agent."""
        researcher = sample_agents['researcher']

        basic_orchestrator.add_agent(researcher)
        assert "Researcher" in basic_orchestrator.agents

        basic_orchestrator.remove_agent("Researcher")
        assert "Researcher" not in basic_orchestrator.agents
        assert "Researcher" not in basic_orchestrator.agent_capabilities

    def test_send_message(self, basic_orchestrator):
        """Test sending messages."""
        msg = Message(
            sender="Agent1",
            receiver="Agent2",
            content="Test message"
        )

        basic_orchestrator.send_message(msg)

        assert len(basic_orchestrator.message_queue) == 1
        assert basic_orchestrator.message_queue[0].content == "Test message"

    def test_get_messages_for_agent(self, basic_orchestrator):
        """Test retrieving messages for a specific agent."""
        msg1 = Message(sender="A1", receiver="A2", content="Message 1")
        msg2 = Message(sender="A1", receiver="A3", content="Message 2")
        msg3 = Message(sender="A2", receiver="A2", content="Message 3")

        basic_orchestrator.send_message(msg1)
        basic_orchestrator.send_message(msg2)
        basic_orchestrator.send_message(msg3)

        messages = basic_orchestrator.get_messages_for_agent("A2", clear=False)

        assert len(messages) == 2  # msg1 and msg3
        assert messages[0].content == "Message 1"
        assert messages[1].content == "Message 3"

    def test_get_messages_with_clear(self, basic_orchestrator):
        """Test retrieving and clearing messages."""
        msg1 = Message(sender="A1", receiver="A2", content="Message 1")
        msg2 = Message(sender="A1", receiver="A2", content="Message 2")

        basic_orchestrator.send_message(msg1)
        basic_orchestrator.send_message(msg2)

        assert len(basic_orchestrator.message_queue) == 2

        messages = basic_orchestrator.get_messages_for_agent("A2", clear=True)

        assert len(messages) == 2
        assert len(basic_orchestrator.message_queue) == 0

    def test_route_task(self, basic_orchestrator, sample_agents):
        """Test task routing to appropriate agent."""
        researcher = sample_agents['researcher']
        analyst = sample_agents['analyst']

        basic_orchestrator.add_agent(researcher)
        basic_orchestrator.add_agent(analyst)

        # Task requiring research skills
        research_task = Task(
            description="Research AI trends",
            required_skills=["research", "search"]
        )

        agent_name = basic_orchestrator.route_task(research_task)
        assert agent_name == "Researcher"

        # Task requiring analysis skills
        analysis_task = Task(
            description="Analyze data",
            required_skills=["analyze", "statistics"]
        )

        agent_name = basic_orchestrator.route_task(analysis_task)
        assert agent_name == "Analyst"

    def test_route_task_no_agents(self, basic_orchestrator):
        """Test routing when no agents are registered."""
        task = Task(description="Test task")

        agent_name = basic_orchestrator.route_task(task)
        assert agent_name is None

    def test_get_stats(self, basic_orchestrator, sample_agents):
        """Test getting orchestrator statistics."""
        researcher = sample_agents['researcher']
        analyst = sample_agents['analyst']

        basic_orchestrator.add_agent(researcher)
        basic_orchestrator.add_agent(analyst)

        stats = basic_orchestrator.get_stats()

        assert stats['name'] == "TestOrchestrator"
        assert stats['registered_agents'] == 2
        assert "Researcher" in stats['agents']
        assert "Analyst" in stats['agents']
        assert stats['execution_mode'] == 'sequential'

    def test_reset(self, basic_orchestrator):
        """Test resetting orchestrator state."""
        # Add some state using proper methods
        basic_orchestrator.send_message(
            Message(sender="A1", receiver="A2", content="Test")
        )
        basic_orchestrator.shared_memory['test'] = 'value'

        assert len(basic_orchestrator.message_queue) > 0
        assert len(basic_orchestrator.shared_memory) > 0

        # Reset
        basic_orchestrator.reset()

        assert len(basic_orchestrator.message_queue) == 0
        assert len(basic_orchestrator.shared_memory) == 0
        assert len(basic_orchestrator.task_history) == 0


class TestCreateSpecializedAgent:
    """Test make_agent helper function."""

    def test_create_researcher_agent(self):
        """Test creating a researcher agent."""
        agent = make_agent(
            name="TestResearcher",
            role=AgentRole.RESEARCHER,
            model="llama2",
            enable_memory=False
        )

        assert agent.name == "TestResearcher"
        assert agent.model == "llama2"
        assert agent.role == "researcher"
        assert "research" in agent.skills
        assert "search" in agent.skills

    def test_create_coder_agent(self):
        """Test creating a coder agent."""
        agent = make_agent(
            name="TestCoder",
            role=AgentRole.CODER,
            model="llama2",
            enable_memory=False
        )

        assert agent.role == "coder"
        assert "code" in agent.skills
        assert "programming" in agent.skills

    def test_create_custom_skills(self):
        """Test creating agent with custom skills."""
        custom_skills = ["skill1", "skill2", "skill3"]

        agent = make_agent(
            name="CustomAgent",
            role=AgentRole.CUSTOM,
            model="llama2",
            skills=custom_skills,
            enable_memory=False
        )

        # Skills are stored as tuples (immutable)
        assert agent.skills == tuple(custom_skills)

    def test_agent_context_set(self):
        """Test that agent context is set based on role."""
        agent = make_agent(
            name="TestAgent",
            role=AgentRole.RESEARCHER,
            enable_memory=False
        )

        assert agent.context != ""
        assert "research" in agent.context.lower()


class TestExecutionModes:
    """Test different execution modes."""

    def test_sequential_mode_enum(self):
        """Test sequential execution mode enum."""
        assert ExecutionMode.SEQUENTIAL.value == "sequential"

    def test_parallel_mode_enum(self):
        """Test parallel execution mode enum."""
        assert ExecutionMode.PARALLEL.value == "parallel"

    def test_hierarchical_mode_enum(self):
        """Test hierarchical execution mode enum."""
        assert ExecutionMode.HIERARCHICAL.value == "hierarchical"

    def test_debate_mode_enum(self):
        """Test debate execution mode enum."""
        assert ExecutionMode.DEBATE.value == "debate"


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring actual agent execution."""

    @pytest.mark.skip(reason="Requires running Ollama instance")
    def test_full_orchestration_workflow(self):
        """Test complete orchestration workflow."""
        # Create orchestrator
        orch = Orchestrator(execution_mode=ExecutionMode.SEQUENTIAL)

        # Create agents
        researcher = make_agent(
            name="Researcher",
            role=AgentRole.RESEARCHER
        )

        analyst = make_agent(
            name="Analyst",
            role=AgentRole.ANALYST
        )

        # Register agents
        orch.register_agent(researcher)
        orch.register_agent(analyst)

        # Create tasks
        tasks = [
            Task(
                description="Research Python best practices",
                required_skills=["research", "search"]
            ),
            Task(
                description="Analyze the research findings",
                required_skills=["analyze", "evaluate"]
            )
        ]

        # Execute
        results = orch.execute(tasks)

        # Verify
        assert len(results) == 2
        assert all(r['status'] in ['completed', 'failed'] for r in results)

    @pytest.mark.skip(reason="Requires running Ollama instance")
    def test_parallel_execution_integration(self):
        """Test parallel execution with real agents."""
        orch = Orchestrator(execution_mode=ExecutionMode.PARALLEL)

        # Create multiple coder agents
        agents = []
        for i in range(3):
            agent, cap = make_agent(
                name=f"Coder{i}",
                role=AgentRole.CODER
            )
            agents.append((agent, cap))
            orch.register_agent(agent, cap)

        # Create parallel tasks
        tasks = [
            Task(description=f"Write a function to {task}", required_skills=["code"])
            for task in ["sort array", "reverse string", "find prime numbers"]
        ]

        # Execute in parallel
        results = orch.execute_task_parallel(tasks, max_workers=3)

        assert len(results) == 3
