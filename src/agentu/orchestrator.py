"""Multi-agent orchestration using functional programming principles."""

import logging
from typing import List, Dict, Any, Optional, Callable, Tuple, NamedTuple
from dataclasses import dataclass, field, replace
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
import json

from .agent import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Enums and immutable data structures

class AgentRole(Enum):
    """Predefined agent roles with specific capabilities."""
    RESEARCHER = "researcher"
    CODER = "coder"
    ANALYST = "analyst"
    PLANNER = "planner"
    CRITIC = "critic"
    WRITER = "writer"
    COORDINATOR = "coordinator"
    CUSTOM = "custom"


class ExecutionMode(Enum):
    """Execution modes for multi-agent tasks."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    DEBATE = "debate"


@dataclass(frozen=True)
class Message:
    """Immutable message for inter-agent communication."""
    sender: str
    receiver: Optional[str]
    content: str
    message_type: str = "info"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'content': self.content,
            'message_type': self.message_type,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


@dataclass(frozen=True)
class AgentCapability:
    """Immutable agent capability definition."""
    role: AgentRole
    skills: Tuple[str, ...] = ()
    priority: int = 5
    description: str = ""

    def matches_task(self, task_requirements: List[str]) -> float:
        """Calculate task match score (pure function).

        Returns:
            Score from 0.0 to 1.0
        """
        if not task_requirements:
            return 0.5

        skills_lower = tuple(s.lower() for s in self.skills)
        matches = sum(1 for req in task_requirements if req.lower() in skills_lower)
        return matches / len(task_requirements)


@dataclass(frozen=True)
class Task:
    """Immutable task definition."""
    description: str
    required_skills: Tuple[str, ...] = ()
    assigned_agent: Optional[str] = None
    result: Optional[Any] = None
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_agent(self, agent_name: str) -> 'Task':
        """Return new task with assigned agent."""
        return replace(self, assigned_agent=agent_name)

    def with_status(self, status: str) -> 'Task':
        """Return new task with updated status."""
        return replace(self, status=status)

    def with_result(self, result: Any) -> 'Task':
        """Return new task with result."""
        return replace(self, result=result, status="completed")

    def as_failed(self, error: str) -> 'Task':
        """Return new task marked as failed."""
        return replace(self, status="failed", result={"error": error})


@dataclass(frozen=True)
class OrchestratorState:
    """Immutable orchestrator state."""
    name: str
    execution_mode: ExecutionMode
    agents: Dict[str, Agent] = field(default_factory=dict)
    capabilities: Dict[str, AgentCapability] = field(default_factory=dict)
    messages: Tuple[Message, ...] = ()
    tasks: Tuple[Task, ...] = ()
    shared_data: Dict[str, Any] = field(default_factory=dict)

    def with_agent(self, name: str, agent: Agent, capability: AgentCapability) -> 'OrchestratorState':
        """Return new state with added agent."""
        return replace(
            self,
            agents={**self.agents, name: agent},
            capabilities={**self.capabilities, name: capability}
        )

    def without_agent(self, name: str) -> 'OrchestratorState':
        """Return new state with removed agent."""
        new_agents = {k: v for k, v in self.agents.items() if k != name}
        new_caps = {k: v for k, v in self.capabilities.items() if k != name}
        return replace(self, agents=new_agents, capabilities=new_caps)

    def with_message(self, message: Message) -> 'OrchestratorState':
        """Return new state with added message."""
        return replace(self, messages=(*self.messages, message))

    def with_task(self, task: Task) -> 'OrchestratorState':
        """Return new state with added task."""
        return replace(self, tasks=(*self.tasks, task))

    def with_shared_data(self, key: str, value: Any) -> 'OrchestratorState':
        """Return new state with updated shared data."""
        return replace(self, shared_data={**self.shared_data, key: value})

    def clear_messages_for(self, agent_name: str) -> 'OrchestratorState':
        """Return new state with messages cleared for agent."""
        remaining = tuple(m for m in self.messages if m.receiver != agent_name)
        return replace(self, messages=remaining)

    def reset(self) -> 'OrchestratorState':
        """Return new state with cleared transient data."""
        return replace(self, messages=(), tasks=(), shared_data={})


# Pure functions for orchestration logic

def calculate_agent_score(capability: AgentCapability, task: Task) -> float:
    """Calculate how well an agent matches a task (pure function)."""
    skill_match = capability.matches_task(list(task.required_skills))
    priority_weight = capability.priority / 10.0
    return skill_match * priority_weight


def find_best_agent(capabilities: Dict[str, AgentCapability], task: Task) -> Optional[str]:
    """Find the best agent for a task (pure function)."""
    if not capabilities:
        return None

    scores = {
        name: calculate_agent_score(cap, task)
        for name, cap in capabilities.items()
    }

    if not scores:
        return None

    best_agent = max(scores.items(), key=lambda x: x[1])
    return best_agent[0] if best_agent[1] > 0 else None


def filter_messages_for_agent(messages: Tuple[Message, ...], agent_name: str) -> List[Message]:
    """Filter messages for a specific agent (pure function)."""
    return [m for m in messages if m.receiver == agent_name or m.receiver is None]


def count_tasks_by_status(tasks: Tuple[Task, ...], status: str) -> int:
    """Count tasks with given status (pure function)."""
    return sum(1 for t in tasks if t.status == status)


def create_stats(state: OrchestratorState) -> Dict[str, Any]:
    """Create statistics dictionary (pure function)."""
    return {
        'name': state.name,
        'execution_mode': state.execution_mode.value,
        'registered_agents': len(state.agents),
        'agents': list(state.agents.keys()),
        'agent_roles': {name: cap.role.value for name, cap in state.capabilities.items()},
        'tasks_completed': count_tasks_by_status(state.tasks, "completed"),
        'tasks_failed': count_tasks_by_status(state.tasks, "failed"),
        'total_tasks': len(state.tasks),
        'message_queue_size': len(state.messages)
    }


def execute_single_task(agent: Agent, task: Task) -> Dict[str, Any]:
    """Execute a single task with an agent (impure - I/O operation)."""
    try:
        result = agent.process_input(task.description)
        return {
            'task': task.description,
            'agent': task.assigned_agent,
            'status': 'completed',
            'result': result
        }
    except Exception as e:
        logger.error(f"Error executing task: {str(e)}")
        return {
            'task': task.description,
            'agent': task.assigned_agent,
            'status': 'failed',
            'error': str(e)
        }


def assign_task_to_agent(task: Task, capabilities: Dict[str, AgentCapability]) -> Task:
    """Assign task to best matching agent (pure function)."""
    if task.assigned_agent:
        return task

    agent_name = find_best_agent(capabilities, task)
    if agent_name:
        logger.info(f"Routed task to '{agent_name}'")
        return task.with_agent(agent_name).with_status("in_progress")

    logger.warning("No suitable agent found for task")
    return task


def format_debate_context(topic: str, history: List[List[Dict[str, Any]]]) -> str:
    """Format debate context for agents (pure function)."""
    context = f"Debate Topic: {topic}\n\n"
    if history:
        context += "Previous Responses:\n"
        for round_data in history:
            for resp in round_data:
                content = str(resp['response'])
                preview = content[:200] + "..." if len(content) > 200 else content
                context += f"- {resp['agent']}: {preview}\n"
        context += "\n"
    return context


def format_hierarchical_summary(results: List[Dict[str, Any]]) -> str:
    """Format worker results for manager (pure function)."""
    summary = "Worker Results:\n\n"
    for i, result in enumerate(results):
        summary += f"{i+1}. Task: {result['task']}\n"
        summary += f"   Agent: {result.get('agent', 'unknown')}\n"
        summary += f"   Status: {result['status']}\n"
        if result['status'] == 'completed':
            summary += f"   Result: {json.dumps(result['result'], indent=2)}\n"
        summary += "\n"
    return summary


# Orchestrator class (facade over functional core)

class Orchestrator:
    """Orchestrator with functional core and imperative shell."""

    def __init__(self, name: str = "Orchestrator", execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL):
        """Initialize orchestrator with immutable state."""
        self._state = OrchestratorState(name=name, execution_mode=execution_mode)

    @property
    def name(self) -> str:
        """Get orchestrator name."""
        return self._state.name

    @property
    def execution_mode(self) -> ExecutionMode:
        """Get execution mode."""
        return self._state.execution_mode

    @property
    def agents(self) -> Dict[str, Agent]:
        """Get registered agents (read-only)."""
        return dict(self._state.agents)

    @property
    def agent_capabilities(self) -> Dict[str, AgentCapability]:
        """Get agent capabilities (read-only)."""
        return dict(self._state.capabilities)

    @property
    def message_queue(self) -> List[Message]:
        """Get current messages (read-only)."""
        return list(self._state.messages)

    @property
    def task_history(self) -> List[Task]:
        """Get task history (read-only)."""
        return list(self._state.tasks)

    @property
    def shared_memory(self) -> Dict[str, Any]:
        """Get shared memory (mutable for backwards compatibility)."""
        return self._state.shared_data

    def add_agent(self, agent: Agent, capability: AgentCapability) -> None:
        """Add an agent to the orchestrator."""
        self._state = self._state.with_agent(agent.name, agent, capability)
        logger.info(f"Added agent '{agent.name}' with role {capability.role.value}")

    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the orchestrator."""
        self._state = self._state.without_agent(agent_name)
        logger.info(f"Removed agent '{agent_name}'")

    def send_message(self, message: Message) -> None:
        """Send a message."""
        self._state = self._state.with_message(message)
        logger.debug(f"Message from {message.sender} to {message.receiver}: {message.content[:50]}...")

    def get_messages_for_agent(self, agent_name: str, clear: bool = True) -> List[Message]:
        """Get messages for an agent."""
        messages = filter_messages_for_agent(self._state.messages, agent_name)

        if clear:
            self._state = self._state.clear_messages_for(agent_name)

        return messages

    def route_task(self, task: Task) -> Optional[str]:
        """Route task to best agent (pure delegation)."""
        return find_best_agent(self._state.capabilities, task)

    def execute(self, tasks: List[Task], mode: Optional[ExecutionMode] = None) -> List[Dict[str, Any]]:
        """Execute tasks using specified mode."""
        exec_mode = mode or self.execution_mode

        if exec_mode == ExecutionMode.SEQUENTIAL:
            return self.execute_task_sequential(tasks)
        elif exec_mode == ExecutionMode.PARALLEL:
            return self.execute_task_parallel(tasks)
        else:
            raise ValueError(f"Mode {exec_mode} requires specific method call")

    def execute_task_sequential(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """Execute tasks sequentially (functional pipeline)."""
        results = []

        for task in tasks:
            # Pure: Assign agent
            assigned_task = assign_task_to_agent(task, self._state.capabilities)

            if not assigned_task.assigned_agent:
                result = {
                    'task': assigned_task.description,
                    'status': 'failed',
                    'error': 'No suitable agent found'
                }
                results.append(result)
                self._state = self._state.with_task(assigned_task.as_failed("No suitable agent"))
                continue

            # Impure: Execute task
            agent = self._state.agents[assigned_task.assigned_agent]
            result = execute_single_task(agent, assigned_task)

            # Update state
            if result['status'] == 'completed':
                completed_task = assigned_task.with_result(result['result'])
                self._state = self._state.with_task(completed_task)
                self._state = self._state.with_shared_data(
                    f"task_{len(self._state.tasks)}",
                    result
                )
            else:
                failed_task = assigned_task.as_failed(result.get('error', 'Unknown error'))
                self._state = self._state.with_task(failed_task)

            results.append(result)

        return results

    def execute_task_parallel(self, tasks: List[Task], max_workers: int = 5) -> List[Dict[str, Any]]:
        """Execute tasks in parallel (functional + concurrent)."""
        # Pure: Assign all agents
        assigned_tasks = [
            assign_task_to_agent(task, self._state.capabilities)
            for task in tasks
        ]

        def execute_task_wrapper(task: Task) -> Tuple[Task, Dict[str, Any]]:
            """Wrapper to return both task and result."""
            if not task.assigned_agent:
                result = {
                    'task': task.description,
                    'status': 'failed',
                    'error': 'No suitable agent found'
                }
                return task, result

            agent = self._state.agents[task.assigned_agent]
            result = execute_single_task(agent, task)
            return task, result

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(execute_task_wrapper, task): task
                for task in assigned_tasks
            }

            for future in as_completed(future_to_task):
                try:
                    task, result = future.result()

                    if result['status'] == 'completed':
                        completed_task = task.with_result(result['result'])
                        self._state = self._state.with_task(completed_task)
                    else:
                        failed_task = task.as_failed(result.get('error', 'Unknown'))
                        self._state = self._state.with_task(failed_task)

                    results.append(result)
                except Exception as e:
                    logger.error(f"Task execution failed: {str(e)}")
                    results.append({
                        'task': 'Unknown',
                        'status': 'failed',
                        'error': str(e)
                    })

        return results

    def execute_hierarchical(self, manager_agent: str, worker_tasks: List[Task]) -> Dict[str, Any]:
        """Execute hierarchical workflow."""
        if manager_agent not in self._state.agents:
            raise ValueError(f"Manager agent '{manager_agent}' not found")

        # Workers execute in parallel
        worker_results = self.execute_task_parallel(worker_tasks)

        # Manager synthesizes (pure summary + impure execution)
        summary = format_hierarchical_summary(worker_results)
        manager = self._state.agents[manager_agent]
        manager_prompt = f"Analyze and summarize the following results:\n\n{summary}"
        manager_result = manager.process_input(manager_prompt)

        return {
            'mode': 'hierarchical',
            'manager': manager_agent,
            'worker_results': worker_results,
            'manager_summary': manager_result
        }

    def execute_debate(self, topic: str, agents: List[str], rounds: int = 3) -> Dict[str, Any]:
        """Execute debate among agents."""
        if not all(a in self._state.agents for a in agents):
            raise ValueError("One or more agents not found")

        debate_history = []

        for round_num in range(rounds):
            round_responses = []

            for agent_name in agents:
                agent = self._state.agents[agent_name]

                # Pure: Format context
                context = format_debate_context(topic, debate_history)
                prompt = f"{context}Provide your perspective on this topic:"

                # Impure: Get response
                response = agent.process_input(prompt)

                round_responses.append({
                    'round': round_num + 1,
                    'agent': agent_name,
                    'response': response
                })

            debate_history.append(round_responses)

        # Consensus using first agent
        synthesizer = self._state.agents[agents[0]]
        summary = f"Debate on: {topic}\n\nAll perspectives:\n"

        for round_data in debate_history:
            for resp in round_data:
                summary += f"\nRound {resp['round']} - {resp['agent']}:\n{resp['response']}\n"

        consensus_prompt = f"{summary}\n\nProvide a consensus summary integrating all perspectives:"
        consensus = synthesizer.process_input(consensus_prompt)

        return {
            'mode': 'debate',
            'topic': topic,
            'participants': agents,
            'rounds': rounds,
            'debate_history': debate_history,
            'consensus': consensus
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics (pure delegation)."""
        return create_stats(self._state)

    def reset(self) -> None:
        """Reset orchestrator state."""
        self._state = self._state.reset()
        logger.info("Orchestrator state reset")


# Helper functions

def create_specialized_agent(
    name: str,
    role: AgentRole,
    model: str = "llama2",
    skills: Optional[List[str]] = None,
    **agent_kwargs
) -> Tuple[Agent, AgentCapability]:
    """Create specialized agent with capabilities (pure + impure)."""

    # Pure: Define default skills
    default_skills_map = {
        AgentRole.RESEARCHER: ["research", "search", "analyze", "gather information"],
        AgentRole.CODER: ["code", "programming", "debug", "implement", "refactor"],
        AgentRole.ANALYST: ["analyze", "evaluate", "statistics", "data analysis"],
        AgentRole.PLANNER: ["plan", "organize", "schedule", "coordinate"],
        AgentRole.CRITIC: ["review", "critique", "evaluate", "feedback"],
        AgentRole.WRITER: ["write", "document", "compose", "edit"],
        AgentRole.COORDINATOR: ["coordinate", "manage", "delegate", "organize"],
    }

    agent_skills = skills or default_skills_map.get(role, [])

    # Impure: Create agent
    agent = Agent(name=name, model=model, **agent_kwargs)

    # Pure: Define context
    role_contexts = {
        AgentRole.RESEARCHER: "You are a research specialist focused on gathering and analyzing information.",
        AgentRole.CODER: "You are a programming expert focused on writing and debugging code.",
        AgentRole.ANALYST: "You are a data analyst focused on analyzing and interpreting information.",
        AgentRole.PLANNER: "You are a planning specialist focused on organizing and coordinating tasks.",
        AgentRole.CRITIC: "You are a critical reviewer focused on evaluating and providing constructive feedback.",
        AgentRole.WRITER: "You are a writing specialist focused on creating clear documentation.",
        AgentRole.COORDINATOR: "You are a coordination specialist focused on managing multiple tasks and agents.",
    }

    if role in role_contexts:
        agent.set_context(role_contexts[role])

    # Pure: Create capability (with immutable tuple)
    capability = AgentCapability(
        role=role,
        skills=tuple(agent_skills),
        description=f"{role.value.capitalize()} agent specialized in {', '.join(agent_skills[:3])}"
    )

    return agent, capability
