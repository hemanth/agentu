# Multi-Agent Orchestration

This document provides detailed information about AgentU's multi-agent orchestration capabilities.

## Overview

The orchestration system allows you to coordinate multiple specialized agents to work together on complex tasks. It provides intelligent task routing, multiple execution modes, inter-agent communication, and shared memory.

## Key Concepts

### 1. Orchestrator

The central coordinator that manages multiple agents, routes tasks, and aggregates results.

```python
from agentu.orchestrator import Orchestrator, ExecutionMode

orchestrator = Orchestrator(
    name="MyOrchestrator",
    execution_mode=ExecutionMode.SEQUENTIAL
)
```

### 2. Agent Roles

Predefined specializations with associated skills:

- **RESEARCHER** - Information gathering, search, analysis
- **CODER** - Programming, implementation, debugging
- **ANALYST** - Data analysis, evaluation, statistics
- **PLANNER** - Task organization, scheduling, coordination
- **CRITIC** - Review, critique, feedback
- **WRITER** - Documentation, composition, editing
- **COORDINATOR** - Multi-agent management, delegation
- **CUSTOM** - User-defined specialization

### 3. Tasks

Work items with descriptions and skill requirements:

```python
from agentu.orchestrator import Task

task = Task(
    description="Research AI safety trends",
    required_skills=["research", "search", "analyze"],
    metadata={"priority": "high"}
)
```

### 4. Agent Capabilities

Define an agent's skills and priority for task routing:

```python
from agentu.orchestrator import AgentCapability, AgentRole

capability = AgentCapability(
    role=AgentRole.RESEARCHER,
    skills=["research", "search", "analyze"],
    priority=8,  # 1-10 scale
    description="Research specialist"
)
```

## Execution Modes

### Sequential

Tasks execute one after another. Results from earlier tasks can inform later ones.

```python
orchestrator = Orchestrator(execution_mode=ExecutionMode.SEQUENTIAL)
results = orchestrator.execute(tasks)
```

**Use when:**
- Tasks depend on previous results
- You need ordered execution
- Sequential reasoning is required

### Parallel

Tasks execute simultaneously using thread pools.

```python
orchestrator = Orchestrator(execution_mode=ExecutionMode.PARALLEL)
results = orchestrator.execute_task_parallel(tasks, max_workers=5)
```

**Use when:**
- Tasks are independent
- You need maximum speed
- Agents can work concurrently

### Hierarchical

A manager agent delegates to worker agents and synthesizes results.

```python
result = orchestrator.execute_hierarchical(
    manager_agent="ManagerBot",
    worker_tasks=[task1, task2, task3]
)
```

**Use when:**
- You need result aggregation
- Complex task breakdown required
- Supervision and synthesis needed

### Debate

Agents discuss a topic over multiple rounds to reach consensus.

```python
result = orchestrator.execute_debate(
    topic="Should we adopt microservices?",
    agents=["Optimist", "Realist", "Skeptic"],
    rounds=3
)
```

**Use when:**
- Multiple perspectives needed
- Consensus building required
- Decision making with pros/cons

## Task Routing

Tasks are automatically routed to the most suitable agent based on:

1. **Skill matching** - Required skills vs agent capabilities
2. **Priority weighting** - Agent priority scores (1-10)
3. **Availability** - Registered and active agents

```python
# Automatic routing
agent_name = orchestrator.route_task(task)

# Manual task assignment
task.assigned_agent = "SpecificAgent"
```

## Inter-Agent Communication

Agents can communicate through messages:

```python
from agentu.orchestrator import Message

# Send message
message = Message(
    sender="Agent1",
    receiver="Agent2",  # or None for broadcast
    content="Analysis complete",
    message_type="result",
    metadata={"confidence": 0.95}
)
orchestrator.send_message(message)

# Retrieve messages
messages = orchestrator.get_messages_for_agent("Agent2")
```

**Message types:**
- `info` - General information
- `question` - Request for input
- `answer` - Response to question
- `task` - Task assignment
- `result` - Task completion

## Shared Memory

Agents can share data through the orchestrator:

```python
# Store data
orchestrator.shared_memory["api_key"] = "abc123"
orchestrator.shared_memory["results"] = {"score": 0.95}

# Retrieve data
key = orchestrator.shared_memory.get("api_key")

# Clear shared memory
orchestrator.reset()  # Clears shared memory + messages + history
```

## Creating Specialized Agents

### Using Helper Function

```python
from agentu.orchestrator import create_specialized_agent, AgentRole

agent, capability = create_specialized_agent(
    name="ResearchBot",
    role=AgentRole.RESEARCHER,
    model="llama3",
    enable_memory=True
)

orchestrator.add_agent(agent, capability)
```

### Custom Configuration

```python
from agentu import Agent
from agentu.orchestrator import AgentCapability, AgentRole

# Create agent manually
agent = Agent(name="CustomBot", model="llama3")
agent.set_context("You are a blockchain specialist...")

# Define custom capabilities
capability = AgentCapability(
    role=AgentRole.CUSTOM,
    skills=["blockchain", "smart-contracts", "solidity"],
    priority=9,
    description="Blockchain development expert"
)

orchestrator.add_agent(agent, capability)
```

## Best Practices

### 1. Choose the Right Execution Mode

- Use **Sequential** for dependent tasks
- Use **Parallel** for independent tasks
- Use **Hierarchical** when synthesis is needed
- Use **Debate** for decision making

### 2. Define Clear Task Requirements

```python
# Good: Specific skills
Task(
    description="Analyze sales data for Q4",
    required_skills=["analyze", "statistics", "data analysis"]
)

# Less ideal: Vague requirements
Task(description="Do analysis", required_skills=[])
```

### 3. Set Agent Priorities

Higher priority agents get preference when multiple agents match:

```python
AgentCapability(
    role=AgentRole.RESEARCHER,
    skills=["research"],
    priority=8  # Higher priority for senior researcher
)
```

### 4. Use Memory for Context

Enable memory for agents that need to remember context:

```python
agent, cap = create_specialized_agent(
    name="ContextBot",
    role=AgentRole.ANALYST,
    enable_memory=True,
    short_term_size=20
)
```

### 5. Monitor Orchestrator Stats

```python
stats = orchestrator.get_stats()
print(f"Completed: {stats['tasks_completed']}")
print(f"Failed: {stats['tasks_failed']}")
print(f"Agents: {stats['agents']}")
```

## Example Workflows

### Research Pipeline

```python
# 1. Gather information
research_task = Task(
    description="Research quantum computing trends",
    required_skills=["research", "search"]
)

# 2. Analyze findings
analysis_task = Task(
    description="Analyze research for business impact",
    required_skills=["analyze", "evaluate"]
)

# 3. Document results
writing_task = Task(
    description="Write executive summary",
    required_skills=["write", "document"]
)

results = orchestrator.execute([
    research_task,
    analysis_task,
    writing_task
])
```

### Parallel Code Review

```python
# Multiple reviewers check different aspects
review_tasks = [
    Task(description="Review code quality", required_skills=["code", "review"]),
    Task(description="Review security", required_skills=["security", "review"]),
    Task(description="Review performance", required_skills=["performance", "review"])
]

results = orchestrator.execute_task_parallel(review_tasks, max_workers=3)
```

### Team Decision Making

```python
result = orchestrator.execute_debate(
    topic="Which database should we use for the project?",
    agents=["TechLead", "DBA", "Architect"],
    rounds=2
)

consensus = result['consensus']
```

## Advanced Features

### Custom Task Filtering

```python
# Get all high-priority tasks
high_priority = [
    t for t in orchestrator.task_history
    if t.metadata.get("priority") == "high"
]
```

### Agent Management

```python
# List all agents
for name in orchestrator.agents.keys():
    print(f"Agent: {name}")

# Remove agent
orchestrator.remove_agent("OldAgent")

# Reset orchestrator
orchestrator.reset()  # Clears state but keeps agents
```

### Error Handling

```python
results = orchestrator.execute(tasks)

for result in results:
    if result['status'] == 'failed':
        print(f"Task failed: {result['task']}")
        print(f"Error: {result['error']}")
    else:
        print(f"Success: {result['result']}")
```

## Performance Considerations

1. **Parallel workers** - Adjust based on CPU cores and task complexity
2. **Memory usage** - Each agent maintains its own memory if enabled
3. **LLM calls** - Each task may require multiple LLM invocations
4. **Thread safety** - Orchestrator handles concurrent access automatically

## Limitations

- Requires Ollama or compatible LLM service running
- No built-in persistence of orchestrator state (agents have their own memory)
- Message queue is in-memory only
- No distributed execution (single machine only)

## Future Enhancements

Potential additions (not yet implemented):

- Agent pools with automatic scaling
- Persistent orchestrator state
- Distributed execution across machines
- Agent health monitoring and failover
- Workflow visualization
- Cost tracking per agent
- Dynamic agent creation based on workload

## See Also

- [README.md](README.md) - Main documentation
- [examples/multi_agent_example.py](examples/multi_agent_example.py) - Comprehensive examples
- [examples/quickstart_orchestration.py](examples/quickstart_orchestration.py) - Quick start guide
- [tests/test_orchestrator.py](tests/test_orchestrator.py) - Test suite
