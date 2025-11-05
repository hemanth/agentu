# Functional Programming Design

This document explains the functional programming architecture of the orchestrator system.

## Overview

The orchestrator has been rewritten following functional programming principles while maintaining a clean, imperative-style API for ease of use. This follows the **Functional Core, Imperative Shell** pattern.

## Architecture Pattern

```
┌─────────────────────────────────────┐
│     Imperative Shell (Public API)   │  ← Mutable interface for users
│  - Orchestrator class methods       │
│  - Familiar OOP interface           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Functional Core (Internal)     │  ← Immutable data, pure functions
│  - OrchestratorState (frozen)       │
│  - Pure functions                   │
│  - Immutable data structures        │
└─────────────────────────────────────┘
```

## Key Principles Applied

### 1. Immutability

All core data structures are immutable using `@dataclass(frozen=True)`:

```python
@dataclass(frozen=True)
class Message:
    sender: str
    receiver: Optional[str]
    content: str
    message_type: str = "info"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None

@dataclass(frozen=True)
class Task:
    description: str
    required_skills: Tuple[str, ...] = ()
    assigned_agent: Optional[str] = None
    result: Optional[Any] = None
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Benefits:**
- Thread-safe by default
- No unexpected mutations
- Easier to reason about
- Enables time-travel debugging

### 2. Pure Functions

Core logic is implemented as pure functions (same input → same output, no side effects):

```python
def calculate_agent_score(capability: AgentCapability, task: Task) -> float:
    """Pure function - deterministic scoring."""
    skill_match = capability.matches_task(list(task.required_skills))
    priority_weight = capability.priority / 10.0
    return skill_match * priority_weight

def find_best_agent(capabilities: Dict[str, AgentCapability], task: Task) -> Optional[str]:
    """Pure function - no side effects."""
    if not capabilities:
        return None

    scores = {
        name: calculate_agent_score(cap, task)
        for name, cap in capabilities.items()
    }

    best_agent = max(scores.items(), key=lambda x: x[1])
    return best_agent[0] if best_agent[1] > 0 else None
```

**Benefits:**
- Easy to test
- Composable
- Parallelizable
- Memoizable

### 3. State Transitions

Instead of mutating state, we create new state:

```python
@dataclass(frozen=True)
class OrchestratorState:
    """Immutable state container."""
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

    def with_message(self, message: Message) -> 'OrchestratorState':
        """Return new state with added message."""
        return replace(self, messages=(*self.messages, message))
```

**Benefits:**
- State history preserved
- Easy rollback
- Concurrent access safe

### 4. Separation of Pure and Impure

Clear distinction between pure logic and I/O:

```python
# Pure: Task routing logic
def assign_task_to_agent(task: Task, capabilities: Dict[str, AgentCapability]) -> Task:
    """Pure function - no I/O."""
    if task.assigned_agent:
        return task

    agent_name = find_best_agent(capabilities, task)
    if agent_name:
        logger.info(f"Routed task to '{agent_name}'")
        return task.with_agent(agent_name).with_status("in_progress")

    return task

# Impure: Agent execution (I/O)
def execute_single_task(agent: Agent, task: Task) -> Dict[str, Any]:
    """Impure - calls external LLM."""
    try:
        result = agent.process_input(task.description)  # I/O operation
        return {
            'task': task.description,
            'agent': task.assigned_agent,
            'status': 'completed',
            'result': result
        }
    except Exception as e:
        return {
            'task': task.description,
            'status': 'failed',
            'error': str(e)
        }
```

## Data Flow

### Sequential Execution Pipeline

```
Input Tasks (List[Task])
    ↓
[Map] Assign agents (pure)
    ↓
[Map] Execute tasks (impure)
    ↓
[Reduce] Collect results
    ↓
Output Results (List[Dict])
```

### State Evolution

```
Initial State
    ↓
[with_agent] → State₁
    ↓
[with_message] → State₂
    ↓
[with_task] → State₃
    ↓
Final State
```

Each arrow creates a new immutable state object.

## Key Design Decisions

### 1. Tuples Over Lists

Skills and messages use tuples (immutable) instead of lists:

```python
@dataclass(frozen=True)
class AgentCapability:
    skills: Tuple[str, ...] = ()  # Immutable

@dataclass(frozen=True)
class Task:
    required_skills: Tuple[str, ...] = ()  # Immutable
```

### 2. Copy-on-Write for Dicts

Since dicts are mutable, we create new dicts for state transitions:

```python
def with_agent(self, name: str, agent: Agent, capability: AgentCapability):
    return replace(
        self,
        agents={**self.agents, name: agent},  # New dict
        capabilities={**self.capabilities, name: capability}  # New dict
    )
```

### 3. Facade Pattern for API

The `Orchestrator` class provides a mutable-looking API but uses immutable state internally:

```python
class Orchestrator:
    def __init__(self, name: str, execution_mode: ExecutionMode):
        self._state = OrchestratorState(name=name, execution_mode=execution_mode)

    def add_agent(self, agent: Agent, capability: AgentCapability) -> None:
        """Appears mutable, but creates new state internally."""
        self._state = self._state.with_agent(agent.name, agent, capability)
```

**Benefits:**
- Familiar API for users
- Functional purity internally
- Best of both worlds

### 4. Properties for Read Access

State is exposed through read-only properties:

```python
@property
def agents(self) -> Dict[str, Agent]:
    """Returns copy to prevent mutation."""
    return dict(self._state.agents)

@property
def message_queue(self) -> List[Message]:
    """Returns list from tuple."""
    return list(self._state.messages)
```

## Functional Patterns Used

### 1. Higher-Order Functions

```python
# Filter pattern
def filter_messages_for_agent(messages: Tuple[Message, ...], agent_name: str) -> List[Message]:
    return [m for m in messages if m.receiver == agent_name or m.receiver is None]

# Map pattern
assigned_tasks = [
    assign_task_to_agent(task, self._state.capabilities)
    for task in tasks
]

# Reduce pattern (implicit)
def count_tasks_by_status(tasks: Tuple[Task, ...], status: str) -> int:
    return sum(1 for t in tasks if t.status == status)
```

### 2. Function Composition

```python
# Compose scoring functions
def calculate_agent_score(capability: AgentCapability, task: Task) -> float:
    skill_match = capability.matches_task(list(task.required_skills))  # Step 1
    priority_weight = capability.priority / 10.0                        # Step 2
    return skill_match * priority_weight                                 # Step 3 (compose)
```

### 3. Builder Pattern (Immutable)

Tasks support immutable updates through builder methods:

```python
task = Task(description="Research AI")
    .with_agent("ResearchBot")
    .with_status("in_progress")
    .with_result({"findings": "..."})
```

### 4. Pipeline Pattern

```python
def execute_task_sequential(self, tasks: List[Task]) -> List[Dict[str, Any]]:
    results = []
    for task in tasks:
        # Pipeline: task → assign → execute → update state
        assigned_task = assign_task_to_agent(task, self._state.capabilities)
        agent = self._state.agents[assigned_task.assigned_agent]
        result = execute_single_task(agent, assigned_task)

        if result['status'] == 'completed':
            completed_task = assigned_task.with_result(result['result'])
            self._state = self._state.with_task(completed_task)

        results.append(result)
    return results
```

## Testing Benefits

Functional code is easier to test:

```python
def test_calculate_agent_score():
    """Pure function → simple test."""
    capability = AgentCapability(
        role=AgentRole.RESEARCHER,
        skills=("research", "search"),
        priority=8
    )
    task = Task(
        description="Research AI",
        required_skills=("research",)
    )

    score = calculate_agent_score(capability, task)

    # Deterministic result
    assert score == 0.4  # (1/2 skills matched) * (8/10 priority)
```

No mocking needed for pure functions!

## Performance Considerations

### Memory

- Creating new state objects has overhead
- Mitigated by sharing unchanged data (structural sharing)
- Python's reference counting helps

### Optimization Strategies

1. **Lazy evaluation** - Only compute when needed
2. **Memoization** - Cache pure function results
3. **Batch updates** - Collect multiple state changes

Example memoization opportunity:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_agent_score_cached(capability_id: str, task_id: str) -> float:
    """Cached version for repeated calculations."""
    # Implementation
    pass
```

## Migration Guide

### Before (Mutable)

```python
task.status = "completed"
task.result = result
```

### After (Immutable)

```python
task = task.with_result(result)  # Creates new task with status="completed"
```

### Before (Direct mutation)

```python
orchestrator.message_queue.append(message)
```

### After (Method call)

```python
orchestrator.send_message(message)  # Updates internal state
```

## Best Practices

### 1. Keep Pure Functions Pure

```python
# Good
def find_best_agent(capabilities, task):
    # No I/O, no mutations
    scores = {name: score(cap, task) for name, cap in capabilities.items()}
    return max(scores, key=scores.get)

# Bad
def find_best_agent(capabilities, task):
    logger.info("Finding agent...")  # Side effect!
    return max(...)
```

### 2. Mark Impure Functions Clearly

```python
def execute_single_task(agent: Agent, task: Task) -> Dict[str, Any]:
    """Execute task (impure - I/O operation)."""
    # Clear documentation
```

### 3. Use Immutable Collections

```python
# Good
skills: Tuple[str, ...] = ("research", "code")

# Less ideal
skills: List[str] = ["research", "code"]  # Mutable
```

### 4. Avoid Shared Mutable State

```python
# Good
def process(data):
    result = transform(data)  # New data
    return result

# Bad
shared_cache = {}
def process(data):
    shared_cache[data.id] = result  # Mutation!
```

## Future Enhancements

Potential functional improvements:

1. **Monadic error handling** - Use `Result` type instead of exceptions
2. **Persistent data structures** - Use libraries like `pyrsistent`
3. **Pure async** - Separate async execution from state management
4. **Event sourcing** - Store all state changes as events
5. **Type-safe state machines** - Use enums for state transitions

## References

- [Functional Core, Imperative Shell](https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell)
- [Python Dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Immutability in Python](https://docs.python.org/3/library/functions.html#property)
- [Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html)
