# AgentU

A flexible Python framework for building AI agents with memory, tools, and orchestration.

```bash
pip install agentu
```

## Quick Start

### Basic Agent

```python
import asyncio
from agentu import Agent, Tool

def calculator(x: float, y: float, op: str) -> float:
    return {"+": x+y, "-": x-y, "*": x*y, "/": x/y}[op]

agent = Agent("assistant", model="llama3")
agent.add_tool(Tool(
    name="calc",
    description="Perform calculations",
    function=calculator,
    parameters={"x": "float", "y": "float", "op": "str: +,-,*,/"}
))

async def main():
    # Natural language execution (requires Ollama)
    result = await agent.process_input("Calculate 15 times 7")
    print(result)  # {'tool_used': 'calc', 'result': 105, ...}

    # Direct execution (no LLM)
    result = await agent.execute_tool("calc", {"x": 10, "y": 5, "op": "+"})
    print(result)  # 15

asyncio.run(main())
```

### Multi-Agent Orchestration

```python
from agentu.orchestrator import Orchestrator, AgentRole, Task, make_agent

async def main():
    orchestrator = Orchestrator()

    orchestrator.add_agents([
        make_agent("Researcher", AgentRole.RESEARCHER),
        make_agent("Analyst", AgentRole.ANALYST),
        make_agent("DataEngineer", "data-engineer", skills=["etl", "sql"])  # Custom role
    ])

    results = await orchestrator.execute([
        Task(description="Research AI trends", required_skills=["research"]),
        Task(description="Analyze findings", required_skills=["analyze"])
    ])

asyncio.run(main())
```

**Execution Modes:** `SEQUENTIAL`, `PARALLEL`, `HIERARCHICAL`, `DEBATE`

**Predefined Roles:** `RESEARCHER`, `CODER`, `ANALYST`, `PLANNER`, `CRITIC`, `WRITER`, `COORDINATOR`

## Features

### Memory System

```python
agent = Agent("smart_agent", memory_path="agent.db")

# Store and recall
agent.remember("API endpoint: /v1/chat", memory_type="fact", importance=0.9)
results = agent.recall(query="API", limit=5)

# SQLite (default) or JSON storage
agent = Agent(memory_path="agent.db", use_sqlite=True)
```

**Memory types:** `conversation`, `fact`, `task`, `observation`

### Web Search

```python
from agentu import SearchAgent

async def main():
    agent = SearchAgent("researcher", max_results=5)
    results = await agent.search("latest AI developments")

asyncio.run(main())
```

### MCP Integration

```python
from agentu import Agent, MCPServerConfig, AuthConfig, TransportType

config = MCPServerConfig(
    name="my_server",
    transport_type=TransportType.HTTP,
    url="https://api.example.com/mcp",
    auth=AuthConfig.bearer_token("token")
)

agent = Agent()
tools = agent.add_mcp_server(config)
result = await agent.execute_tool("tool_name", {"param": "value"})
```

Load from config file:
```python
agent.load_mcp_tools("mcp_config.json")
```

## API

### Agent

```python
Agent(
    name: str,
    model: str = "llama2",
    temperature: float = 0.7,
    enable_memory: bool = True,
    memory_path: Optional[str] = None,
    role: Optional[str] = None,
    skills: Optional[List[str]] = None
)
```

**Execution:**
- `await process_input(text)` - Natural language execution (requires Ollama)
- `await execute_tool(name, params)` - Direct tool execution

**Tools:**
- `add_tool(tool)` - Add custom tool
- `add_mcp_server(config)` - Connect to MCP server
- `load_mcp_tools(path)` - Load from config

**Memory:**
- `remember(content, memory_type, importance)`
- `recall(query, memory_type, limit)`
- `consolidate_memory(threshold)`

### Orchestrator

```python
Orchestrator(name: str = "Orchestrator", execution_mode: ExecutionMode = SEQUENTIAL)
```

**Methods:**
- `add_agent(agent)` - Add single agent
- `add_agents([agents])` - Add multiple agents
- `await execute(tasks)` - Execute task list
- `route_task(task)` - Get best agent for task

### Helper Functions

```python
make_agent(name, role, model="llama2", skills=None, **kwargs) -> Agent
```

Create agent with predefined or custom role.

## Examples

- `examples/simple_agent.py` - Basic agent execution
- `examples/quickstart_orchestration.py` - Multi-agent demo
- `examples/multi_agent_example.py` - Complete orchestration examples

## License

MIT

## Contributing

Contributions welcome at [github.com/hemanth/agentu](https://github.com/hemanth/agentu)
