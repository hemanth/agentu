# AgentU

A flexible Python framework for building AI agents with memory, tools, and MCP integration.

## Installation

```bash
pip install agentu
```

## Features

- **Persistent Memory** - SQLite-backed long-term memory with full-text search
- **Multi-Agent Orchestration** - Coordinate multiple specialized agents
- **MCP Integration** - Connect to Model Context Protocol servers (HTTP & SSE)
- **Built-in Search** - DuckDuckGo web search out of the box
- **Custom Tools** - Easy-to-define tool system
- **Flexible Auth** - Bearer tokens, API keys, custom headers

## Quick Start

### Simple Agent

```python
import asyncio
from agentu import Agent, Tool

# Create agent
agent = Agent(name="assistant", model="llama3")

# Add a custom tool
def calculator(x: float, y: float, op: str) -> float:
    ops = {"+": x+y, "-": x-y, "*": x*y, "/": x/y}
    return ops[op]

agent.add_tool(Tool(
    name="calc",
    description="Perform calculations",
    function=calculator,
    parameters={"x": "float", "y": "float", "op": "str: +,-,*,/"}
))

# Execute with natural language
async def main():
    result = await agent.process_input("Calculate 15 times 7")
    print(result)
    # {'tool_used': 'calc', 'parameters': {'x': 15, 'y': 7, 'op': '*'}, 'result': 105}

asyncio.run(main())
```

### Search Agent

```python
import asyncio
from agentu import SearchAgent

async def main():
    agent = SearchAgent(name="researcher", max_results=5)
    results = await agent.search("latest AI developments")
    print(results)

asyncio.run(main())
```

## Memory System

### Basic Usage

```python
agent = Agent("smart_agent", memory_path="agent.db")

# Store memories with importance scoring
agent.remember("API endpoint: /v1/chat", memory_type="fact", importance=0.9)
agent.remember("Fix bug in auth module", memory_type="task", importance=0.8)

# Recall by query or type
results = agent.recall(query="API")
tasks = agent.recall(memory_type="task", limit=5)

# Get stats
print(agent.get_memory_stats())
```

### Storage Options

```python
# SQLite (default) - recommended for production
agent = Agent(memory_path="agent.db", use_sqlite=True)

# JSON - simple file storage
agent = Agent(memory_path="agent.json", use_sqlite=False)
```

**SQLite Benefits:**
- Indexed queries for fast searches
- Full-text search (FTS5)
- Better performance at scale
- ACID compliance

### Memory Types

- `conversation` - Chat history
- `fact` - Knowledge and information
- `task` - To-dos and action items
- `observation` - Events and logs

## MCP (Model Context Protocol)

### Single Server

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
result = agent.execute_tool("tool_name", {"param": "value"})
```

### Multiple Servers (Config File)

Create `mcp_config.json`:

```json
{
  "mcp_servers": {
    "server1": {
      "type": "http",
      "url": "https://api.example.com/mcp",
      "auth": {"type": "bearer", "headers": {"Authorization": "Bearer token1"}}
    },
    "server2": {
      "type": "sse",
      "url": "https://sse.example.com/sse",
      "auth": {"type": "bearer", "headers": {"Authorization": "Bearer token2"}}
    }
  }
}
```

Load all servers:

```python
agent = Agent()
tools = agent.load_mcp_tools("mcp_config.json")
```

### Authentication

```python
# Bearer token
auth = AuthConfig.bearer_token("your_token")

# API key
auth = AuthConfig.api_key("key", header_name="X-API-Key")

# Custom headers
auth = AuthConfig(type="custom", headers={"Auth": "value"})
```

## Multi-Agent Orchestration

Coordinate specialized agents to solve complex tasks with async/await.

```python
import asyncio
from agentu.orchestrator import Orchestrator, AgentRole, Task, make_agent

async def main():
    orchestrator = Orchestrator()

    # Use predefined roles or custom strings
    orchestrator.add_agents([
        make_agent("ResearchBot", AgentRole.RESEARCHER),
        make_agent("AnalystBot", AgentRole.ANALYST),
        make_agent("CustomBot", "data-engineer", skills=["etl", "sql", "spark"])
    ])

    # Define and execute tasks
    tasks = [
        Task(description="Research AI safety trends", required_skills=["research"]),
        Task(description="Analyze findings", required_skills=["analyze"])
    ]

    results = await orchestrator.execute(tasks)

asyncio.run(main())
```

**Execution Modes:** Sequential, Parallel, Hierarchical, Debate

**Predefined Roles:** `RESEARCHER`, `CODER`, `ANALYST`, `PLANNER`, `CRITIC`, `WRITER`, `COORDINATOR`

You can use predefined roles (`AgentRole.RESEARCHER`) or define custom roles as strings.

See `examples/multi_agent_example.py` for detailed examples.

## Advanced Usage

<details>
<summary><b>Custom Tools</b></summary>

```python
from agentu import Agent, Tool

def fetch_data(source: str, filters: dict) -> dict:
    # Your implementation
    return {"data": [...]}

tool = Tool(
    name="data_fetcher",
    description="Fetch data from various sources",
    function=fetch_data,
    parameters={
        "source": "str: data source name",
        "filters": "dict: filter criteria"
    }
)

agent = Agent()
agent.add_tool(tool)
```

</details>

<details>
<summary><b>Memory Management</b></summary>

```python
agent = Agent(memory_path="agent.db", short_term_size=20)

# Store with metadata
agent.remember(
    "Database schema updated",
    memory_type="observation",
    metadata={"timestamp": "2024-01-15", "severity": "high"},
    importance=0.7,
    store_long_term=True
)

# Consolidate short-term to long-term
agent.consolidate_memory(importance_threshold=0.6)

# Clear short-term only
agent.clear_short_term_memory()

# Save manually
agent.save_memory()

# Get context for prompts
context = agent.get_memory_context(max_entries=10)
```

</details>

<details>
<summary><b>Direct Memory Access</b></summary>

```python
from agentu import Memory

memory = Memory(
    short_term_size=15,
    storage_path="custom.db",
    use_sqlite=True,
    auto_consolidate=True
)

# Store with metadata
memory.remember(
    content="Important data",
    memory_type="fact",
    metadata={"source": "api", "version": "2.0"},
    importance=0.85
)

# Search
results = memory.recall(query="data", limit=5)

# Filter by type
facts = memory.recall(memory_type="fact")

# Stats
stats = memory.stats()
print(f"Total: {stats['total_memories']}")
print(f"Short-term: {stats['short_term_size']}")
print(f"Long-term: {stats['long_term_size']}")
```

</details>

## API Reference

### Agent

```python
Agent(
    name: str,
    model: str = "llama2",
    temperature: float = 0.7,
    enable_memory: bool = True,
    memory_path: Optional[str] = None,
    short_term_size: int = 10,
    use_sqlite: bool = True,
    role: Optional[str] = None,
    skills: Optional[List[str]] = None,
    priority: int = 5
)
```

**Main Methods:**
- `await process_input(user_input: str)` - Execute agent with natural language input (returns dict with tool_used, result)
- `await execute_tool(name: str, params: dict)` - Execute a specific tool directly

**Tool Management:**
- `add_tool(tool: Tool)` - Add a custom tool
- `add_mcp_server(config: MCPServerConfig)` - Connect to MCP server
- `load_mcp_tools(config_path: str)` - Load multiple MCP servers

**Memory:**
- `remember(content, memory_type, importance)` - Store memory
- `recall(query, memory_type, limit)` - Retrieve memories
- `get_memory_context(max_entries)` - Get formatted context
- `consolidate_memory(threshold)` - Move important memories to long-term

### Memory

```python
Memory(
    short_term_size: int = 10,
    storage_path: Optional[str] = None,
    use_sqlite: bool = True,
    auto_consolidate: bool = True
)
```

**Methods:**
- `remember(content, memory_type, metadata, importance, store_long_term)`
- `recall(query, memory_type, limit, include_short_term)`
- `consolidate_to_long_term(importance_threshold)`
- `get_context(max_entries)`
- `stats()`

### SearchAgent

```python
SearchAgent(
    name: str = "search_agent",
    model: str = "llama2",
    max_results: int = 3
)
```

**Methods:**
- `search(query, max_results, region, safesearch)`

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/hemanth/agentu).
