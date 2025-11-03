# agentu

Agentu is a flexible Python package for creating and managing AI agents with customizable tools using Ollama for evaluation.

## Installation

```bash
pip install agentu
```

## Quick Start - Using the Search Agent

The easiest way to get started is to use the built-in SearchAgent:

```python
from agentu import SearchAgent

# Create a search agent
agent = SearchAgent(
    name="research_assistant",
    model="llama3",
    max_results=3
)

# Perform a search
result = agent.search(
    query="Latest developments in quantum computing",
    region="wt-wt",  # worldwide
    safesearch="moderate"
)

# Print the results
print(result)
```

## Creating Custom Agents

You can also create custom agents with your own tools:

```python
from agentu import Agent, Tool, search_tool

# Create a new agent
agent = Agent("my_agent", model="llama3")

# Add the built-in search tool
agent.add_tool(search_tool)

# Add your own custom tool
def custom_tool(param1: str, param2: int) -> str:
    return f"{param1} repeated {param2} times"

my_tool = Tool(
    name="repeater",
    description="Repeats a string n times",
    function=custom_tool,
    parameters={
        "param1": "str: String to repeat",
        "param2": "int: Number of repetitions"
    }
)

agent.add_tool(my_tool)

# Use the agent
result = agent.process_input("Search for quantum computing and repeat the first title 3 times")
print(result)
```

## Features

- Built-in SearchAgent for easy web searches
- Integration with DuckDuckGo search
- Customizable search parameters (region, SafeSearch, etc.)
- Easy-to-use API for creating custom agents
- Type hints and comprehensive documentation
- **Memory System**: Short-term and long-term memory with persistent storage
- **MCP Remote Server Support**: Connect to remote MCP (Model Context Protocol) servers
- **Flexible Authentication**: Bearer tokens, API keys, and custom headers
- **Multi-Server Support**: Connect to multiple MCP servers simultaneously

## Advanced Search Options

The SearchAgent supports various options:

```python
agent = SearchAgent()

# Custom number of results
result = agent.search("AI news", max_results=5)

# Region-specific search
result = agent.search("local news", region="us-en")

# SafeSearch settings
result = agent.search("images", safesearch="strict")
```


__Example output:__

```python
{
    "tool_used": "web_search",
    "parameters": {
        "query": "James Webb Space Telescope recent discoveries",
        "max_results": 3
    },
    "reasoning": "User wants information about the James Webb Space Telescope. Using web_search to find recent and relevant information.",
    "result": [
        {
            "title": "James Webb Space Telescope - NASA",
            "link": "https://www.nasa.gov/mission/webb/",
            "snippet": "The James Webb Space Telescope is the largest, most powerful space telescope ever built..."
        },
        # Additional results...
    ]
}
```

## MCP Remote Server Support

Connect to remote MCP (Model Context Protocol) servers to access additional tools. Supports both **HTTP** and **SSE** transports.

### Simple Example

```python
from agentu import Agent, MCPServerConfig, AuthConfig, TransportType

# Configure MCP server
auth = AuthConfig.bearer_token("your_token")
config = MCPServerConfig(
    name="my_server",
    transport_type=TransportType.HTTP,  # or TransportType.SSE
    url="https://api.example.com/mcp",
    auth=auth
)

# Connect and use
agent = Agent(name="mcp_agent")
tools = agent.add_mcp_server(config)
result = agent.execute_tool("server_tool_name", {"param": "value"})
```

<details>
<summary><b>Multiple MCP Servers (Programmatic)</b></summary>

```python
from agentu import Agent, MCPServerConfig, AuthConfig, TransportType

agent = Agent(name="multi_agent")

# HTTP server
http_config = MCPServerConfig(
    name="server1",
    transport_type=TransportType.HTTP,
    url="https://api.example.com/mcp",
    auth=AuthConfig.bearer_token("token1")
)
agent.add_mcp_server(http_config)

# SSE server (e.g., PayPal MCP)
sse_config = MCPServerConfig(
    name="paypal",
    transport_type=TransportType.SSE,
    url="https://mcp.paypal.com/sse",
    auth=AuthConfig.bearer_token("token2")
)
agent.add_mcp_server(sse_config)

# Now you have tools from both servers
print(f"Total tools: {len(agent.tools)}")
```
</details>

<details>
<summary><b>Multiple MCP Servers (Config File)</b></summary>

Create a JSON configuration file (e.g., `mcp_config.json`):

```json
{
  "mcp_servers": {
    "server1": {
      "type": "http",
      "url": "https://api.example.com/mcp",
      "auth": {
        "type": "bearer",
        "headers": {
          "Authorization": "Bearer token1"
        }
      }
    },
    "paypal": {
      "type": "sse",
      "url": "https://mcp.paypal.com/sse",
      "auth": {
        "type": "bearer",
        "headers": {
          "Authorization": "Bearer token2"
        }
      },
      "timeout": 30
    }
  }
}
```

Load all servers from the config file:

```python
from agentu import Agent

agent = Agent(name="multi_agent")

# Load all MCP servers from config file
tools = agent.load_mcp_tools("mcp_config.json")

print(f"Loaded {len(tools)} tools from {len(agent.mcp_tool_manager.adapters)} servers")

# Use any tool from any server
result = agent.execute_tool("server_tool_name", {"param": "value"})
```
</details>

### Auth Options

```python
# Bearer token
auth = AuthConfig.bearer_token("your_token")

# API key
auth = AuthConfig.api_key("key", header_name="X-API-Key")

# Custom headers
auth = AuthConfig(type="custom", headers={"Auth": "value"})
```

## Memory System

AgentU includes a powerful memory system that combines short-term (working memory) and long-term (persistent storage) memory capabilities.

### Basic Memory Usage

```python
from agentu import Agent

# Create agent with memory enabled (default)
agent = Agent(name="memory_agent", enable_memory=True)

# Store memories
agent.remember("User prefers Python over JavaScript", memory_type="fact", importance=0.8)
agent.remember("Project deadline is next Friday", memory_type="task", importance=0.9)
agent.remember("Meeting notes: Discussed API design", memory_type="observation")

# Recall memories
recent = agent.recall(limit=5)  # Get recent memories
facts = agent.recall(memory_type="fact")  # Get all facts
search = agent.recall(query="deadline")  # Search memories

# Get memory context for prompts
context = agent.get_memory_context(max_entries=5)
print(context)

# Get memory statistics
stats = agent.get_memory_stats()
print(f"Total memories: {stats['total_memories']}")
```

### Persistent Memory

```python
from agentu import Agent

# Create agent with persistent memory storage
agent = Agent(
    name="persistent_agent",
    enable_memory=True,
    memory_path="agent_memory.json"  # Saves to file
)

# Memories are automatically saved to file
agent.remember("Important information", store_long_term=True)

# Manually save memory
agent.save_memory()

# Memory is automatically loaded on next initialization
agent2 = Agent(name="agent2", memory_path="agent_memory.json")
memories = agent2.recall(limit=10)  # Previously stored memories are loaded
```

### Memory Types and Importance

```python
from agentu import Agent

agent = Agent(name="smart_agent", enable_memory=True)

# Different memory types
agent.remember("User said hello", memory_type="conversation", importance=0.4)
agent.remember("API key format: abc-123-xyz", memory_type="fact", importance=0.9)
agent.remember("Complete documentation by Monday", memory_type="task", importance=0.8)
agent.remember("Error rate increased at 3pm", memory_type="observation", importance=0.6)

# High importance memories (>= 0.7) automatically go to long-term storage
# Consolidate important short-term memories to long-term
agent.consolidate_memory(importance_threshold=0.6)

# Clear short-term memory while keeping long-term
agent.clear_short_term_memory()
```

### Advanced Memory Features

<details>
<summary><b>Direct Memory System Usage</b></summary>

```python
from agentu import Memory

# Create standalone memory system
memory = Memory(
    short_term_size=10,  # Max short-term entries
    storage_path="custom_memory.json",
    auto_consolidate=True
)

# Store with metadata
memory.remember(
    content="Database connection timeout after 30s",
    memory_type="fact",
    metadata={
        "source": "config",
        "category": "database",
        "priority": "high"
    },
    importance=0.85
)

# Search with relevance ranking
results = memory.recall(query="database", limit=5)

# Get all memories of a specific type
tasks = memory.recall(memory_type="task", limit=10)

# Get formatted context
context = memory.get_context(max_entries=5)

# Statistics
stats = memory.stats()
print(f"Short-term: {stats['short_term_size']}")
print(f"Long-term: {stats['long_term_size']}")
print(f"Types: {stats['memory_types']}")
```
</details>

### Memory in Conversations

Agents automatically store conversation history in memory:

```python
from agentu import Agent, Tool

agent = Agent(name="chat_agent", enable_memory=True)

# Add some tools
# ... tool setup ...

# Process inputs - automatically stored in memory
agent.process_input("What's the weather?")
agent.process_input("Book a flight to Tokyo")

# Recall conversation history
conversations = agent.recall(memory_type="conversation", limit=10)

for memory in conversations:
    print(f"{memory.content} (importance: {memory.importance})")
```
