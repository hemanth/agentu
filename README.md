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
