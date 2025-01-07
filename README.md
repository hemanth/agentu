# AgentU

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
