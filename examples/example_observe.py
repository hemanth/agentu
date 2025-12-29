"""Example: Observability and Monitoring

Demonstrates how to track agent behavior and performance.
"""

import asyncio
from agentu import Agent, Tool, observe


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"


async def main():
    print("=== Observability Example ===\n")
    
    # Configure observability
    observe.configure(
        output="console",  # "json" | "console" | "silent"
        enabled=True
    )
    
    # Create agent (observer auto-initialized)
    agent = Agent("demo").with_tools([
        Tool(add),
        Tool(search_web)
    ])
    
    print("Executing tools...\n")
    
    # Direct tool call (tracked)
    result1 = await agent.call("add", {"x": 5, "y": 3})
    print(f"Result: {result1}\n")
    
    # Natural language inference (tracked)
    result2 = await agent.infer("Search for AI news")
    print(f"Result: {result2.get('result')}\n")
    
    # View metrics
    print("\n=== Metrics ===")
    metrics = agent.observer.get_metrics()
    print(f"Tool calls: {metrics['tool_calls']}")
    print(f"LLM requests: {metrics['llm_requests']}")
    print(f"Errors: {metrics['errors']}")
    print(f"Total duration: {metrics['total_duration_ms']:.2f}ms\n")
    
    # View recent events
    print("=== Recent Events ===")
    events = agent.observer.get_events(limit=10)
    for event in events:
        print(f"{event['event']}: {event.get('duration_ms', 0):.1f}ms")
    
    # Export to JSON
    print("\n=== JSON Export ===")
    import json
    print(json.dumps(metrics, indent=2))


async def demo_json_output():
    """Example with JSON output."""
    print("\n=== JSON Output Mode ===\n")
    
    observe.configure(output="json")
    
    agent = Agent("json_demo").with_tools([Tool(add)])
    await agent.call("add", {"x": 10, "y": 20})
    
    # Each event printed as JSON line


async def demo_silent_mode():
    """Example with silent mode (no output)."""
    print("\n=== Silent Mode (for production) ===\n")
    
    observe.configure(output="silent")
    
    agent = Agent("prod").with_tools([Tool(add)])
    await agent.call("add", {"x": 100, "y": 200})
    
    # No console output, but metrics still collected
    print(f"Metrics: {agent.observer.get_metrics()}")


if __name__ == "__main__":
    asyncio.run(main())
    
    # Uncomment to test other modes:
    # asyncio.run(demo_json_output())
    # asyncio.run(demo_silent_mode())
