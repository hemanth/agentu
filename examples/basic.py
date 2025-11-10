"""Basic agent with tools."""
import asyncio
from agentu import Agent


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


async def main():
    # Create agent and add tools
    agent = Agent("calculator").with_tools([add, multiply])

    # Direct execution (no LLM needed)
    result = await agent.call("add", {"x": 5, "y": 3})
    print(f"5 + 3 = {result}")

    result = await agent.call("multiply", {"x": 5, "y": 3})
    print(f"5 * 3 = {result}")


if __name__ == "__main__":
    asyncio.run(main())
