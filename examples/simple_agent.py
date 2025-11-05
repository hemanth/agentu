"""Simple agent execution example."""

import asyncio
from agentu import Agent, Tool


def calculator(x: float, y: float, operation: str) -> float:
    """Simple calculator function."""
    operations = {
        "add": x + y,
        "subtract": x - y,
        "multiply": x * y,
        "divide": x / y if y != 0 else "Error: Division by zero"
    }
    return operations.get(operation, "Unknown operation")


async def main():
    """Run simple agent example."""
    print("Simple Agent Execution Example")
    print("=" * 50)

    # Create agent
    agent = Agent(name="MathBot", model="llama2", enable_memory=False)

    # Add calculator tool
    calc_tool = Tool(
        name="calculator",
        description="Perform basic arithmetic operations (add, subtract, multiply, divide)",
        function=calculator,
        parameters={
            "x": "float: First number",
            "y": "float: Second number",
            "operation": "str: Operation to perform (add, subtract, multiply, divide)"
        }
    )
    agent.add_tool(calc_tool)

    print("\nAgent created with calculator tool")
    print(f"Agent name: {agent.name}")
    print(f"Tools available: {[t.name for t in agent.tools]}")

    # Example 1: Natural language execution
    print("\n" + "=" * 50)
    print("Example 1: Natural language input")
    print("=" * 50)
    print("\nInput: 'What is 25 times 4?'")

    # Note: This requires Ollama running locally
    # Uncomment to test:
    # result = await agent.process_input("What is 25 times 4?")
    # print(f"Result: {result}")

    print("\n(Skipped - requires Ollama running)")
    print("Expected output:")
    print({
        'tool_used': 'calculator',
        'parameters': {'x': 25, 'y': 4, 'operation': 'multiply'},
        'result': 100
    })

    # Example 2: Direct tool execution
    print("\n" + "=" * 50)
    print("Example 2: Direct tool execution")
    print("=" * 50)
    print("\nDirect execution: calculator(150, 50, 'divide')")

    result = await agent.execute_tool("calculator", {
        "x": 150,
        "y": 50,
        "operation": "divide"
    })
    print(f"Result: {result}")

    # Example 3: Multiple operations
    print("\n" + "=" * 50)
    print("Example 3: Multiple operations")
    print("=" * 50)

    operations = [
        ("add", 10, 5),
        ("subtract", 20, 8),
        ("multiply", 7, 6),
        ("divide", 100, 4)
    ]

    for op, x, y in operations:
        result = await agent.execute_tool("calculator", {
            "x": x,
            "y": y,
            "operation": op
        })
        print(f"{x} {op} {y} = {result}")

    print("\n" + "=" * 50)
    print("Example complete!")
    print("\nTo use natural language execution:")
    print("1. Ensure Ollama is running: ollama serve")
    print("2. Uncomment the process_input() call above")
    print("3. Run: python examples/simple_agent.py")


if __name__ == "__main__":
    asyncio.run(main())
