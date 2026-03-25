"""
Ralph Mode Example - Autonomous Agent Loop

This example demonstrates how to run an agent in "Ralph mode" -
a continuous autonomous loop that works toward a goal.

Usage:
    python examples/ralph_demo.py
"""

import asyncio
from agentu import Agent


# Example tools for the agent
def create_file(filename: str, content: str) -> str:
    """Create a file with the given content."""
    with open(filename, 'w') as f:
        f.write(content)
    return f"Created {filename}"


def read_file(filename: str) -> str:
    """Read content from a file."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"File {filename} not found"


def run_tests() -> str:
    """Run the test suite."""
    # Simulated test run
    return "All tests passed! ✅"


async def main():
    # Create an agent with tools
    agent = Agent("builder").with_tools([
        create_file,
        read_file,
        run_tests
    ])
    
    # Create a sample PROMPT.md file
    prompt_content = """# Goal
Build a simple greeting module.

## Signs (Guardrails)
- Keep code simple and readable
- Use proper docstrings
- Test before marking complete

## Checkpoints
- [ ] Create greeting.py with a greet() function
- [ ] Create test_greeting.py with tests
- [ ] Run tests and verify they pass

## Current State
Last iteration: 0
"""
    
    with open("PROMPT.md", "w") as f:
        f.write(prompt_content)
    
    # Progress callback
    def on_progress(iteration: int, data: dict):
        print(f"🔄 Iteration {iteration}: {data.get('result', '')[:50]}...")
    
    # Run Ralph mode
    print("🚀 Starting Ralph mode...")
    result = await agent.ralph(
        prompt_file="PROMPT.md",
        max_iterations=10,
        timeout_minutes=5,
        on_iteration=on_progress
    )
    
    print("\n📊 Final Summary:")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Stopped by: {result['stopped_by']}")
    print(f"  Errors: {len(result['errors'])}")


if __name__ == "__main__":
    asyncio.run(main())
