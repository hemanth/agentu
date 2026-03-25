"""Example: Agent Evaluation

Demonstrates how to evaluate agent performance with test cases.
"""

import asyncio
from agentu import Agent, Tool, evaluate


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is sunny and 72°F"


async def main():
    print("=== Agent Evaluation Example ===\n")
    
    # Create agent with tools
    agent = Agent(
        name="MathWeatherAgent",
        model="qwen3:latest"
    ).with_tools([
        Tool(add),
        Tool(get_weather)
    ])
    
    # Define test cases
    test_cases = [
        # Math tests
        {
            "ask": "What is 5 plus 3?",
            "expect": 8,
            "expect_tool": "add"
        },
        {
            "ask": "Add 10 and 20",
            "expect": 30
        },
        
        # Weather tests
        {
            "ask": "What's the weather in San Francisco?",
            "expect": "sunny",  # Substring match
        },
        {
            "ask": "Tell me about NYC weather",
            "expect": "72",  # Checks for temperature
        },
        
        # Custom validator
        {
            "ask": "What's 100 plus 50?",
            "expect": 100,
            "validator": lambda expected, actual: actual > expected  # Any result > 100
        }
    ]
    
    print(f"Running {len(test_cases)} test cases...\n")
    
    # Run evaluation
    results = await evaluate(agent, test_cases)
    
    # Print results
    print(results)
    print()
    
    # Export to JSON
    print("=== JSON Export ===")
    print(results.to_json())
    print()
    
    # Show detailed stats
    print("=== Detailed Stats ===")
    print(f"Success rate: {results.accuracy}%")
    print(f"Average turns per test: {results.avg_turns:.2f}")
    print(f"Total duration: {results.duration:.2f}s")
    print(f"Failed cases: {len(results.failures)}")
    
    if results.failures:
        print("\nFailed test details:")
        for fail in results.failures:
            print(f"  - {fail.query}: {fail.reason}")


async def demo_llm_judge():
    """Example using LLM-as-judge for semantic matching."""
    print("\n=== LLM Judge Example ===\n")
    
    agent = Agent("assistant").with_tools([Tool(get_weather)])
    
    # These require semantic understanding
    cases = [
        {
            "ask": "How's the weather in Seattle?",
            "expect": "The forecast shows good conditions"  # Semantic match
        }
    ]
    
    # Use LLM to judge semantic similarity
    results = await evaluate(agent, cases, use_llm_judge=True)
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
    
    # Uncomment to test LLM judge:
    # asyncio.run(demo_llm_judge())
