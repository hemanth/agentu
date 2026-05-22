"""Code Mode example: LLM writes Python code to call tools.

Instead of making individual JSON tool calls, the LLM writes Python
code that chains multiple tool calls together. This is more token-efficient
and leverages LLMs' strength at writing code.

Inspired by Cloudflare's Code Mode: https://blog.cloudflare.com/code-mode/
"""

import asyncio
from agentu import Agent


# Define some tools
def search(query: str) -> str:
    """Search for products by keyword."""
    # Simulated search results
    products = {
        "laptop": [
            {"name": "MacBook Pro", "price": 1999},
            {"name": "ThinkPad X1", "price": 1299},
            {"name": "Dell XPS 15", "price": 1499},
        ],
        "phone": [
            {"name": "iPhone 16", "price": 999},
            {"name": "Pixel 9", "price": 799},
        ],
    }
    for key, items in products.items():
        if key in query.lower():
            return str(items)
    return "No products found"


def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.(). ")
    if all(c in allowed for c in expression):
        return str(eval(expression))
    return "Invalid expression"


def save_report(title: str, content: str) -> str:
    """Save a report with the given title and content."""
    print(f"[Report saved: {title}]")
    return f"Report '{title}' saved successfully"


async def main():
    # Create agent with codemode enabled
    agent = Agent(
        name="analyst",
        codemode=True,  # <-- LLM writes code instead of making tool calls
    )
    agent.with_tools([search, calculate, save_report])

    # The LLM will write Python code that calls these tools
    # instead of making one tool call at a time
    result = await agent.infer(
        "Search for laptops, calculate the average price, "
        "and save a report with the findings"
    )

    print(f"\n{'='*50}")
    print(f"Tool used: {result['tool_used']}")
    print(f"Code generated:\n{result['parameters'].get('code', 'N/A')}")
    print(f"\nResult:\n{result['result']}")


if __name__ == "__main__":
    asyncio.run(main())
