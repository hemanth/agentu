"""Workflow composition examples."""
import asyncio
from agentu import Agent


def research(topic: str) -> dict:
    """Research a topic."""
    return {"topic": topic, "findings": f"Findings about {topic}"}


def analyze(data: str) -> dict:
    """Analyze data."""
    return {"analysis": "Mock analysis", "confidence": 0.85}


async def main():
    # Create agents
    researcher = Agent("researcher").with_tools([research])
    analyst = Agent("analyst").with_tools([analyze])

    # Sequential workflow
    workflow = researcher("Research AI trends") >> analyst("Analyze findings")
    result = await workflow.run()
    print(f"Sequential result: {result}")

    # Parallel workflow
    workflow = researcher("AI") & researcher("ML") & researcher("Crypto")
    results = await workflow.run()
    print(f"Parallel results: {len(results)} tasks completed")


if __name__ == "__main__":
    asyncio.run(main())
