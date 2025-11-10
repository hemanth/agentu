"""Workflow composition with >> and & operators."""
import asyncio
from agentu import Agent


def research(topic: str) -> dict:
    """Research a topic."""
    return {"topic": topic, "findings": f"Research on {topic}"}


def analyze(data: str) -> dict:
    """Analyze data."""
    return {"analysis": f"Analysis of {data}", "confidence": 0.85}


def summarize(text: str) -> dict:
    """Summarize text."""
    return {"summary": f"Summary: {text}"}


async def main():
    # Create agents
    researcher = Agent("researcher").with_tools([research])
    analyst = Agent("analyst").with_tools([analyze])
    writer = Agent("writer").with_tools([summarize])

    print("=== Sequential Workflow (>>)===")
    # Chain steps sequentially
    workflow = (
        researcher("Research AI trends")
        >> analyst("Analyze findings")
        >> writer("Write summary")
    )
    result = await workflow.run()
    print(f"Result: {result}")

    print("\n=== Parallel Workflow (&) ===")
    # Run multiple steps in parallel
    workflow = (
        researcher("AI trends")
        & researcher("ML trends")
        & researcher("Crypto trends")
    )
    results = await workflow.run()
    print(f"Results: {len(results)} parallel executions")

    print("\n=== Combined: Parallel then Sequential ===")
    # Fan-out (parallel) then fan-in (sequential)
    workflow = (
        researcher("AI") & researcher("ML") & researcher("Crypto")
    ) >> analyst("Compare all trends")

    result = await workflow.run()
    print(f"Combined result: {result}")

    print("\n=== Using Lambda for Control ===")
    # Lambda for precise data flow control
    workflow = (
        researcher("Find top AI companies")
        >> analyst(lambda prev: f"Extract company names from: {prev}")
        >> writer(lambda prev: f"Create report about: {prev}")
    )
    result = await workflow.run()
    print(f"Lambda workflow result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
