"""Structured output example — get typed, validated results from LLM."""

import asyncio
from pydantic import BaseModel
from agentu import Agent


class Sentiment(BaseModel):
    label: str      # "positive" | "negative" | "neutral"
    score: float    # 0.0 to 1.0
    reasoning: str


async def main():
    agent = Agent("analyst")

    result = await agent.infer(
        "Analyze the sentiment: 'I absolutely love this product, best purchase ever!'",
        output_schema=Sentiment,
    )

    sentiment: Sentiment = result["structured"]
    print(f"Label:     {sentiment.label}")
    print(f"Score:     {sentiment.score}")
    print(f"Reasoning: {sentiment.reasoning}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
