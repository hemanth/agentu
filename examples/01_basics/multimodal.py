"""Multi-modal example — send images alongside text prompts."""

import asyncio
from agentu import Agent


async def main():
    # Use a vision-capable model (e.g., llava, llama3.2-vision)
    agent = Agent("vision", model="llava:latest")

    # From URL
    result = await agent.infer(
        "What's in this image? Describe it briefly.",
        images=["https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"],
    )
    print("URL result:", result.get("result", result))

    # From local file
    # result = await agent.infer(
    #     "Describe this diagram",
    #     images=["./diagram.png"],
    # )
    # print("File result:", result.get("result", result))

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
