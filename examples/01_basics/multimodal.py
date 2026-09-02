"""Multi-modal example — send mixed media (images, audio, video) alongside text prompts."""

import asyncio
from agentu import Agent


async def main():
    # Use a multimodal-capable model (e.g. Gemini, GPT-4o, LLaVA)
    agent = Agent("multimodal-agent", model="gemini-2.5-flash")

    # 1. Unified `media` list with image URL
    result = await agent.infer(
        "What's in this image? Describe it briefly.",
        media=["https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"],
    )
    print("Image result:", result.get("result", result))

    # 2. Video URL (YouTube, Vimeo, or .mp4)
    # result = await agent.infer(
    #     "What are the key points in this video?",
    #     media=["https://youtu.be/7Z5Vy9JBANs"],
    # )

    # 3. Explicit dict with custom provider options
    # result = await agent.infer(
    #     "Analyze video with agentic processing:",
    #     media=[{"type": "video", "url": "https://youtu.be/7Z5Vy9JBANs", "processing": "agentic"}],
    # )

    # 4. Backward-compatible `images` kwarg still works
    # result = await agent.infer("Describe", images=["./chart.png"])

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())

