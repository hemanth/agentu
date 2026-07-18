"""Agent with memory — basic and structured.

Shows both the classic remember/recall pattern and the new
auto-extraction that enriches memories with entities, topics,
and summaries.
"""
import asyncio
from agentu import Agent


async def main():
    agent = Agent("assistant", enable_memory=True)

    # Classic: store facts with manual importance
    agent.remember("User prefers email", memory_type="fact", importance=0.8)
    agent.remember("Customer ordered item #12345", memory_type="conversation")

    # New: LLM auto-extracts entities, topics, summary, and importance
    agent.remember("Acme Corp signed a $2M deal with BigTech for cloud services")
    # → entities: ["Acme Corp", "BigTech"], topics: ["deals", "cloud"], importance: 0.8

    # Manual override — skips the LLM extraction call
    agent.remember(
        "Board meeting next Tuesday",
        entities=["Board"],
        topics=["meetings", "scheduling"],
        source="calendar.ics",
    )

    # Recall
    memories = agent.recall(query="email", limit=5)
    for mem in memories:
        print(f"- {mem.content}")
        if mem.entities:
            print(f"  entities: {mem.entities}")

    # Stats
    stats = agent.get_memory_stats()
    print(f"\nShort-term: {stats['short_term_size']}")
    print(f"Long-term: {stats['long_term_size']}")


if __name__ == "__main__":
    asyncio.run(main())
