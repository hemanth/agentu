"""Agent with memory."""
import asyncio
from agentu import Agent


async def main():
    agent = Agent("assistant", enable_memory=True)

    # Store facts
    agent.remember("User prefers email", memory_type="fact", importance=0.8)
    agent.remember("Customer ordered item #12345", memory_type="conversation")

    # Recall
    memories = agent.recall(query="email", limit=5)
    for mem in memories:
        print(f"- {mem.content}")

    # Stats
    stats = agent.get_memory_stats()
    print(f"\nShort-term: {stats['short_term_size']}")
    print(f"Long-term: {stats['long_term_size']}")


if __name__ == "__main__":
    asyncio.run(main())
