"""Always-on memory — structured extraction, consolidation, inbox.

Demonstrates the three memory patterns inspired by Google's always-on-memory-agent:
1. Structured extraction — LLM auto-extracts entities, topics, summary
2. Background consolidation — periodic memory review and pattern discovery
3. Inbox file watcher — drop files, agent ingests them
"""
import asyncio
import os
from agentu import Agent


async def structured_extraction():
    """The LLM auto-extracts metadata on every remember() call."""
    print("=== Structured Extraction ===\n")

    agent = Agent("researcher", enable_memory=True)

    # Just pass raw text — the LLM extracts entities, topics, summary, importance
    agent.remember("Anthropic reports 62% of Claude usage is code-related")
    agent.remember("OpenAI launched GPT-5 with native tool use")
    agent.remember("Google released Gemini 3.1 Flash-Lite for always-on agents")

    # Recall and inspect the structured fields
    memories = agent.recall(limit=3, include_short_term=True)
    for mem in memories:
        print(f"Content:    {mem.content[:60]}...")
        print(f"Entities:   {mem.entities}")
        print(f"Topics:     {mem.topics}")
        print(f"Importance: {mem.importance}")
        print(f"Summary:    {mem.summary}")
        print()

    # Conversation turns skip extraction (no point tagging chat)
    agent.remember("user said hello", memory_type="conversation")

    # Manual override — skips the LLM call
    agent.remember(
        "Custom data point",
        entities=["Custom Corp"],
        topics=["business"],
        importance=0.9,
    )
    print("Manual entry entities:", agent.recall(limit=1, include_short_term=True)[0].entities)


async def consolidation():
    """Background consolidation — like the brain during sleep."""
    print("\n=== Background Consolidation ===\n")

    agent = Agent("analyst", enable_memory=True)

    # Enable consolidation every 60 minutes
    agent.with_consolidation(every=60)

    # The consolidate_memories tool is now available
    tool_names = {t.name for t in agent.tools}
    print(f"Tools: {tool_names}")
    assert "consolidate_memories" in tool_names

    # Store some memories for consolidation
    agent.remember("AI agents are growing fast but reliability is a challenge")
    agent.remember("Q1 priority: reduce inference costs by 40%")
    agent.remember("Current LLM memory approaches all have gaps")

    # In production, the consolidation runs on a timer.
    # Here we call the tool directly to demonstrate:
    tool = next(t for t in agent.tools if t.name == "consolidate_memories")
    result = tool.function(
        insight="AI growth and cost reduction are connected: reliable agents need efficient inference",
        related_topics=["AI", "cost", "reliability"],
        source_summaries=["AI agents growing", "reduce costs", "memory gaps"],
    )
    print(f"Consolidation: {result['status']}")
    print(f"Insight: {result['insight']}")

    # The insight is stored as a high-importance memory
    insights = agent.recall(memory_type="consolidation", include_short_term=True)
    print(f"Stored insights: {len(insights)}")


async def inbox_watcher():
    """Inbox file watcher — drop files, agent processes them."""
    print("\n=== Inbox File Watcher ===\n")

    # Create a temporary inbox
    inbox = "/tmp/agentu-inbox-demo"
    os.makedirs(inbox, exist_ok=True)

    agent = Agent("ingestor", enable_memory=True)
    agent.with_inbox(inbox)

    print(f"Inbox path: {agent._inbox_path}")
    print(f"Poll interval: {agent._inbox_poll_interval}s")

    # Drop a file in the inbox
    with open(os.path.join(inbox, "note.txt"), "w") as f:
        f.write("Important: The Q2 board meeting is on July 25th.\n")

    # In production, start_inbox() runs the watcher loop.
    # Here we just poll once:
    await agent._poll_inbox()

    # Check that the file was moved to .processed/
    processed = os.path.join(inbox, ".processed")
    if os.path.exists(os.path.join(processed, "note.txt")):
        print("✓ File processed and moved to .processed/")
    else:
        print("Note: File processing requires a running LLM")

    # Cleanup
    import shutil
    shutil.rmtree(inbox, ignore_errors=True)


async def workspace_config():
    """All three patterns configurable via agent.yaml."""
    print("\n=== Workspace Configuration ===\n")

    yaml_example = """
# .agentu/agent.yaml
name: always-on-agent
model: gemini-2.5-flash

# LLM auto-extracts entities, topics, summary
# (enabled by default, disable with auto_extract_memory: false)

# Inbox — drop files here, agent processes them
inbox:
  watch: ./inbox
  poll_interval: 10

# Tools discovered from directory
tools:
  discover: ./tools

# Context loaded into system prompt
context:
  files:
    - ./context/company-docs.md
    """
    print(yaml_example.strip())
    print()
    print('Then: agent = await Agent.from_workspace(".agentu/")')
    print('      agent.with_consolidation(every=30)')


async def main():
    await structured_extraction()
    await consolidation()
    await inbox_watcher()
    await workspace_config()


if __name__ == "__main__":
    asyncio.run(main())
