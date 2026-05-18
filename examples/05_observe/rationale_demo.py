"""Example: Recording Rationale (ADRs)

Demonstrates how an agent can use the record_rationale tool to log
architectural decisions or reasons why it took a specific action.
"""

import asyncio
from agentu import Agent, observe

async def main():
    print("=== Rationale Recording Example ===\n")
    
    # Configure observability to output to console
    observe.configure(
        output="console",
        enabled=True
    )
    
    # Create an agent with rationale recording enabled
    # This automatically adds the 'record_rationale' tool to the agent.
    agent = Agent(
        "architect", 
        enable_memory=True, 
        enable_rationale_recording=True
    )
    
    print("Asking the agent to make an architectural decision...\n")
    
    # The agent will evaluate the prompt and use the record_rationale tool
    prompt = (
        "We are building a highly concurrent web server in Python. "
        "Should we use threading or asyncio? "
        "Make a decision, then use the record_rationale tool to record your "
        "decision and reasoning so future developers understand why."
    )
    
    result = await agent.infer(prompt)
    print(f"\nFinal LLM response: {result.get('result')}\n")
    
    print("=== Memory Recall ===")
    print("Let's see if the rationale is searchable in the agent's memory:")
    memories = agent.recall(query="asyncio", limit=5)
    for mem in memories:
        if mem.memory_type == "rationale":
            print(f"✅ Recalled Rationale -> {mem.content}")
        else:
            print(f"- {mem.content}")

if __name__ == "__main__":
    asyncio.run(main())
