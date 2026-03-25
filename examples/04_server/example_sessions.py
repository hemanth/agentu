"""Example: Stateful Intelligence with Sessions

This demonstrates how to use agentu's Sessions for Interactions API-style
stateful intelligence where the server automatically remembers everything.
"""

import asyncio
from agentu import Agent, Tool, SessionManager


# Define some tools
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is sunny and 72°F"


def set_reminder(task: str, time: str) -> str:
    """Set a reminder."""
    return f"Reminder set: '{task}' at {time}"


async def main():
    # Create an agent with tools
    agent = Agent(
        name="Assistant",
        model="qwen3:latest",
        enable_memory=True  # Memory enabled for session
    ).with_tools([
        Tool(get_weather),
        Tool(set_reminder)
    ])
    
    # Create session manager
    manager = SessionManager()
    
    # === Example 1: Single stateful conversation ===
    print("=== Creating Session ===")
    session = manager.create_session(
        agent=agent,
        metadata={'user_id': 'user123'}
    )
    
    print(f"Session ID: {session.session_id}\n")
    
    # First turn
    print("User: What's the weather in San Francisco?")
    response1 = await session.send("What's the weather in San Francisco?")
    print(f"Agent: {response1.get('result')}")
    print(f"Turn: {response1['session_info']['turn']}\n")
    
    # Second turn - agent remembers context!
    print("User: What about Los Angeles?")
    response2 = await session.send("What about Los Angeles?")
    print(f"Agent: {response2.get('result')}")
    print(f"Turn: {response2['session_info']['turn']}\n")
    
    # Third turn - implicit context
    print("User: Set a reminder to check it tomorrow")
    response3 = await session.send("Set a reminder to check it tomorrow")
    print(f"Agent: {response3.get('result')}")
    print(f"Turn: {response3['session_info']['turn']}\n")
    
    # Get conversation history
    print("=== Conversation History ===")
    history = session.get_history(limit=10)
    for entry in history:
        print(f"- {entry.content}")
    
    print(f"\nMemory Stats: {response3['session_info']['memory_stats']}\n")
    
    # === Example 2: Multiple concurrent sessions ===
    print("\n=== Multiple Users/Sessions ===")
    
    # User 1
    session1 = manager.create_session(agent, metadata={'user_id': 'alice'})
    await session1.send("I love pizza")
    
    # User 2
    session2 = manager.create_session(agent, metadata={'user_id': 'bob'})
    await session2.send("I prefer sushi")
    
    # Each session maintains separate memory
    r1 = await session1.send("What do I like?")
    r2 = await session2.send("What do I like?")
    
    print(f"Alice's session remembers: {r1.get('reasoning', 'N/A')}")
    print(f"Bob's session remembers: {r2.get('reasoning', 'N/A')}\n")
    
    # === Example 3: Session management ===
    print("=== Session Management ===")
    
    # List all sessions
    all_sessions = manager.list_sessions()
    print(f"Total active sessions: {len(all_sessions)}")
    
    # Get specific session
    retrieved = manager.get_session(session.session_id)
    print(f"Retrieved session: {retrieved.session_id}")
    print(f"Turns in session: {retrieved.turn_count}")
    
    # Save and cleanup
    manager.save_all()
    manager.delete_session(session.session_id)
    
    print(f"\nSessions after cleanup: {len(manager.list_sessions())}")


if __name__ == "__main__":
    asyncio.run(main())
