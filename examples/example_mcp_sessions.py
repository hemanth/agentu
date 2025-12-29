"""Example: Stateful MCP Tools with Sessions

This demonstrates how Sessions provide automatic memory/context
for MCP tools, creating a truly stateful integration.

Prerequisites:
1. Run an MCP server (e.g., filesystem MCP server)
2. Configure it in ~/.agentu/mcp_config.json

Example MCP config:
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
"""

import asyncio
from agentu import Agent, SessionManager


async def main():
    print("=== Stateful MCP Tools Example ===\n")
    
    # Create agent with MCP tools loaded
    agent = Agent(
        name="FileAssistant",
        model="qwen3:latest",
        enable_memory=True
    ).with_mcp([
        # Load MCP servers from config
        "~/.agentu/mcp_config.json"
        
        # Or directly specify MCP server URLs:
        # "http://localhost:3000",
    ])
    
    print(f"Agent loaded with {len(agent.tools)} tools\n")
    print("MCP Tools available:")
    for tool in agent.tools:
        print(f"  - {tool.name}")
    print()
    
    # Create stateful session
    manager = SessionManager()
    session = manager.create_session(
        agent=agent,
        metadata={'user_id': 'demo-user'}
    )
    
    print(f"Session created: {session.session_id}\n")
    
    # === Scenario: Multi-turn file operations with context ===
    
    # Turn 1: Create a file using MCP tool
    print("User: Create a file called notes.txt with 'Hello World'")
    response1 = await session.send(
        "Create a file called notes.txt with the content 'Hello World'"
    )
    print(f"Agent: {response1.get('reasoning', 'Done')}")
    print(f"Result: {response1.get('result', 'N/A')}\n")
    
    # Turn 2: Reference previous context implicitly
    print("User: Now add 'Second line' to it")
    response2 = await session.send(
        "Now add 'Second line' to it"  # "it" refers to notes.txt!
    )
    print(f"Agent: {response2.get('reasoning', 'Done')}")
    print(f"Result: {response2.get('result', 'N/A')}\n")
    
    # Turn 3: Query about the file
    print("User: What's in that file now?")
    response3 = await session.send(
        "What's in that file now?"  # Session remembers notes.txt
    )
    print(f"Agent: {response3.get('reasoning', 'Done')}")
    print(f"Result: {response3.get('result', 'N/A')}\n")
    
    # === Show conversation history ===
    print("=== Conversation History ===")
    history = session.get_history(limit=10)
    for entry in history:
        print(f"- [{entry.memory_type}] {entry.content[:60]}...")
    
    print(f"\nMemory Stats: {session.agent.get_memory_stats()}")
    
    # === Demonstrate session isolation ===
    print("\n=== Session Isolation Demo ===")
    
    # Create second session
    session2 = manager.create_session(agent, metadata={'user_id': 'other-user'})
    
    # This session doesn't know about notes.txt
    print("User (Session 2): What file did I just create?")
    response4 = await session2.send("What file did I just create?")
    print(f"Agent: {response4.get('reasoning', 'No memory of previous session')}\n")
    
    # Original session still remembers
    print("User (Session 1): What was the file called?")
    response5 = await session.send("What was the file called?")
    print(f"Agent: {response5.get('reasoning', 'Remembers notes.txt from history')}\n")
    
    # === Advanced: Storing facts in session ===
    print("=== Storing Long-term Facts ===")
    
    # Explicitly store important information
    session.agent.remember(
        "User prefers markdown format for documentation",
        memory_type='fact',
        importance=0.8,
        store_long_term=True
    )
    
    # Later turns can access this fact
    print("User: Create a doc file for me")
    response6 = await session.send("Create a doc file for me")
    print(f"Agent should prefer .md format based on stored fact")
    print(f"Result: {response6.get('result', 'N/A')}\n")
    
    # === Cleanup ===
    print("=== Session Management ===")
    all_sessions = manager.list_sessions()
    print(f"Active sessions: {len(all_sessions)}")
    
    manager.save_all()
    print("All sessions saved to disk\n")
    
    print("Sessions are automatically persisted and can be resumed later!")


async def demo_session_resume():
    """Example: Resume a previous session"""
    print("\n=== Resume Session Example ===\n")
    
    manager = SessionManager()
    
    # Simulate getting session ID from previous interaction
    # (In real app, you'd store this in a database or user session)
    session_id = "previous-session-id"
    
    # Try to get existing session
    session = manager.get_session(session_id)
    
    if session:
        print(f"Resumed session: {session_id}")
        print(f"Previous turns: {session.turn_count}")
        
        # Continue conversation with full context
        response = await session.send("What were we talking about?")
        print(f"Agent remembers: {response.get('reasoning', 'N/A')}")
    else:
        print(f"Session {session_id} not found or expired")


if __name__ == "__main__":
    print(__doc__)
    
    # Run main demo
    asyncio.run(main())
    
    # Uncomment to test session resumption:
    # asyncio.run(demo_session_resume())
