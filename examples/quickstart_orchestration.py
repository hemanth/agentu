"""Quick start example for multi-agent orchestration."""

import asyncio
from agentu.orchestrator import (
    Orchestrator,
    AgentRole,
    ExecutionMode,
    Task,
    make_agent
)


async def main():
    """Run a simple multi-agent orchestration example."""

    print("Creating Multi-Agent System...")
    print("="*60)

    # Create orchestrator
    orchestrator = Orchestrator(
        name="Research Team",
        execution_mode=ExecutionMode.SEQUENTIAL
    )

    # Create specialized agents
    print("\n1. Creating specialized agents...")

    researcher = make_agent("Dr. Research", AgentRole.RESEARCHER, model="llama2", enable_memory=False)
    print(f"   - Created {researcher.name} (Role: {researcher.role})")
    print(f"     Skills: {', '.join(researcher.skills)}")

    analyst = make_agent("Data Analyst", AgentRole.ANALYST, model="llama2", enable_memory=False)
    print(f"   - Created {analyst.name} (Role: {analyst.role})")
    print(f"     Skills: {', '.join(analyst.skills)}")

    writer = make_agent("Tech Writer", AgentRole.WRITER, model="llama2", enable_memory=False)
    print(f"   - Created {writer.name} (Role: {writer.role})")
    print(f"     Skills: {', '.join(writer.skills)}")

    # Add agents
    print("\n2. Adding agents to orchestrator...")
    orchestrator.add_agent(researcher)
    orchestrator.add_agent(analyst)
    orchestrator.add_agent(writer)
    print(f"   - Added {len(orchestrator.agents)} agents")

    # Create tasks
    print("\n3. Creating tasks...")
    tasks = [
        Task(
            description="Research the latest trends in quantum computing",
            required_skills=["research", "search", "analyze"],
            metadata={"priority": "high"}
        ),
        Task(
            description="Analyze the key findings and identify business opportunities",
            required_skills=["analyze", "evaluate", "statistics"],
            metadata={"priority": "medium"}
        ),
        Task(
            description="Write a executive summary of the analysis",
            required_skills=["write", "document", "compose"],
            metadata={"priority": "high"}
        )
    ]

    for i, task in enumerate(tasks):
        print(f"   Task {i+1}: {task.description[:50]}...")
        print(f"           Required skills: {', '.join(task.required_skills)}")

    # Show task routing
    print("\n4. Task routing preview...")
    for i, task in enumerate(tasks):
        agent_name = orchestrator.route_task(task)
        print(f"   Task {i+1} -> {agent_name}")

    # Execute tasks
    print("\n5. Executing tasks sequentially...")
    print("   (Note: This requires Ollama running locally)")
    print("   Skipping actual execution in demo mode...")

    # In real usage, you would do:
    # results = await orchestrator.execute(tasks)
    # for result in results:
    #     print(f"\n   Result: {result}")

    # Show statistics
    print("\n6. Orchestrator statistics:")
    stats = orchestrator.get_stats()
    print(f"   - Name: {stats['name']}")
    print(f"   - Execution mode: {stats['execution_mode']}")
    print(f"   - Registered agents: {stats['registered_agents']}")
    print(f"   - Agents: {', '.join(stats['agents'])}")
    print(f"   - Agent roles:")
    for agent, role in stats['agent_roles'].items():
        print(f"     * {agent}: {role}")

    print("\n" + "="*60)
    print("Multi-Agent System Ready!")
    print("\nTo execute tasks, ensure Ollama is running and call:")
    print("  results = await orchestrator.execute(tasks)")
    print("\nFor more examples, see examples/multi_agent_example.py")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
