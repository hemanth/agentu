"""Example demonstrating multi-agent orchestration."""

from agentu import Agent
from agentu.orchestrator import (
    Orchestrator,
    AgentRole,
    ExecutionMode,
    Task,
    create_specialized_agent
)


def example_sequential_execution():
    """Example of sequential task execution with multiple specialized agents."""
    print("\n=== Sequential Execution Example ===\n")

    # Create orchestrator
    orchestrator = Orchestrator(
        name="Sequential Orchestrator",
        execution_mode=ExecutionMode.SEQUENTIAL
    )

    # Create specialized agents
    researcher, researcher_cap = create_specialized_agent(
        name="ResearchBot",
        role=AgentRole.RESEARCHER,
        model="llama2"
    )

    analyst, analyst_cap = create_specialized_agent(
        name="AnalystBot",
        role=AgentRole.ANALYST,
        model="llama2"
    )

    writer, writer_cap = create_specialized_agent(
        name="WriterBot",
        role=AgentRole.WRITER,
        model="llama2"
    )

    # Add agents
    orchestrator.add_agent(researcher, researcher_cap)
    orchestrator.add_agent(analyst, analyst_cap)
    orchestrator.add_agent(writer, writer_cap)

    # Create tasks
    tasks = [
        Task(
            description="Research the latest trends in AI and machine learning",
            required_skills=["research", "search", "analyze"]
        ),
        Task(
            description="Analyze the research findings and identify key insights",
            required_skills=["analyze", "evaluate", "statistics"]
        ),
        Task(
            description="Write a summary report of the analysis",
            required_skills=["write", "document", "compose"]
        )
    ]

    # Execute tasks
    results = orchestrator.execute(tasks)

    # Print results
    for i, result in enumerate(results):
        print(f"\nTask {i+1}:")
        print(f"  Description: {result['task']}")
        print(f"  Assigned to: {result.get('agent', 'N/A')}")
        print(f"  Status: {result['status']}")
        if result['status'] == 'completed':
            print(f"  Result: {result['result']}")

    # Print stats
    print("\n" + "="*50)
    print("Orchestrator Stats:")
    stats = orchestrator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_parallel_execution():
    """Example of parallel task execution with multiple agents."""
    print("\n=== Parallel Execution Example ===\n")

    # Create orchestrator with parallel mode
    orchestrator = Orchestrator(
        name="Parallel Orchestrator",
        execution_mode=ExecutionMode.PARALLEL
    )

    # Create multiple coder agents
    coder1, coder1_cap = create_specialized_agent(
        name="CoderBot1",
        role=AgentRole.CODER,
        model="llama2"
    )

    coder2, coder2_cap = create_specialized_agent(
        name="CoderBot2",
        role=AgentRole.CODER,
        model="llama2"
    )

    coder3, coder3_cap = create_specialized_agent(
        name="CoderBot3",
        role=AgentRole.CODER,
        model="llama2"
    )

    # Add agents
    orchestrator.add_agent(coder1, coder1_cap)
    orchestrator.add_agent(coder2, coder2_cap)
    orchestrator.add_agent(coder3, coder3_cap)

    # Create parallel tasks
    tasks = [
        Task(
            description="Implement a function to calculate fibonacci numbers",
            required_skills=["code", "programming", "implement"]
        ),
        Task(
            description="Implement a function to sort an array using quicksort",
            required_skills=["code", "programming", "implement"]
        ),
        Task(
            description="Implement a function to check if a string is a palindrome",
            required_skills=["code", "programming", "implement"]
        )
    ]

    # Execute tasks in parallel
    results = orchestrator.execute_task_parallel(tasks, max_workers=3)

    # Print results
    print("\nParallel Execution Results:")
    for i, result in enumerate(results):
        print(f"\nTask {i+1}:")
        print(f"  Description: {result['task']}")
        print(f"  Agent: {result.get('agent', 'N/A')}")
        print(f"  Status: {result['status']}")

    print(f"\nTotal tasks completed: {len([r for r in results if r['status'] == 'completed'])}")


def example_hierarchical_execution():
    """Example of hierarchical execution with a manager coordinating workers."""
    print("\n=== Hierarchical Execution Example ===\n")

    orchestrator = Orchestrator(name="Hierarchical Orchestrator")

    # Create manager agent
    manager, manager_cap = create_specialized_agent(
        name="ManagerBot",
        role=AgentRole.COORDINATOR,
        model="llama2"
    )

    # Create worker agents
    researcher, researcher_cap = create_specialized_agent(
        name="ResearchWorker",
        role=AgentRole.RESEARCHER,
        model="llama2"
    )

    analyst, analyst_cap = create_specialized_agent(
        name="AnalystWorker",
        role=AgentRole.ANALYST,
        model="llama2"
    )

    # Add all agents
    orchestrator.add_agent(manager, manager_cap)
    orchestrator.add_agent(researcher, researcher_cap)
    orchestrator.add_agent(analyst, analyst_cap)

    # Create worker tasks
    worker_tasks = [
        Task(
            description="Research current market trends in electric vehicles",
            required_skills=["research", "analyze"]
        ),
        Task(
            description="Analyze the competitive landscape of EV manufacturers",
            required_skills=["analyze", "evaluate"]
        )
    ]

    # Execute hierarchically
    result = orchestrator.execute_hierarchical(
        manager_agent="ManagerBot",
        worker_tasks=worker_tasks
    )

    print("\nHierarchical Execution Result:")
    print(f"Manager: {result['manager']}")
    print(f"\nWorker Results ({len(result['worker_results'])} tasks):")
    for wr in result['worker_results']:
        print(f"  - {wr['task'][:50]}... (Agent: {wr.get('agent', 'N/A')})")

    print("\nManager Summary:")
    print(result['manager_summary'])


def example_debate_mode():
    """Example of debate mode where agents discuss and reach consensus."""
    print("\n=== Debate Mode Example ===\n")

    orchestrator = Orchestrator(name="Debate Orchestrator")

    # Create agents with different perspectives
    critic1, critic1_cap = create_specialized_agent(
        name="Optimist",
        role=AgentRole.CRITIC,
        model="llama2"
    )
    critic1.set_context("You are an optimistic critic who focuses on positive aspects and potential.")

    critic2, critic2_cap = create_specialized_agent(
        name="Realist",
        role=AgentRole.CRITIC,
        model="llama2"
    )
    critic2.set_context("You are a realistic critic who provides balanced, practical perspectives.")

    critic3, critic3_cap = create_specialized_agent(
        name="Skeptic",
        role=AgentRole.CRITIC,
        model="llama2"
    )
    critic3.set_context("You are a skeptical critic who identifies potential risks and challenges.")

    # Add agents
    orchestrator.add_agent(critic1, critic1_cap)
    orchestrator.add_agent(critic2, critic2_cap)
    orchestrator.add_agent(critic3, critic3_cap)

    # Run debate
    topic = "Should companies invest heavily in AI automation despite potential job displacement?"
    result = orchestrator.execute_debate(
        topic=topic,
        agents=["Optimist", "Realist", "Skeptic"],
        rounds=2
    )

    print(f"\nDebate Topic: {result['topic']}")
    print(f"Participants: {', '.join(result['participants'])}")
    print(f"Rounds: {result['rounds']}\n")

    print("Debate History:")
    for round_num, round_data in enumerate(result['debate_history'], 1):
        print(f"\nRound {round_num}:")
        for response in round_data:
            print(f"  {response['agent']}: {str(response['response'])[:100]}...")

    print("\n" + "="*50)
    print("Consensus:")
    print(result['consensus'])


def example_custom_agent_with_tools():
    """Example showing how to add tools to orchestrated agents."""
    print("\n=== Custom Agent with Tools Example ===\n")

    from agentu import Tool

    # Create a custom tool
    def calculate(x: float, y: float, operation: str) -> float:
        """Perform basic calculations."""
        ops = {
            "add": x + y,
            "subtract": x - y,
            "multiply": x * y,
            "divide": x / y if y != 0 else float('inf')
        }
        return ops.get(operation, 0)

    calc_tool = Tool(
        name="calculator",
        description="Perform basic arithmetic operations",
        parameters={
            "x": "First number",
            "y": "Second number",
            "operation": "Operation to perform (add, subtract, multiply, divide)"
        },
        function=calculate
    )

    # Create orchestrator
    orchestrator = Orchestrator(execution_mode=ExecutionMode.PARALLEL)

    # Create specialized agents with tools
    analyst1, analyst1_cap = create_specialized_agent(
        name="MathAnalyst1",
        role=AgentRole.ANALYST,
        model="llama2"
    )
    analyst1.add_tool(calc_tool)

    analyst2, analyst2_cap = create_specialized_agent(
        name="MathAnalyst2",
        role=AgentRole.ANALYST,
        model="llama2"
    )
    analyst2.add_tool(calc_tool)

    # Add agents
    orchestrator.add_agent(analyst1, analyst1_cap)
    orchestrator.add_agent(analyst2, analyst2_cap)

    # Create calculation tasks
    tasks = [
        Task(
            description="Calculate 15 multiplied by 8",
            required_skills=["analyze", "calculate"]
        ),
        Task(
            description="Calculate 100 divided by 4",
            required_skills=["analyze", "calculate"]
        )
    ]

    # Execute
    results = orchestrator.execute(tasks)

    print("\nCalculation Results:")
    for result in results:
        print(f"  {result['task']}: {result.get('result', 'N/A')}")


if __name__ == "__main__":
    print("="*70)
    print("Multi-Agent Orchestration Examples")
    print("="*70)

    # Run examples
    try:
        example_sequential_execution()
    except Exception as e:
        print(f"Sequential example error: {e}")

    try:
        example_parallel_execution()
    except Exception as e:
        print(f"Parallel example error: {e}")

    try:
        example_hierarchical_execution()
    except Exception as e:
        print(f"Hierarchical example error: {e}")

    try:
        example_debate_mode()
    except Exception as e:
        print(f"Debate example error: {e}")

    try:
        example_custom_agent_with_tools()
    except Exception as e:
        print(f"Custom agent example error: {e}")

    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
