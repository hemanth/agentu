"""Workspace — load agents from .agentu/ directories.

Demonstrates the filesystem-first pattern: define your agent in YAML,
drop tools as .py files, and context as .md files.
"""
import asyncio
import os
import tempfile
from pathlib import Path


def create_example_workspace(base: str) -> str:
    """Create a sample .agentu/ workspace directory."""
    ws = Path(base) / ".agentu"
    ws.mkdir(parents=True, exist_ok=True)

    # agent.yaml — the declarative source of truth
    (ws / "agent.yaml").write_text("""
name: research-assistant
model: gemini-2.5-flash
temperature: 0.3
system_prompt: "You are a helpful research assistant."

tools:
  discover: ./tools

context:
  files:
    - ./context/guidelines.md

memory:
  enabled: true
  short_term_size: 20

inbox:
  watch: ./inbox
  poll_interval: 10
""".lstrip())

    # Tools — any public function with a docstring becomes a tool
    tools_dir = ws / "tools"
    tools_dir.mkdir(exist_ok=True)

    (tools_dir / "research.py").write_text('''
def search_papers(query: str, max_results: int = 5) -> dict:
    """Search for academic papers on a topic.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.

    Returns:
        dict with list of paper titles and abstracts.
    """
    # In production, call an API like Semantic Scholar
    return {
        "papers": [
            {"title": f"Paper about {query}", "abstract": "..."}
        ],
        "count": 1,
    }


def summarize_text(text: str) -> str:
    """Summarize a long piece of text into key points.

    Args:
        text: The text to summarize.

    Returns:
        A concise summary.
    """
    return f"Summary of {len(text)} characters: {text[:100]}..."
'''.lstrip())

    # Context — loaded into the system prompt
    ctx_dir = ws / "context"
    ctx_dir.mkdir(exist_ok=True)

    (ctx_dir / "guidelines.md").write_text("""
# Research Guidelines

- Always cite sources
- Prefer peer-reviewed papers
- Summarize findings in plain language
- Flag any uncertainty or conflicting evidence
""".lstrip())

    # Inbox — for file watching
    inbox_dir = ws / "inbox"
    inbox_dir.mkdir(exist_ok=True)

    return str(ws)


async def main():
    from agentu import Agent

    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as tmp:
        ws_path = create_example_workspace(tmp)
        print(f"Workspace: {ws_path}\n")

        # Load the agent from the workspace
        agent = await Agent.from_workspace(ws_path)

        print(f"Name:        {agent.name}")
        print(f"Model:       {agent.model}")
        print(f"Temperature: {agent.temperature}")
        print(f"Context:     {len(agent.context)} chars")
        print(f"Tools:       {[t.name for t in agent.tools]}")
        print(f"Inbox:       {getattr(agent, '_inbox_path', 'not set')}")

        # The agent is fully configured — chain on top for runtime config
        agent.with_consolidation(every=60)
        print(f"Tools after: {[t.name for t in agent.tools]}")

        # Tools are callable
        tool = next(t for t in agent.tools if t.name == "search_papers")
        result = tool.function("machine learning")
        print(f"\nSearch result: {result}")

        print("\n✓ Agent loaded from workspace successfully")


if __name__ == "__main__":
    asyncio.run(main())
