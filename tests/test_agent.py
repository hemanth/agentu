import pytest
from agentu import Agent, Tool

def test_agent_creation():
    agent = Agent("test_agent")
    assert agent.name == "test_agent"
    assert agent.model == "llama2"
    assert len(agent.tools) == 0

def test_add_tool():
    def dummy_tool(x: int) -> int:
        return x * 2

    agent = Agent("test_agent")
    tool = Tool(
        name="dummy",
        description="Dummy tool",
        function=dummy_tool,
        parameters={"x": "int: Input number"}
    )
    
    agent.add_tool(tool)
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "dummy"
