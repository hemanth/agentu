import pytest
from agentu import Agent, Tool

def test_agent_creation():
    agent = Agent("test_agent")
    assert agent.name == "test_agent"
    assert agent.model == "llama2"
    assert len(agent.tools) == 0

def test_with_tools():
    def dummy_tool(x: int) -> int:
        return x * 2

    agent = Agent("test_agent")
    tool = Tool(
        name="dummy",
        description="Dummy tool",
        function=dummy_tool,
        parameters={"x": "int: Input number"}
    )

    agent.with_tools([tool])
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "dummy"


def test_with_tools_auto_wrap():
    """Test that add_tool auto-wraps raw functions."""
    agent = Agent(name="test_agent")

    def my_function(x: int, y: str) -> dict:
        """A test function."""
        return {"x": x, "y": y}

    # Pass function directly - should auto-wrap
    agent.with_tools([my_function])

    assert len(agent.tools) == 1
    assert agent.tools[0].name == "my_function"
    assert agent.tools[0].description == "A test function."
    assert "x" in agent.tools[0].parameters
    assert "y" in agent.tools[0].parameters


def test_with_toolss_auto_wrap():
    """Test that add_tools auto-wraps raw functions."""
    agent = Agent(name="test_agent")

    def func1(x: int) -> int:
        """Function 1."""
        return x

    def func2(y: str) -> str:
        """Function 2."""
        return y

    # Pass functions directly - should auto-wrap
    agent.with_tools([func1, func2])

    assert len(agent.tools) == 2
    assert agent.tools[0].name == "func1"
    assert agent.tools[1].name == "func2"


def test_with_toolss_mixed():
    """Test that add_tools handles both Tool objects and raw functions."""
    agent = Agent(name="test_agent")

    def raw_func(x: int) -> int:
        """Raw function."""
        return x

    def another_func(y: str) -> str:
        """Another function."""
        return y

    wrapped_tool = Tool(another_func, "Custom description")

    # Mix of raw function and Tool object
    agent.with_tools([raw_func, wrapped_tool])

    assert len(agent.tools) == 2
    assert agent.tools[0].name == "raw_func"
    assert agent.tools[0].description == "Raw function."
    assert agent.tools[1].name == "another_func"
    assert agent.tools[1].description == "Custom description"  # Custom override


def test_with_tools_invalid_type():
    """Test that add_tool raises TypeError for invalid types."""
    agent = Agent(name="test_agent")

    with pytest.raises(TypeError, match="Expected Tool or callable"):
        agent.with_tools(["not a function"])

    with pytest.raises(TypeError, match="Expected Tool or callable"):
        agent.with_tools([123])


def test_with_toolss_invalid_type():
    """Test that add_tools raises TypeError for invalid types in list."""
    agent = Agent(name="test_agent")

    def valid_func(x: int) -> int:
        return x

    with pytest.raises(TypeError, match="Expected Tool or callable"):
        agent.with_tools([valid_func, "not a function"])
