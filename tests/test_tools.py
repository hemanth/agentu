import pytest
from agentu import Tool

def test_tool_creation():
    def dummy_function(x: int) -> int:
        return x * 2

    tool = Tool(
        name="dummy",
        description="Dummy tool",
        function=dummy_function,
        parameters={"x": "int: Input number"}
    )
    
    assert tool.name == "dummy"
    assert tool.description == "Dummy tool"
    assert tool.function(2) == 4
    assert "x" in tool.parameters
