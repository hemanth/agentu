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


def test_tool_auto_inference():
    """Test that Tool auto-infers name, description, and parameters from function."""
    def calculate_total(price: float, quantity: int, tax_rate: float = 0.1) -> float:
        """Calculate total price including tax."""
        return price * quantity * (1 + tax_rate)

    # Just pass the function - everything auto-inferred!
    tool = Tool(calculate_total)

    # Verify auto-inferred name
    assert tool.name == "calculate_total"

    # Verify auto-inferred description from docstring
    assert tool.description == "Calculate total price including tax."

    # Verify auto-inferred parameters with type hints
    assert "price" in tool.parameters
    assert "quantity" in tool.parameters
    assert "tax_rate" in tool.parameters
    assert "float" in tool.parameters["price"]
    assert "int" in tool.parameters["quantity"]
    assert "default" in tool.parameters["tax_rate"]  # Has default value

    # Verify function still works
    assert tool.function(10.0, 2) == 22.0


def test_tool_auto_inference_no_docstring():
    """Test Tool handles functions without docstrings."""
    def no_docs(x: int, y: str):
        return f"{y}: {x}"

    tool = Tool(no_docs)

    assert tool.name == "no_docs"
    assert tool.description == "Execute no_docs"  # Default description
    assert "x" in tool.parameters
    assert "y" in tool.parameters


def test_tool_custom_override():
    """Test that custom name/description can override auto-inference."""
    def original_name(x: int) -> int:
        """Original description."""
        return x

    tool = Tool(original_name, "Custom description", "custom_name")

    assert tool.name == "custom_name"
    assert tool.description == "Custom description"
