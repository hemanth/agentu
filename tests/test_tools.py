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


def test_tool_decorator_and_json_schema():
    from enum import Enum
    from typing import Annotated, Literal, Optional
    from pydantic import BaseModel
    from agentu import tool, ToolPermission

    class SortOrder(Enum):
        ASC = "asc"
        DESC = "desc"

    class FilterSpec(BaseModel):
        min_price: float
        in_stock: bool = True

    @tool(readonly=True)
    def search_catalog(
        query: Annotated[str, "Keywords to match"],
        limit: int = 10,
        category: Literal["laptop", "phone"] = "laptop",
        order: SortOrder = SortOrder.ASC,
        filters: Optional[FilterSpec] = None,
    ) -> list[str]:
        """Search product catalog.

        Args:
            limit: Maximum items to return
        """
        return [f"{query}:{limit}"]

    assert search_catalog.name == "search_catalog"
    assert search_catalog.permission == ToolPermission.READONLY
    # Directly callable!
    assert search_catalog("macbook", limit=3) == ["macbook:3"]

    openai_def = search_catalog.to_openai_tool()
    assert openai_def["type"] == "function"
    fn = openai_def["function"]
    assert fn["name"] == "search_catalog"
    schema = fn["parameters"]
    assert schema["type"] == "object"
    assert schema["required"] == ["query"]
    assert schema["properties"]["query"] == {"type": "string", "description": "Keywords to match"}
    assert schema["properties"]["limit"]["type"] == "integer"
    assert schema["properties"]["limit"]["description"] == "Maximum items to return"
    assert schema["properties"]["limit"]["default"] == 10
    assert schema["properties"]["category"]["enum"] == ["laptop", "phone"]
    assert schema["properties"]["order"]["enum"] == ["asc", "desc"]
    assert "properties" in schema["properties"]["filters"]


def test_bare_tool_decorator():
    from agentu import tool, ToolPermission

    @tool
    def ping() -> str:
        """Health check."""
        return "pong"

    assert ping.permission == ToolPermission.WRITE
    assert ping() == "pong"
