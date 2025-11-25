import pytest
from unittest.mock import patch, MagicMock
from agentu import Agent, Tool
from agentu.agent import get_ollama_models, get_default_model

def test_get_ollama_models_success():
    """Test successful retrieval of Ollama models."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [
            {"name": "qwen3:latest"},
            {"name": "llama2:latest"},
            {"name": "mistral:latest"}
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch('requests.get', return_value=mock_response) as mock_get:
        models = get_ollama_models("http://localhost:11434")
        assert models == ["qwen3:latest", "llama2:latest", "mistral:latest"]
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=2)


def test_get_ollama_models_failure():
    """Test handling of Ollama API failure."""
    with patch('requests.get', side_effect=Exception("Connection error")):
        models = get_ollama_models("http://localhost:11434")
        assert models == []


def test_get_default_model_with_available_models():
    """Test get_default_model returns first available model."""
    with patch('agentu.agent.get_ollama_models', return_value=["qwen3:latest", "llama2:latest"]):
        model = get_default_model("http://localhost:11434")
        assert model == "qwen3:latest"


def test_get_default_model_no_models():
    """Test get_default_model returns qwen3:latest fallback when no models available."""
    with patch('agentu.agent.get_ollama_models', return_value=[]):
        model = get_default_model("http://localhost:11434")
        assert model == "qwen3:latest"


def test_agent_creation_auto_detect_model():
    """Test agent auto-detects model from Ollama."""
    with patch('agentu.agent.get_ollama_models', return_value=["qwen3:latest", "llama2:latest"]):
        agent = Agent("test_agent")
        assert agent.name == "test_agent"
        assert agent.model == "qwen3:latest"  # Should use first available model
        assert len(agent.tools) == 0


def test_agent_creation_explicit_model():
    """Test agent uses explicit model when provided."""
    with patch('agentu.agent.get_ollama_models', return_value=["qwen3:latest", "llama2:latest"]):
        agent = Agent("test_agent", model="mistral:latest")
        assert agent.name == "test_agent"
        assert agent.model == "mistral:latest"  # Should use explicit model, not auto-detected
        assert len(agent.tools) == 0


def test_agent_creation_fallback_model():
    """Test agent falls back to qwen3:latest when no Ollama models available."""
    with patch('agentu.agent.get_ollama_models', return_value=[]):
        agent = Agent("test_agent")
        assert agent.name == "test_agent"
        assert agent.model == "qwen3:latest"  # Should fall back to qwen3:latest
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


# Tool Search Tests

def test_with_tools_defer():
    """Test adding deferred tools."""
    agent = Agent(name="test_agent")

    def core_func(x: int) -> int:
        """Core function."""
        return x

    def deferred_func(y: str) -> str:
        """Deferred function."""
        return y

    agent.with_tools([core_func], defer=[deferred_func])

    # Core tool + search_tools should be in active tools
    assert len(agent.tools) == 2
    assert agent.tools[0].name == "core_func"
    assert agent.tools[1].name == "search_tools"

    # Deferred tool should be in deferred_tools
    assert len(agent.deferred_tools) == 1
    assert agent.deferred_tools[0].name == "deferred_func"


def test_with_tools_defer_only():
    """Test adding only deferred tools."""
    agent = Agent(name="test_agent")

    def api_func1(x: int) -> int:
        """API function 1."""
        return x

    def api_func2(y: str) -> str:
        """API function 2."""
        return y

    agent.with_tools(defer=[api_func1, api_func2])

    # Only search_tools should be in active tools
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "search_tools"

    # Both should be in deferred_tools
    assert len(agent.deferred_tools) == 2


def test_search_tools_finds_matching():
    """Test that search_tools finds matching deferred tools."""
    agent = Agent(name="test_agent")

    def payment_process(amount: float) -> str:
        """Process a payment transaction."""
        return f"Processed {amount}"

    def user_lookup(user_id: str) -> dict:
        """Look up user information."""
        return {"id": user_id}

    def order_create(items: list) -> str:
        """Create a new order."""
        return "Order created"

    agent.with_tools(defer=[payment_process, user_lookup, order_create])

    # Search for payment-related tools
    result = agent._search_tools("payment process")

    assert "Activated tools: payment_process" in result
    assert agent.deferred_tools[0] in agent.tools  # payment_process now active


def test_search_tools_no_match():
    """Test search_tools returns message when no match found."""
    agent = Agent(name="test_agent")

    def payment_process(amount: float) -> str:
        """Process a payment transaction."""
        return f"Processed {amount}"

    agent.with_tools(defer=[payment_process])

    result = agent._search_tools("something completely unrelated xyz")

    assert result == "No matching tools found."


def test_search_tools_already_active():
    """Test search_tools handles already active tools."""
    agent = Agent(name="test_agent")

    def payment_process(amount: float) -> str:
        """Process a payment transaction."""
        return f"Processed {amount}"

    agent.with_tools(defer=[payment_process])

    # First search activates
    result1 = agent._search_tools("payment")
    assert "Activated tools: payment_process" in result1

    # Second search finds already active
    result2 = agent._search_tools("payment")
    assert "Tools already active: payment_process" in result2


def test_find_matching_tools_scoring():
    """Test that _find_matching_tools scores and ranks correctly."""
    agent = Agent(name="test_agent")

    def payment_process(amount: float) -> str:
        """Process a payment transaction."""
        return f"Processed {amount}"

    def payment_refund(amount: float) -> str:
        """Refund a payment transaction payment."""
        return f"Refunded {amount}"

    def user_lookup(user_id: str) -> dict:
        """Look up user information."""
        return {"id": user_id}

    agent.with_tools(defer=[payment_process, payment_refund, user_lookup])

    # "payment" should match both payment tools, "refund" should boost payment_refund
    matches = agent._find_matching_tools("payment refund", limit=5)

    assert len(matches) == 2
    # payment_refund should rank first (matches both "payment" and "refund")
    assert matches[0].name == "payment_refund"
    assert matches[1].name == "payment_process"


def test_find_matching_tools_limit():
    """Test that _find_matching_tools respects limit."""
    agent = Agent(name="test_agent")

    def api_func1(x: int) -> int:
        """API function."""
        return x

    def api_func2(x: int) -> int:
        """API function."""
        return x

    def api_func3(x: int) -> int:
        """API function."""
        return x

    agent.with_tools(defer=[api_func1, api_func2, api_func3])

    matches = agent._find_matching_tools("api function", limit=2)

    assert len(matches) == 2


def test_max_turns_parameter():
    """Test that max_turns parameter is set correctly."""
    agent1 = Agent(name="test_agent")
    assert agent1.max_turns == 10  # Default

    agent2 = Agent(name="test_agent", max_turns=5)
    assert agent2.max_turns == 5


def test_deferred_tools_initialized():
    """Test that deferred_tools list is initialized."""
    agent = Agent(name="test_agent")
    assert agent.deferred_tools == []


def test_ensure_search_tool_only_added_once():
    """Test that search_tools is only added once."""
    agent = Agent(name="test_agent")

    def func1(x: int) -> int:
        """Function 1."""
        return x

    def func2(y: str) -> str:
        """Function 2."""
        return y

    # Add deferred tools twice
    agent.with_tools(defer=[func1])
    agent.with_tools(defer=[func2])

    # search_tools should only appear once
    search_tools_count = sum(1 for t in agent.tools if t.name == "search_tools")
    assert search_tools_count == 1
