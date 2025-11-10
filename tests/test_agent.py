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
