import pytest
import os
from unittest.mock import patch

from agentu import Agent, AgentConfig
from agentu._core.config import CacheConfig, MCPConfig

@pytest.fixture
def test_yaml_path(tmp_path):
    """Creates a temporary full-featured yaml configuration."""
    yaml_content = """
name: "yaml-bot"
model: "openai/gpt-4o"
system_prompt: "You are configured natively via yaml."
notify:
  - "discord://webhook/id"
cache:
  preset: "offline"
  ttl: 7200
    """
    path = tmp_path / "agent.yaml"
    with open(path, "w") as f:
        f.write(yaml_content)
    return str(path)


def test_schema_parsing(test_yaml_path):
    """Test that pydantic correctly infers and loads the model from file."""
    pytest.importorskip("yaml", reason="PyYAML not installed")
    config = AgentConfig.load(test_yaml_path)
    
    assert config.name == "yaml-bot"
    assert config.model == "openai/gpt-4o"
    assert config.notify == ["discord://webhook/id"]
    assert config.cache.preset == "offline"
    assert config.cache.ttl == 7200
    assert config.mcp is None

@pytest.mark.asyncio
async def test_agent_from_config_factory(test_yaml_path):
    """Test that Agent.from_config parses the schema and applies the builders."""
    pytest.importorskip("yaml", reason="PyYAML not installed")
    pytest.importorskip("apprise", reason="apprise not installed")
    agent = await Agent.from_config(test_yaml_path)
    
    # Base attributes
    assert agent.name == "yaml-bot"
    assert agent.model == "openai/gpt-4o"
    
    # Assert cache builder fired
    assert agent.cache_enabled is True
    assert agent.cache is not None
    
    # Assert notify middleware fired
    assert len(agent._middleware_chain.middlewares) == 1
    assert agent._middleware_chain.middlewares[0].name == "notify"
    assert agent._middleware_chain.middlewares[0].targets == ["discord://webhook/id"]

def test_json_config_fallback(tmp_path):
    import json
    json_path = tmp_path / "agent.json"
    data = {"name": "json_bot", "model": "claude-3", "system_prompt": "hello"}
    with open(json_path, "w") as f:
        json.dump(data, f)
        
    cfg = AgentConfig.load(str(json_path))
    assert cfg.name == "json_bot"
    assert cfg.system_prompt == "hello"
