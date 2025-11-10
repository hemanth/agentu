import pytest
from agentu.search import SearchAgent, search_duckduckgo

def test_search_agent_creation():
    agent = SearchAgent("test_search")
    assert agent.name == "test_search"
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "web_search"

@pytest.mark.asyncio
async def test_search_tool():
    results = await search_duckduckgo("python programming", max_results=1)
    assert isinstance(results, list)
    assert len(results) <= 1
    if results and "error" not in results[0]:
        assert all(key in results[0] for key in ["title", "link", "snippet"])

@pytest.mark.skip(reason="Requires OpenAI-compatible inference endpoint")
@pytest.mark.asyncio
async def test_search_agent_search():
    agent = SearchAgent("test_search")
    result = await agent.search("python programming", max_results=1)
    assert "tool_used" in result
    assert result["tool_used"] == "web_search"
    assert "parameters" in result
    assert "query" in result["parameters"]