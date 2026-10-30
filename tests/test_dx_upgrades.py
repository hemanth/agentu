"""Tests for DX, zero-config routing, plain-function tools + output_type, and packaging."""

from pathlib import Path
from unittest.mock import AsyncMock
import pytest

import agentu
from agentu import (
    Agent,
    ClaudeDriver,
    DriverResponse,
    GeminiDriver,
    InferResult,
    OllamaDriver,
    OpenAIDriver,
)


class WeatherReport:
    city: str
    temp_f: int
    summary: str


class TestZeroConfigRouting:
    def test_cloud_models_auto_route_without_manual_driver(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
        monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")

        a_openai = Agent("a1", model="gpt-4o", enable_memory=False, auto_discover_rules=False)
        assert isinstance(a_openai.driver, OpenAIDriver)
        assert a_openai.api_base == "https://api.openai.com/v1"
        assert a_openai.api_key == "sk-test-openai"

        a_prefixed = Agent("a2", model="openai/gpt-4o-mini", enable_memory=False, auto_discover_rules=False)
        assert isinstance(a_prefixed.driver, OpenAIDriver)
        assert a_prefixed.model == "gpt-4o-mini"
        assert a_prefixed.api_base == "https://api.openai.com/v1"

        a_gemini = Agent("a3", model="gemini-2.5-flash", enable_memory=False, auto_discover_rules=False)
        assert isinstance(a_gemini.driver, GeminiDriver)
        assert a_gemini.api_key == "gem-test"

        a_claude = Agent("a4", model="claude-sonnet-4-20250514", enable_memory=False, auto_discover_rules=False)
        assert isinstance(a_claude.driver, ClaudeDriver)
        assert a_claude.api_key == "ant-test"

        a_groq = Agent("a5", model="groq/llama-3.3-70b-versatile", enable_memory=False, auto_discover_rules=False)
        assert isinstance(a_groq.driver, OpenAIDriver)
        assert a_groq.model == "llama-3.3-70b-versatile"
        assert a_groq.api_base == "https://api.groq.com/openai/v1"

        a_ds = Agent("a6", model="deepseek-chat", enable_memory=False, auto_discover_rules=False)
        assert isinstance(a_ds.driver, OpenAIDriver)
        assert a_ds.api_base == "https://api.deepseek.com/v1"

        a_ollama = Agent("a7", model="ollama/qwen3:latest", enable_memory=False, auto_discover_rules=False)
        assert isinstance(a_ollama.driver, OllamaDriver)
        assert a_ollama.model == "qwen3:latest"

    def test_env_key_auto_selects_default_model_when_no_local_ollama(self, monkeypatch):
        import urllib.request
        for k in ("AGENTU_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gem-auto")
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(OSError("offline")))

        agent = Agent("zero-cfg", enable_memory=False, auto_discover_rules=False)
        assert isinstance(agent.driver, GeminiDriver)
        assert agent.model == "gemini-2.5-flash"

        monkeypatch.setenv("AGENTU_MODEL", "openai/gpt-4o")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-explicit")
        agent2 = Agent("explicit-env", enable_memory=False, auto_discover_rules=False)
        assert isinstance(agent2.driver, OpenAIDriver)
        assert agent2.model == "gpt-4o"

    def test_custom_api_base_override_is_respected(self):
        agent = Agent(
            "custom-gw",
            model="gpt-4o",
            api_base="http://my-litellm-proxy:4000/v1",
            enable_memory=False,
            auto_discover_rules=False,
        )
        assert agent.driver is None
        assert agent.api_base == "http://my-litellm-proxy:4000/v1"


class TestUnifiedInferResultAndPlainTools:
    @pytest.mark.asyncio
    async def test_zero_tools_conversational_infer_works(self):
        agent = Agent("chat", model="test-model", enable_memory=False, auto_discover_rules=False)
        agent._call_llm = AsyncMock(return_value="Hello! How can I help you today?")

        res = await agent.infer("Hi!")
        assert isinstance(res, InferResult)
        assert res.text == "Hello! How can I help you today?"
        assert "error" not in res
        assert res.success is True

    @pytest.mark.asyncio
    async def test_plain_function_tools_compose_with_output_type(self):
        driver = GeminiDriver(api_key="fake-key", model="gemini-2.5-flash")
        driver.call = AsyncMock(side_effect=[
            DriverResponse(
                text='{"selected_tool": "lookup_weather", "parameters": {"city": "Paris"}, "reasoning": "Need Paris weather"}',
                tool_calls=[],
            ),
            DriverResponse(
                text='{"text_response": "Fetched Paris weather.", "selected_tool": null, "parameters": {}}',
                tool_calls=[],
            ),
            DriverResponse(
                text='{"city": "Paris", "temp_f": 68, "summary": "Mild spring evening"}',
                tool_calls=[],
            ),
        ])

        def lookup_weather(city: str) -> dict:
            """Fetch weather metrics."""
            return {"city": city, "temp_f": 68, "condition": "Mild spring evening"}

        agent = Agent("reporter", driver=driver, enable_memory=False, auto_discover_rules=False).with_tools([lookup_weather])
        res = await agent.infer("Weather report for Paris", output_type=WeatherReport)

        assert isinstance(res, InferResult)
        assert isinstance(res.data, WeatherReport)
        assert res.data.city == "Paris"
        assert res.data.temp_f == 68
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0]["tool_used"] == "lookup_weather"

    def test_no_unrequested_item3_exports(self):
        assert not hasattr(agentu, "tool")
        assert not hasattr(Agent, "run")


class TestPackagingAndLazyExports:
    def test_py_typed_marker_exists(self):
        marker = Path(agentu.__file__).parent / "py.typed"
        assert marker.is_file()

    def test_all_public_symbols_resolve_lazily(self):
        optional_skip = {
            "RedisBackend", "RedisSessionStore", "RedisStorageBackend",
            "TaskQueue", "TaskStatus", "TaskInfo", "LanceDBBackend",
        }
        for sym in agentu.__all__:
            if sym in optional_skip:
                continue
            val = getattr(agentu, sym)
            assert val is not None

    def test_core_dependencies_are_zero(self):
        pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        deps_block = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
        assert deps_block.strip() == ""
