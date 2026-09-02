import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agentu.drivers import (
    BaseDriver,
    DriverResponse,
    OpenAIDriver,
    GeminiDriver,
    ClaudeDriver,
    OllamaDriver,
    get_driver_for_model,
)


class TestDrivers:
    def test_auto_detect_gemini(self):
        d = get_driver_for_model("gemini-2.5-flash")
        assert isinstance(d, GeminiDriver)
        assert d.model == "gemini-2.5-flash"

    def test_auto_detect_claude(self):
        d = get_driver_for_model("claude-3-7-sonnet-20250219")
        assert isinstance(d, ClaudeDriver)
        assert d.model == "claude-3-7-sonnet-20250219"

    def test_auto_detect_openai_default(self):
        d = get_driver_for_model("gpt-4o")
        assert isinstance(d, OpenAIDriver)
        assert d.model == "gpt-4o"

    def test_auto_detect_deepseek_as_openai_compatible(self):
        d = get_driver_for_model("deepseek-r1")
        assert isinstance(d, OpenAIDriver)
        assert d.model == "deepseek-r1"

    def test_gemini_message_conversion(self):
        d = GeminiDriver(api_key="test-key")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        sys_inst, contents = d._convert_messages(messages)
        assert sys_inst == {"parts": [{"text": "You are a helpful assistant."}]}
        assert len(contents) == 2
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "Hello!"
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0]["text"] == "Hi there!"

    def test_claude_message_conversion(self):
        d = ClaudeDriver(api_key="test-key")
        messages = [
            {"role": "system", "content": "You are an expert."},
            {"role": "user", "content": "Explain relativity."},
        ]
        sys_text, msgs = d._convert_messages(messages)
        assert sys_text == "You are an expert."
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Explain relativity."
