import json

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
    GeneratedImage,
    ImageGenerationResponse,
)
from agentu import Agent


class TestDrivers:
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
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

    def test_provider_document_conversion(self):
        messages = [{
            "role": "user",
            "content": [{
                "type": "document",
                "document": {
                    "url": "data:application/pdf;base64,JVBERg==",
                    "mime_type": "application/pdf",
                },
            }],
        }]

        _, gemini = GeminiDriver()._convert_messages(messages)
        assert gemini[0]["parts"][0]["inline_data"]["mime_type"] == "application/pdf"

        _, claude = ClaudeDriver()._convert_messages(messages)
        assert claude[0]["content"][0]["type"] == "document"
        assert claude[0]["content"][0]["source"]["media_type"] == "application/pdf"

    def test_gemini_converts_and_parses_tools(self):
        converted = GeminiDriver._convert_tools([self.tool])
        declaration = converted[0]["functionDeclarations"][0]
        assert declaration["name"] == "get_weather"
        assert declaration["parametersJsonSchema"]["required"] == ["city"]

        calls = GeminiDriver._parse_tool_calls([{
            "functionCall": {
                "id": "call-1",
                "name": "get_weather",
                "args": {"city": "Paris"},
            }
        }])
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_claude_converts_and_parses_tools(self):
        converted = ClaudeDriver._convert_tools([self.tool])
        assert converted[0]["name"] == "get_weather"
        assert converted[0]["input_schema"]["required"] == ["city"]

        calls = ClaudeDriver._parse_tool_calls([{
            "type": "tool_use",
            "id": "toolu-1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }])
        assert calls[0]["id"] == "toolu-1"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_provider_message_conversion_preserves_tool_round_trip(self):
        assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                },
            }],
        }
        result = {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "sunny",
        }

        _, gemini_messages = GeminiDriver()._convert_messages([assistant, result])
        assert gemini_messages[0]["parts"][0]["functionCall"]["name"] == "get_weather"
        assert gemini_messages[1]["parts"][0]["functionResponse"]["name"] == "get_weather"

        _, claude_messages = ClaudeDriver()._convert_messages([assistant, result])
        assert claude_messages[0]["content"][0]["type"] == "tool_use"
        assert claude_messages[1]["content"][0]["type"] == "tool_result"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("driver", "patch_target", "payload", "request_tools_key"),
        [
            (
                GeminiDriver(api_key="key"),
                "agentu.drivers.gemini_driver.aiohttp.ClientSession",
                {
                    "candidates": [{"content": {"parts": [{
                        "functionCall": {
                            "id": "call-1",
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                        }
                    }]}}]
                },
                "functionDeclarations",
            ),
            (
                ClaudeDriver(api_key="key"),
                "agentu.drivers.claude_driver.aiohttp.ClientSession",
                {
                    "content": [{
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    }]
                },
                None,
            ),
        ],
    )
    async def test_provider_calls_send_tools_and_parse_response(
        self, driver, patch_target, payload, request_tools_key
    ):
        response = AsyncMock()
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(return_value=payload)
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post.return_value = response
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        with patch(patch_target, return_value=session):
            result = await driver.call(
                [{"role": "user", "content": "Weather?"}],
                tools=[self.tool],
            )

        sent_tools = session.post.call_args.kwargs["json"]["tools"]
        if request_tools_key:
            assert sent_tools[0][request_tools_key][0]["name"] == "get_weather"
        else:
            assert sent_tools[0]["name"] == "get_weather"
        assert result.tool_calls[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_agent_uses_injected_driver_for_calls(self):
        driver = OpenAIDriver(model="gpt-4o", api_base="https://provider.test/v1")
        driver.call = AsyncMock(return_value=DriverResponse(text="provider response"))
        agent = Agent(
            "driver-agent",
            driver=driver,
            enable_memory=False,
            auto_discover_rules=False,
        )

        response = await agent._raw_llm_call(
            "Describe this",
            images=["https://example.com/image.png"],
        )

        assert response == "provider response"
        messages = driver.call.await_args.args[0]
        assert messages[0]["content"][1]["type"] == "image_url"
        assert agent.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_agent_uses_injected_driver_for_streaming(self):
        driver = OpenAIDriver(model="provider-model")

        async def chunks(*args, **kwargs):
            for chunk in ("provider ", "stream"):
                yield chunk

        driver.stream = chunks
        agent = Agent(
            "driver-agent",
            driver=driver,
            enable_memory=False,
            auto_discover_rules=False,
        )

        received = [chunk async for chunk in agent._stream_llm("Hello")]

        assert received == ["provider ", "stream"]

    @pytest.mark.asyncio
    async def test_openai_image_generation_normalizes_inline_data(self):
        response = AsyncMock()
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(return_value={
            "data": [{
                "b64_json": "aW1hZ2UtYnl0ZXM=",
                "revised_prompt": "A refined prompt",
            }]
        })
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post.return_value = response
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        driver = OpenAIDriver(api_base="https://images.test/v1", api_key="key")

        with patch("agentu.drivers.openai_driver.aiohttp.ClientSession", return_value=session):
            result = await driver.generate_image(
                "A paper-cut fox",
                model="gpt-image-2",
                size="1024x1024",
                quality="high",
            )

        request = session.post.call_args
        assert request.args[0] == "https://images.test/v1/images/generations"
        assert request.kwargs["json"]["size"] == "1024x1024"
        assert result.images[0].data == b"image-bytes"
        assert result.images[0].revised_prompt == "A refined prompt"

    def test_generated_image_save_is_non_destructive(self, tmp_path):
        image = GeneratedImage(data=b"png")
        destination = tmp_path / "nested" / "image.png"

        assert image.save(destination) == destination
        with pytest.raises(FileExistsError):
            image.save(destination)

    @pytest.mark.asyncio
    async def test_agent_delegates_image_generation_to_driver(self):
        driver = OpenAIDriver(model="gpt-4o")
        expected = ImageGenerationResponse(images=[GeneratedImage(url="https://example.com/image.png")])
        driver.generate_image = AsyncMock(return_value=expected)
        agent = Agent(
            "image-agent",
            driver=driver,
            enable_memory=False,
            auto_discover_rules=False,
        )

        result = await agent.generate_image("A lighthouse", model="gpt-image-2")

        assert result is expected.images[0]
        driver.generate_image.assert_awaited_once_with(
            "A lighthouse", model="gpt-image-2", n=1
        )

    @pytest.mark.asyncio
    async def test_agent_can_generate_and_save_in_one_call(self, tmp_path):
        driver = OpenAIDriver(model="gpt-4o")
        driver.generate_image = AsyncMock(return_value=ImageGenerationResponse(
            images=[GeneratedImage(data=b"png")]
        ))
        agent = Agent(
            "image-agent",
            driver=driver,
            enable_memory=False,
            auto_discover_rules=False,
        )
        destination = tmp_path / "generated.png"

        image = await agent.generate_image("A lighthouse", output_path=destination)

        assert image.data == b"png"
        assert destination.read_bytes() == b"png"

    @pytest.mark.asyncio
    async def test_agent_uses_explicit_plural_api_for_batches(self):
        driver = OpenAIDriver(model="gpt-4o")
        expected = ImageGenerationResponse(
            images=[GeneratedImage(url="one"), GeneratedImage(url="two")]
        )
        driver.generate_image = AsyncMock(return_value=expected)
        agent = Agent(
            "image-agent",
            driver=driver,
            enable_memory=False,
            auto_discover_rules=False,
        )

        result = await agent.generate_images("Two lighthouses", n=2)

        assert result is expected
        assert len(result.images) == 2

    @pytest.mark.asyncio
    async def test_gemini_driver_passes_api_key_in_header_not_url(self):
        response = AsyncMock()
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(return_value={
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
        })
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post.return_value = response

        driver = GeminiDriver(api_key="super-secret-key")
        with patch("agentu.drivers.gemini_driver.aiohttp.ClientSession", return_value=session):
            await driver.call([{"role": "user", "content": "hi"}])

        called_url = session.post.call_args.args[0]
        called_headers = session.post.call_args.kwargs["headers"]
        assert "super-secret-key" not in called_url
        assert "key=" not in called_url
        assert called_headers["x-goog-api-key"] == "super-secret-key"

    def test_gemini_and_claude_preserve_explicit_media_metadata(self):
        from agentu._core.multimodal import build_content_parts

        content = build_content_parts(
            "Analyze clip",
            media=[{"type": "video", "url": "https://youtu.be/7Z5Vy9JBANs", "processing": "agentic"}],
            capabilities={"image": True, "audio": True, "video": True, "document": True},
        )
        _, gemini_contents = GeminiDriver()._convert_messages([{"role": "user", "content": content}])
        video_part = gemini_contents[0]["parts"][1]
        assert video_part["file_data"]["file_uri"] == "https://youtu.be/7Z5Vy9JBANs"
        assert video_part["processing"] == "agentic"

    @pytest.mark.asyncio
    async def test_driver_4xx_error_is_not_retried(self):
        import aiohttp

        driver = OpenAIDriver(model="gpt-4o")
        req_info = MagicMock()
        err_401 = aiohttp.ClientResponseError(
            request_info=req_info,
            history=(),
            status=401,
            message="Unauthorized",
        )
        driver.call = AsyncMock(side_effect=err_401)
        agent = Agent("driver-agent", driver=driver, enable_memory=False, auto_discover_rules=False)

        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            await agent._raw_llm_call("Hello", max_retries=2)

        assert exc_info.value.status == 401
        assert driver.call.await_count == 1

    @pytest.mark.asyncio
    async def test_driver_session_reuse_and_close(self):
        driver = OpenAIDriver(model="gpt-4o")
        s1 = await driver._get_session()
        s2 = await driver._get_session()
        assert s1 is s2
        assert not s1.closed
        await driver.close()
        assert s1.closed
        assert driver._session is None

