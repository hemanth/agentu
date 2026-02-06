"""Tests for Agent streaming functionality."""

import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agentu import Agent


def make_sse_lines(chunks):
    """Create SSE byte lines from a list of text chunks."""
    lines = []
    for chunk in chunks:
        data = json.dumps({
            "choices": [{"delta": {"content": chunk}, "index": 0}]
        })
        lines.append(f"data: {data}\n\n".encode("utf-8"))
    lines.append(b"data: [DONE]\n\n")
    return lines


class AsyncLineIterator:
    """Async iterator over SSE byte lines."""
    def __init__(self, lines):
        self._lines = lines
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


def _make_mock_session(sse_lines):
    """Create a mock aiohttp session that returns SSE lines."""
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = AsyncLineIterator(sse_lines)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.closed = False
    return mock_session


@pytest.fixture
def agent():
    with patch.object(Agent, '__init__', lambda self, *a, **kw: None):
        a = Agent.__new__(Agent)
        a.name = "test"
        a.model = "test-model"
        a.temperature = 0.7
        a.api_base = "http://localhost:11434/v1"
        a.api_key = None
        a.tools = []
        a.deferred_tools = []
        a.skills = []
        a.max_turns = 10
        a.context = ""
        a.conversation_history = []
        a.mcp_manager = MagicMock()
        a.memory_enabled = False
        a.memory = None
        a.priority = 5
        a.observer = MagicMock()
        a.observer.trace = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        a.observer.record = MagicMock()
        a.cache_enabled = False
        a.cache = None
        a._llm_session = None
        a._pending_mcp_config = None
        return a


@pytest.mark.asyncio
async def test_stream_llm_yields_chunks(agent):
    """_stream_llm yields content from SSE chunks."""
    chunks = ["Hello", " world", "!"]
    agent._llm_session = _make_mock_session(make_sse_lines(chunks))

    collected = []
    async for chunk in agent._stream_llm("test prompt"):
        collected.append(chunk)

    assert collected == ["Hello", " world", "!"]


@pytest.mark.asyncio
async def test_stream_llm_skips_empty_deltas(agent):
    """_stream_llm skips chunks without content."""
    lines = [
        f'data: {json.dumps({"choices": [{"delta": {"content": "hi"}}]})}\n\n'.encode(),
        f'data: {json.dumps({"choices": [{"delta": {"role": "assistant"}}]})}\n\n'.encode(),
        f'data: {json.dumps({"choices": [{"delta": {"content": " there"}}]})}\n\n'.encode(),
        b"data: [DONE]\n\n",
    ]
    agent._llm_session = _make_mock_session.__wrapped__(lines) if hasattr(_make_mock_session, '__wrapped__') else None
    # Build mock manually for custom lines
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = AsyncLineIterator(lines)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.closed = False
    agent._llm_session = mock_session

    collected = []
    async for chunk in agent._stream_llm("test"):
        collected.append(chunk)

    assert collected == ["hi", " there"]


@pytest.mark.asyncio
async def test_stream_llm_ignores_non_data_lines(agent):
    """_stream_llm ignores comment and empty lines."""
    lines = [
        b": comment line\n\n",
        b"\n",
        f'data: {json.dumps({"choices": [{"delta": {"content": "ok"}}]})}\n\n'.encode(),
        b"data: [DONE]\n\n",
    ]

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = AsyncLineIterator(lines)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.closed = False
    agent._llm_session = mock_session

    collected = []
    async for chunk in agent._stream_llm("test"):
        collected.append(chunk)

    assert collected == ["ok"]


@pytest.mark.asyncio
async def test_stream_public_method(agent):
    """stream() yields chunks and records events."""
    chunks = ["Hello", " from", " stream"]
    agent._llm_session = _make_mock_session(make_sse_lines(chunks))

    collected = []
    async for chunk in agent.stream("hello"):
        collected.append(chunk)

    assert collected == ["Hello", " from", " stream"]
    assert agent.observer.record.call_count >= 2  # start + end


@pytest.mark.asyncio
async def test_stream_stores_memory(agent):
    """stream() stores conversation in memory when enabled."""
    agent.memory_enabled = True
    agent.memory = MagicMock()
    agent.memory.remember = MagicMock()

    agent._llm_session = _make_mock_session(make_sse_lines(["Hi"]))

    collected = []
    async for chunk in agent.stream("test"):
        collected.append(chunk)

    assert collected == ["Hi"]
    # memory.remember called via asyncio.to_thread — 2 calls (user + agent)
    # In test env, to_thread actually runs the function
    assert agent.memory.remember.call_count == 2


@pytest.mark.asyncio
async def test_stream_caches_full_response(agent):
    """stream() caches the complete response after streaming."""
    agent.cache_enabled = True
    agent.cache = MagicMock()
    agent.cache.set = MagicMock()

    agent._llm_session = _make_mock_session(make_sse_lines(["one", " two"]))

    async for _ in agent.stream("test"):
        pass

    agent.cache.set.assert_called_once()
    call_args = agent.cache.set.call_args
    # Check positional args: (prompt, model, response, ...)
    assert "one two" in str(call_args)


@pytest.mark.asyncio
async def test_stream_updates_conversation_history(agent):
    """stream() appends to conversation_history."""
    agent._llm_session = _make_mock_session(make_sse_lines(["response"]))

    async for _ in agent.stream("hello"):
        pass

    assert len(agent.conversation_history) == 1
    entry = agent.conversation_history[0]
    assert entry["user_input"] == "hello"
    assert entry["response"]["streaming"] is True
    assert entry["response"]["result"] == "response"


@pytest.mark.asyncio
async def test_stream_sends_stream_true(agent):
    """_stream_llm sends stream=True to the API."""
    mock_session = _make_mock_session(make_sse_lines(["ok"]))
    agent._llm_session = mock_session

    async for _ in agent._stream_llm("test"):
        pass

    call_kwargs = mock_session.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["stream"] is True
    assert request_body["model"] == "test-model"
