"""Tests for MCP protocol extensions."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from agentu.mcp.extensions import (
    StreamableHTTPTransport,
    ElicitationRequest,
    ElicitationResponse,
    ElicitationMixin,
    OAuthConfig,
    OAuthTokenManager,
    TaskState,
    MCPTask,
    TaskManager,
)


class TestStreamableHTTPTransport:
    """Tests for the Streamable HTTP transport."""

    def test_init(self):
        transport = StreamableHTTPTransport(url="http://localhost:3000/mcp")
        assert transport.url == "http://localhost:3000/mcp"
        assert transport.timeout == 30
        assert transport.request_id == 0

    def test_init_strips_trailing_slash(self):
        transport = StreamableHTTPTransport(url="http://localhost:3000/mcp/")
        assert transport.url == "http://localhost:3000/mcp"

    def test_next_id_increments(self):
        transport = StreamableHTTPTransport(url="http://localhost:3000")
        assert transport._next_id() == 1
        assert transport._next_id() == 2
        assert transport._next_id() == 3

    @pytest.mark.asyncio
    async def test_send_request_json_response(self):
        transport = StreamableHTTPTransport(url="http://localhost:3000")

        mock_response = AsyncMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": [{"name": "test_tool"}]},
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_response)
        transport._session = mock_session

        result = await transport.send_request("tools/list")
        assert result == {"tools": [{"name": "test_tool"}]}

    @pytest.mark.asyncio
    async def test_send_request_error_response(self):
        transport = StreamableHTTPTransport(url="http://localhost:3000")

        mock_response = AsyncMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32600, "message": "Invalid Request"},
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_response)
        transport._session = mock_session

        with pytest.raises(Exception, match="Invalid Request"):
            await transport.send_request("tools/list")

    @pytest.mark.asyncio
    async def test_list_tools(self):
        transport = StreamableHTTPTransport(url="http://localhost:3000")
        transport.send_request = AsyncMock(return_value={
            "tools": [
                {"name": "search", "description": "Search stuff"},
                {"name": "calc", "description": "Calculate"},
            ]
        })

        tools = await transport.list_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_call_tool(self):
        transport = StreamableHTTPTransport(url="http://localhost:3000")
        transport.send_request = AsyncMock(return_value={
            "content": [
                {"type": "text", "text": "Hello from tool"}
            ]
        })

        result = await transport.call_tool("test_tool", {"input": "hello"})
        assert result == "Hello from tool"

    @pytest.mark.asyncio
    async def test_close(self):
        transport = StreamableHTTPTransport(url="http://localhost:3000")
        mock_session = AsyncMock()
        mock_session.closed = False
        transport._session = mock_session

        await transport.close()
        mock_session.close.assert_called_once()


class TestElicitation:
    """Tests for elicitation support."""

    def test_elicitation_request_creation(self):
        req = ElicitationRequest(
            request_id="req-1",
            message="Please enter your name",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        assert req.request_id == "req-1"
        assert req.message == "Please enter your name"
        assert req.schema is not None

    def test_elicitation_response_defaults(self):
        resp = ElicitationResponse(request_id="req-1")
        assert resp.action == "submit"
        assert resp.data is None

    def test_elicitation_url_mode(self):
        req = ElicitationRequest(
            request_id="req-1",
            message="Please authorize",
            url="https://auth.example.com/authorize",
        )
        assert req.url is not None

    @pytest.mark.asyncio
    async def test_mixin_with_handler(self):
        class TestTransport(ElicitationMixin):
            pass

        transport = TestTransport()

        async def handler(req):
            return ElicitationResponse(
                request_id=req.request_id,
                data={"name": "test user"},
            )

        transport.set_elicitation_handler(handler)

        result = await transport._handle_elicitation({
            "requestId": "req-1",
            "message": "Enter name",
        })
        assert result.data == {"name": "test user"}

    @pytest.mark.asyncio
    async def test_mixin_without_handler_dismisses(self):
        class TestTransport(ElicitationMixin):
            pass

        transport = TestTransport()
        result = await transport._handle_elicitation({
            "requestId": "req-1",
            "message": "Enter name",
        })
        assert result.action == "dismiss"


class TestOAuth:
    """Tests for OAuth 2.1 + CIMD auth."""

    def test_oauth_config_creation(self):
        config = OAuthConfig(
            client_id="https://myapp.example.com/.well-known/oauth-client",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
        assert config.is_cimd is True

    def test_non_cimd_config(self):
        config = OAuthConfig(
            client_id="my-app-id",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
        assert config.is_cimd is False

    def test_token_manager_init(self):
        config = OAuthConfig(
            client_id="test",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
        manager = OAuthTokenManager(config)
        assert manager.has_valid_token is False

    def test_get_auth_headers_empty(self):
        config = OAuthConfig(
            client_id="test",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
        manager = OAuthTokenManager(config)
        assert manager.get_auth_headers() == {}

    def test_get_auth_headers_with_token(self):
        config = OAuthConfig(
            client_id="test",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
        manager = OAuthTokenManager(config)
        manager._access_token = "test-token"
        manager._expires_at = 9999999999
        assert manager.get_auth_headers() == {"Authorization": "Bearer test-token"}
        assert manager.has_valid_token is True

    def test_authorization_url(self):
        config = OAuthConfig(
            client_id="test-client",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            scopes=["read", "write"],
        )
        manager = OAuthTokenManager(config)
        url = manager.get_authorization_url(state="abc123")
        assert "https://auth.example.com/authorize" in url
        assert "client_id=test-client" in url
        assert "state=abc123" in url
        assert "scope=read+write" in url


class TestTasks:
    """Tests for MCP task lifecycle."""

    def test_task_creation(self):
        task = MCPTask(task_id="task-1")
        assert task.state == TaskState.SUBMITTED
        assert task.result is None

    def test_task_state_values(self):
        assert TaskState("submitted") == TaskState.SUBMITTED
        assert TaskState("completed") == TaskState.COMPLETED
        assert TaskState("failed") == TaskState.FAILED

    @pytest.mark.asyncio
    async def test_submit_task(self):
        mock_transport = AsyncMock()
        mock_transport.send_request = AsyncMock(return_value={
            "taskId": "task-123",
            "state": "submitted",
        })

        manager = TaskManager(mock_transport)
        task = await manager.submit("long_tool", {"input": "data"})
        assert task.task_id == "task-123"
        assert task.state == TaskState.SUBMITTED

    @pytest.mark.asyncio
    async def test_get_status(self):
        mock_transport = AsyncMock()
        mock_transport.send_request = AsyncMock(return_value={
            "state": "working",
            "progress": 0.5,
        })

        manager = TaskManager(mock_transport)
        manager._tasks["task-123"] = MCPTask(task_id="task-123")

        task = await manager.get_status("task-123")
        assert task.state == TaskState.WORKING
        assert task.progress == 0.5

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        mock_transport = AsyncMock()
        mock_transport.send_request = AsyncMock(return_value={
            "state": "cancelled",
        })

        manager = TaskManager(mock_transport)
        manager._tasks["task-123"] = MCPTask(task_id="task-123")

        task = await manager.cancel("task-123")
        assert task.state == TaskState.CANCELLED

    @pytest.mark.asyncio
    async def test_wait_for_completion_immediate(self):
        mock_transport = AsyncMock()
        mock_transport.send_request = AsyncMock(return_value={
            "state": "completed",
            "result": {"output": "done"},
        })

        manager = TaskManager(mock_transport)
        manager._tasks["task-123"] = MCPTask(task_id="task-123")

        task = await manager.wait_for_completion("task-123", poll_interval=0.1)
        assert task.state == TaskState.COMPLETED
        assert task.result == {"output": "done"}

    @pytest.mark.asyncio
    async def test_wait_for_completion_polls(self):
        call_count = 0

        async def mock_send(method, params=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {"state": "working", "progress": call_count * 0.3}
            return {"state": "completed", "result": "final"}

        mock_transport = AsyncMock()
        mock_transport.send_request = mock_send

        manager = TaskManager(mock_transport)
        manager._tasks["task-123"] = MCPTask(task_id="task-123")

        task = await manager.wait_for_completion("task-123", poll_interval=0.05)
        assert task.state == TaskState.COMPLETED
        assert call_count == 3
