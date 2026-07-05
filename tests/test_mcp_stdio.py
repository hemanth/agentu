"""Tests for MCP STDIO transport."""
import json
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from agentu.mcp.transport import (
    MCPSTDIOTransport,
    MCPServerConfig,
    TransportType,
    create_transport,
)


def _make_stdio_config(**overrides):
    """Helper to create a minimal STDIO MCPServerConfig."""
    defaults = {
        "name": "test_stdio",
        "transport_type": TransportType.STDIO,
        "command": "node",
        "args": ["server.js"],
        "timeout": 5,
    }
    defaults.update(overrides)
    return MCPServerConfig(**defaults)


def _jsonrpc_response(req_id, result):
    """Return a newline-terminated JSON-RPC success response as bytes."""
    return (json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n").encode()


def _jsonrpc_error(req_id, code, message):
    """Return a newline-terminated JSON-RPC error response as bytes."""
    return (
        json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})
        + "\n"
    ).encode()


def _mock_process(stdout_lines=None, returncode=None):
    """Create a mock asyncio.subprocess.Process.

    Args:
        stdout_lines: list of bytes lines that readline() will yield in order.
                      Append b"" to simulate EOF.
        returncode: the process return code (None means still running).
    """
    proc = MagicMock()
    proc.pid = 12345

    # stdin
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()

    # stdout
    proc.stdout = MagicMock()
    if stdout_lines is not None:
        proc.stdout.readline = AsyncMock(side_effect=stdout_lines)
    else:
        proc.stdout.readline = AsyncMock(return_value=b"")

    # stderr
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(return_value=b"")

    # returncode property
    type(proc).returncode = PropertyMock(return_value=returncode)

    # wait
    proc.wait = AsyncMock(return_value=returncode or 0)
    proc.terminate = MagicMock()
    proc.kill = MagicMock()

    return proc


# ---------------------------------------------------------------------------
# Construction & Config Tests
# ---------------------------------------------------------------------------

class TestMCPSTDIOTransportConfig:
    """Test MCPSTDIOTransport configuration and construction."""

    def test_create_with_config(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        assert transport.config.command == "node"
        assert transport.config.args == ["server.js"]
        assert transport.request_id == 0
        assert transport._process is None
        assert transport._initialized is False

    def test_config_from_dict_stdio(self):
        data = {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "mcp_server"],
            "env": {"API_KEY": "secret"},
            "timeout": 60,
        }
        config = MCPServerConfig.from_dict("my_server", data)

        assert config.name == "my_server"
        assert config.transport_type == TransportType.STDIO
        assert config.command == "python"
        assert config.args == ["-m", "mcp_server"]
        assert config.env == {"API_KEY": "secret"}
        assert config.timeout == 60
        assert config.url is None

    def test_config_from_dict_stdio_minimal(self):
        """STDIO config with only command, no args or env."""
        data = {"type": "stdio", "command": "mcp-server"}
        config = MCPServerConfig.from_dict("bare", data)

        assert config.command == "mcp-server"
        assert config.args is None
        assert config.env is None

    def test_config_backwards_compatible(self):
        """Existing HTTP configs still work without args/env fields."""
        data = {
            "type": "http",
            "url": "https://example.com/mcp",
        }
        config = MCPServerConfig.from_dict("http_server", data)
        assert config.args is None
        assert config.env is None

    def test_factory_creates_stdio_transport(self):
        config = _make_stdio_config()
        transport = create_transport(config)

        assert isinstance(transport, MCPSTDIOTransport)


# ---------------------------------------------------------------------------
# Subprocess Lifecycle
# ---------------------------------------------------------------------------

class TestMCPSTDIOSubprocessLifecycle:
    """Test subprocess start / stop."""

    @pytest.mark.asyncio
    async def test_start_process(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        mock_proc = _mock_process(returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await transport._start_process()

        assert transport._process is mock_proc

    @pytest.mark.asyncio
    async def test_start_process_no_command(self):
        config = _make_stdio_config(command=None)
        transport = MCPSTDIOTransport(config)

        with pytest.raises(ValueError, match="Command is required"):
            await transport._start_process()

    @pytest.mark.asyncio
    async def test_start_process_idempotent(self):
        """Calling _start_process when process is already running is a no-op."""
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        mock_proc = _mock_process(returncode=None)
        transport._process = mock_proc

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            await transport._start_process()
            mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_process_with_env(self):
        config = _make_stdio_config(env={"MY_VAR": "hello"})
        transport = MCPSTDIOTransport(config)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = _mock_process(returncode=None)
            await transport._start_process()

            call_kwargs = mock_exec.call_args
            env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
            assert env is not None
            assert env["MY_VAR"] == "hello"

    @pytest.mark.asyncio
    async def test_start_process_no_env_passes_none(self):
        config = _make_stdio_config(env=None)
        transport = MCPSTDIOTransport(config)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = _mock_process(returncode=None)
            await transport._start_process()

            call_kwargs = mock_exec.call_args
            env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
            assert env is None


# ---------------------------------------------------------------------------
# _write_message / _read_message low-level
# ---------------------------------------------------------------------------

class TestMCPSTDIOReadWrite:
    """Test low-level read and write helpers."""

    @pytest.mark.asyncio
    async def test_write_message(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        mock_proc = _mock_process(returncode=None)
        transport._process = mock_proc

        msg = {"jsonrpc": "2.0", "method": "test", "id": 1}
        await transport._write_message(msg)

        expected = json.dumps(msg) + "\n"
        mock_proc.stdin.write.assert_called_once_with(expected.encode("utf-8"))
        mock_proc.stdin.drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_message_no_process(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        with pytest.raises(RuntimeError, match="not running"):
            await transport._write_message({"test": True})

    @pytest.mark.asyncio
    async def test_read_message_success(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        response_bytes = _jsonrpc_response(1, {"tools": []})
        mock_proc = _mock_process(stdout_lines=[response_bytes], returncode=None)
        transport._process = mock_proc

        result = await transport._read_message()
        assert result["id"] == 1
        assert result["result"] == {"tools": []}

    @pytest.mark.asyncio
    async def test_read_message_skips_empty_lines(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        response_bytes = _jsonrpc_response(1, {"ok": True})
        mock_proc = _mock_process(
            stdout_lines=[b"\n", b"  \n", response_bytes],
            returncode=None,
        )
        transport._process = mock_proc

        result = await transport._read_message()
        assert result["result"] == {"ok": True}

    @pytest.mark.asyncio
    async def test_read_message_skips_non_json_lines(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        response_bytes = _jsonrpc_response(1, {"ok": True})
        mock_proc = _mock_process(
            stdout_lines=[b"INFO: server starting\n", response_bytes],
            returncode=None,
        )
        transport._process = mock_proc

        result = await transport._read_message()
        assert result["result"] == {"ok": True}

    @pytest.mark.asyncio
    async def test_read_message_eof_raises_connection_error(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        mock_proc = _mock_process(stdout_lines=[b""], returncode=1)
        transport._process = mock_proc

        with pytest.raises(ConnectionError, match="exited unexpectedly"):
            await transport._read_message()

    @pytest.mark.asyncio
    async def test_read_message_timeout(self):
        config = _make_stdio_config(timeout=0.01)
        transport = MCPSTDIOTransport(config)

        async def slow_readline():
            await asyncio.sleep(10)
            return b""

        mock_proc = _mock_process(returncode=None)
        mock_proc.stdout.readline = slow_readline
        transport._process = mock_proc

        with pytest.raises(TimeoutError, match="Timeout"):
            await transport._read_message()


# ---------------------------------------------------------------------------
# send_request
# ---------------------------------------------------------------------------

class TestMCPSTDIOSendRequest:
    """Test send_request method."""

    @pytest.mark.asyncio
    async def test_send_request_success(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        response_bytes = _jsonrpc_response(1, {"tools": []})
        mock_proc = _mock_process(stdout_lines=[response_bytes], returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            result = await transport.send_request("tools/list")

        assert result == {"tools": []}

    @pytest.mark.asyncio
    async def test_send_request_with_params(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        response_bytes = _jsonrpc_response(1, {"content": [{"text": "hello"}]})
        mock_proc = _mock_process(stdout_lines=[response_bytes], returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            result = await transport.send_request("tools/call", {"name": "greet"})

        assert result == {"content": [{"text": "hello"}]}

        # Verify the written message includes params
        written_data = mock_proc.stdin.write.call_args[0][0]
        written_msg = json.loads(written_data.decode("utf-8"))
        assert written_msg["params"] == {"name": "greet"}

    @pytest.mark.asyncio
    async def test_send_request_server_error(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        error_bytes = _jsonrpc_error(1, -32600, "Invalid Request")
        mock_proc = _mock_process(stdout_lines=[error_bytes], returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            with pytest.raises(Exception, match="MCP server error"):
                await transport.send_request("bad/method")

    @pytest.mark.asyncio
    async def test_request_id_increments(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        resp1 = _jsonrpc_response(1, {})
        resp2 = _jsonrpc_response(2, {})
        mock_proc = _mock_process(stdout_lines=[resp1, resp2], returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await transport.send_request("method1")
            await transport.send_request("method2")

        assert transport.request_id == 2

        # Verify the two written messages had ids 1 and 2
        calls = mock_proc.stdin.write.call_args_list
        msg1 = json.loads(calls[0][0][0].decode())
        msg2 = json.loads(calls[1][0][0].decode())
        assert msg1["id"] == 1
        assert msg2["id"] == 2


# ---------------------------------------------------------------------------
# Initialize Handshake
# ---------------------------------------------------------------------------

class TestMCPSTDIOInitialize:
    """Test MCP initialize handshake."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        init_response = _jsonrpc_response(1, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "test-server"},
        })
        mock_proc = _mock_process(stdout_lines=[init_response], returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            result = await transport.initialize()

        assert result["protocolVersion"] == "2024-11-05"
        assert transport._initialized is True

        # Verify the initialize request was sent
        first_write = mock_proc.stdin.write.call_args_list[0][0][0]
        init_msg = json.loads(first_write.decode())
        assert init_msg["method"] == "initialize"
        assert init_msg["params"]["clientInfo"]["name"] == "agentu"

        # Verify notifications/initialized was sent (second write)
        second_write = mock_proc.stdin.write.call_args_list[1][0][0]
        notif_msg = json.loads(second_write.decode())
        assert notif_msg["method"] == "notifications/initialized"
        assert "id" not in notif_msg  # Notifications have no id

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)
        transport._initialized = True

        result = await transport.initialize()
        assert result == {}

    @pytest.mark.asyncio
    async def test_list_tools_triggers_initialize(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        init_response = _jsonrpc_response(1, {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
        })
        tools_response = _jsonrpc_response(2, {
            "tools": [
                {"name": "search", "description": "Search tool", "inputSchema": {}}
            ]
        })
        mock_proc = _mock_process(
            stdout_lines=[init_response, tools_response],
            returncode=None,
        )

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            tools = await transport.list_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert transport._initialized is True


# ---------------------------------------------------------------------------
# list_tools & call_tool
# ---------------------------------------------------------------------------

class TestMCPSTDIOToolOperations:
    """Test list_tools and call_tool methods."""

    @pytest.mark.asyncio
    async def test_list_tools(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)
        transport._initialized = True

        tools_data = [
            {"name": "tool_a", "description": "Tool A", "inputSchema": {}},
            {"name": "tool_b", "description": "Tool B", "inputSchema": {}},
        ]
        response = _jsonrpc_response(1, {"tools": tools_data})
        mock_proc = _mock_process(stdout_lines=[response], returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            tools = await transport.list_tools()

        assert len(tools) == 2
        assert tools[0]["name"] == "tool_a"
        assert tools[1]["name"] == "tool_b"

    @pytest.mark.asyncio
    async def test_list_tools_empty(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)
        transport._initialized = True

        response = _jsonrpc_response(1, {"tools": []})
        mock_proc = _mock_process(stdout_lines=[response], returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            tools = await transport.list_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_call_tool_text_content(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)
        transport._initialized = True

        response = _jsonrpc_response(1, {
            "content": [{"type": "text", "text": "Hello, world!"}]
        })
        mock_proc = _mock_process(stdout_lines=[response], returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            result = await transport.call_tool("greet", {"name": "world"})

        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_call_tool_dict_result(self):
        """When result has no content key, return the raw result dict."""
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)
        transport._initialized = True

        response = _jsonrpc_response(1, {"data": 42})
        mock_proc = _mock_process(stdout_lines=[response], returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            result = await transport.call_tool("compute", {"x": 42})

        assert result == {"data": 42}

    @pytest.mark.asyncio
    async def test_call_tool_empty_content_list(self):
        """When content is an empty list, return the raw result."""
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)
        transport._initialized = True

        response = _jsonrpc_response(1, {"content": []})
        mock_proc = _mock_process(stdout_lines=[response], returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            result = await transport.call_tool("noop", {})

        assert result == {"content": []}

    @pytest.mark.asyncio
    async def test_call_tool_error_propagates(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)
        transport._initialized = True

        error_response = _jsonrpc_error(1, -32601, "Method not found")
        mock_proc = _mock_process(stdout_lines=[error_response], returncode=None)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            with pytest.raises(Exception, match="MCP server error"):
                await transport.call_tool("nonexistent", {})


# ---------------------------------------------------------------------------
# Close / Cleanup
# ---------------------------------------------------------------------------

class TestMCPSTDIOClose:
    """Test subprocess cleanup on close."""

    @pytest.mark.asyncio
    async def test_close_graceful(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        mock_proc = _mock_process(returncode=None)
        # Simulate the process exiting after stdin close
        mock_proc.wait = AsyncMock(return_value=0)
        transport._process = mock_proc
        transport._initialized = True

        # After close, make returncode indicate process exited
        await transport.close()

        mock_proc.stdin.close.assert_called_once()
        mock_proc.wait.assert_awaited()
        assert transport._process is None
        assert transport._initialized is False

    @pytest.mark.asyncio
    async def test_close_force_terminate(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        mock_proc = _mock_process(returncode=None)

        # First wait times out, second succeeds after terminate
        call_count = 0

        async def wait_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate process not exiting within timeout
                raise asyncio.TimeoutError()
            return 0

        mock_proc.wait = AsyncMock(side_effect=wait_side_effect)
        transport._process = mock_proc

        await transport.close()

        mock_proc.terminate.assert_called_once()
        assert transport._process is None

    @pytest.mark.asyncio
    async def test_close_force_kill(self):
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        mock_proc = _mock_process(returncode=None)

        # Both waits time out, requiring kill
        call_count = 0

        async def wait_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise asyncio.TimeoutError()
            return -9

        mock_proc.wait = AsyncMock(side_effect=wait_side_effect)
        transport._process = mock_proc

        await transport.close()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert transport._process is None

    @pytest.mark.asyncio
    async def test_close_already_exited(self):
        """Close when process already exited is a no-op for termination."""
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        mock_proc = _mock_process(returncode=0)
        transport._process = mock_proc

        await transport.close()

        # Should not try to close stdin or wait since process already exited
        mock_proc.terminate.assert_not_called()
        mock_proc.kill.assert_not_called()
        assert transport._process is None

    @pytest.mark.asyncio
    async def test_close_no_process(self):
        """Close with no process should not raise."""
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        await transport.close()  # Should not raise
        assert transport._process is None


# ---------------------------------------------------------------------------
# Full Flow Integration
# ---------------------------------------------------------------------------

class TestMCPSTDIOFullFlow:
    """Integration-style tests for the complete STDIO transport flow."""

    @pytest.mark.asyncio
    async def test_complete_list_and_call_flow(self):
        """Test initialize -> list_tools -> call_tool -> close."""
        config = _make_stdio_config()
        transport = MCPSTDIOTransport(config)

        init_response = _jsonrpc_response(1, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
        })
        tools_response = _jsonrpc_response(2, {
            "tools": [
                {
                    "name": "calculator",
                    "description": "Math calculator",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"],
                    },
                }
            ]
        })
        call_response = _jsonrpc_response(3, {
            "content": [{"type": "text", "text": "42"}]
        })
        mock_proc = _mock_process(
            stdout_lines=[init_response, tools_response, call_response],
            returncode=None,
        )
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            # list_tools triggers initialize
            tools = await transport.list_tools()
            assert len(tools) == 1
            assert tools[0]["name"] == "calculator"

            # call_tool
            result = await transport.call_tool("calculator", {"expression": "6*7"})
            assert result == "42"

            # close
            await transport.close()

        assert transport._process is None
        assert transport._initialized is False

    @pytest.mark.asyncio
    async def test_create_transport_factory_stdio(self):
        """Test that create_transport returns an MCPSTDIOTransport for STDIO config."""
        config = _make_stdio_config()
        transport = create_transport(config)

        assert isinstance(transport, MCPSTDIOTransport)
        assert transport.config is config
