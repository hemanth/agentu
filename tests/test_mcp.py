"""Tests for MCP functionality."""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from agentu import (
    MCPServerConfig,
    AuthConfig,
    TransportType,
    MCPToolAdapter,
    MCPConfigLoader,
    Agent
)
from agentu.mcp_transport import MCPHTTPTransport


class TestAuthConfig:
    """Test AuthConfig class."""

    def test_bearer_token(self):
        auth = AuthConfig.bearer_token("test_token")
        assert auth.type == "bearer"
        assert auth.headers["Authorization"] == "Bearer test_token"

    def test_api_key(self):
        auth = AuthConfig.api_key("test_key", header_name="X-API-Key")
        assert auth.type == "apikey"
        assert auth.headers["X-API-Key"] == "test_key"

    def test_custom_headers(self):
        auth = AuthConfig(
            type="custom",
            headers={"Authorization": "Custom token", "X-Client": "test"}
        )
        assert auth.type == "custom"
        assert auth.headers["Authorization"] == "Custom token"
        assert auth.headers["X-Client"] == "test"

    def test_from_dict(self):
        data = {
            "type": "bearer",
            "headers": {"Authorization": "Bearer dict_token"}
        }
        auth = AuthConfig.from_dict(data)
        assert auth.type == "bearer"
        assert auth.headers["Authorization"] == "Bearer dict_token"


class TestMCPServerConfig:
    """Test MCPServerConfig class."""

    def test_from_dict_http(self):
        data = {
            "type": "http",
            "url": "https://api.example.com/mcp",
            "auth": {
                "type": "bearer",
                "headers": {"Authorization": "Bearer test"}
            },
            "timeout": 45
        }
        config = MCPServerConfig.from_dict("test_server", data)

        assert config.name == "test_server"
        assert config.transport_type == TransportType.HTTP
        assert config.url == "https://api.example.com/mcp"
        assert config.auth is not None
        assert config.timeout == 45

    def test_from_dict_no_auth(self):
        data = {
            "type": "http",
            "url": "https://api.example.com/mcp"
        }
        config = MCPServerConfig.from_dict("test_server", data)

        assert config.name == "test_server"
        assert config.auth is None


class TestMCPHTTPTransport:
    """Test MCPHTTPTransport class."""

    def test_get_headers_with_auth(self):
        auth = AuthConfig.bearer_token("test_token")
        config = MCPServerConfig(
            name="test",
            transport_type=TransportType.HTTP,
            url="https://example.com/mcp",
            auth=auth
        )
        transport = MCPHTTPTransport(config)

        headers = transport._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test_token"

    def test_get_headers_no_auth(self):
        config = MCPServerConfig(
            name="test",
            transport_type=TransportType.HTTP,
            url="https://example.com/mcp"
        )
        transport = MCPHTTPTransport(config)

        headers = transport._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    @patch('agentu.mcp_transport.requests.post')
    def test_send_request_success(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"success": True},
            "id": 1
        }
        mock_post.return_value = mock_response

        config = MCPServerConfig(
            name="test",
            transport_type=TransportType.HTTP,
            url="https://example.com/mcp"
        )
        transport = MCPHTTPTransport(config)

        result = transport.send_request("test_method", {"param": "value"})
        assert result == {"success": True}

    @patch('agentu.mcp_transport.requests.post')
    def test_send_request_error(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "error": {"code": -1, "message": "Test error"},
            "id": 1
        }
        mock_post.return_value = mock_response

        config = MCPServerConfig(
            name="test",
            transport_type=TransportType.HTTP,
            url="https://example.com/mcp"
        )
        transport = MCPHTTPTransport(config)

        with pytest.raises(Exception, match="MCP server error"):
            transport.send_request("test_method")


class TestMCPConfigLoader:
    """Test MCPConfigLoader class."""

    def test_load_from_dict(self):
        config_data = {
            "mcp_servers": {
                "server1": {
                    "type": "http",
                    "url": "https://api1.example.com/mcp",
                    "auth": {
                        "type": "bearer",
                        "headers": {"Authorization": "Bearer token1"}
                    }
                },
                "server2": {
                    "type": "http",
                    "url": "https://api2.example.com/mcp"
                }
            }
        }

        servers = MCPConfigLoader.load_from_dict(config_data)
        assert len(servers) == 2
        assert "server1" in servers
        assert "server2" in servers
        assert servers["server1"].url == "https://api1.example.com/mcp"
        assert servers["server2"].auth is None

    def test_create_example_config(self, tmp_path):
        output_file = tmp_path / "test_config.json"
        MCPConfigLoader.create_example_config(str(output_file))

        assert output_file.exists()

        with open(output_file) as f:
            config = json.load(f)

        assert "mcp_servers" in config
        assert "example_server" in config["mcp_servers"]


class TestAgentMCPIntegration:
    """Test Agent integration with MCP."""

    @patch.object(MCPToolAdapter, 'load_tools')
    def test_with_mcp_url(self, mock_load_tools):
        """Test with_mcp with simple URL."""
        mock_load_tools.return_value = []

        agent = Agent(name="test_agent")
        result = agent.with_mcp(["https://example.com/mcp"])

        assert result is agent  # Check chainable
        assert len(agent.tools) == 0  # Mock returns empty
        mock_load_tools.assert_called_once()

    @patch.object(MCPToolAdapter, 'load_tools')
    def test_with_mcp_dict(self, mock_load_tools):
        """Test with_mcp with dict containing auth."""
        mock_load_tools.return_value = []

        agent = Agent(name="test_agent")
        result = agent.with_mcp([{
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer token123"}
        }])

        assert result is agent  # Check chainable
        mock_load_tools.assert_called_once()


class TestMCPToolAdapter:
    """Test MCPToolAdapter class."""

    @patch.object(MCPHTTPTransport, 'list_tools')
    def test_load_tools(self, mock_list_tools):
        """Test loading tools from MCP server."""
        mock_list_tools.return_value = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Result limit"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

        config = MCPServerConfig(
            name="test_server",
            transport_type=TransportType.HTTP,
            url="https://example.com/mcp"
        )
        adapter = MCPToolAdapter(config)
        tools = adapter.load_tools()

        assert len(tools) == 1
        assert tools[0].name == "test_server_test_tool"
        assert "test_server" in tools[0].description
        assert "query" in tools[0].parameters
        assert "limit" in tools[0].parameters

    @patch.object(MCPHTTPTransport, 'list_tools')
    def test_convert_schema_to_parameters(self, mock_list_tools):
        """Test JSON schema conversion to parameters."""
        mock_list_tools.return_value = [
            {
                "name": "tool",
                "description": "Test",
                "inputSchema": {
                    "properties": {
                        "required_param": {
                            "type": "string",
                            "description": "Required parameter"
                        },
                        "optional_param": {
                            "type": "number",
                            "description": "Optional parameter"
                        }
                    },
                    "required": ["required_param"]
                }
            }
        ]

        config = MCPServerConfig(
            name="test",
            transport_type=TransportType.HTTP,
            url="https://example.com/mcp"
        )
        adapter = MCPToolAdapter(config)
        tools = adapter.load_tools()

        params = tools[0].parameters
        assert "required_param" in params
        assert "optional_param" in params
        assert "(required)" in params["required_param"]
        assert "(optional)" in params["optional_param"]


class TestMCPHTTPTransportIntegration:
    """Integration tests for MCPHTTPTransport."""

    @patch('agentu.mcp_transport.requests.post')
    def test_list_tools_flow(self, mock_post):
        """Test complete list_tools flow."""
        # Mock initialize response
        init_response = Mock()
        init_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {}
            },
            "id": 1
        }

        # Mock list_tools response
        tools_response = Mock()
        tools_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search tool",
                        "inputSchema": {}
                    }
                ]
            },
            "id": 2
        }

        mock_post.side_effect = [init_response, tools_response]

        config = MCPServerConfig(
            name="test",
            transport_type=TransportType.HTTP,
            url="https://example.com/mcp"
        )
        transport = MCPHTTPTransport(config)
        tools = transport.list_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert mock_post.call_count == 2

    @patch('agentu.mcp_transport.requests.post')
    def test_call_tool_flow(self, mock_post):
        """Test complete call_tool flow."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {
                "content": [
                    {"text": "Tool result"}
                ]
            },
            "id": 1
        }
        mock_post.return_value = mock_response

        config = MCPServerConfig(
            name="test",
            transport_type=TransportType.HTTP,
            url="https://example.com/mcp"
        )
        transport = MCPHTTPTransport(config)
        result = transport.call_tool("test_tool", {"param": "value"})

        assert result == "Tool result"

    @patch('agentu.mcp_transport.requests.post')
    def test_call_tool_with_auth(self, mock_post):
        """Test tool call with authentication headers."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"content": [{"text": "Result"}]},
            "id": 1
        }
        mock_post.return_value = mock_response

        auth = AuthConfig.bearer_token("secret_token")
        config = MCPServerConfig(
            name="test",
            transport_type=TransportType.HTTP,
            url="https://example.com/mcp",
            auth=auth
        )
        transport = MCPHTTPTransport(config)
        transport.call_tool("tool", {})

        # Verify auth header was included
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer secret_token"


class TestMCPConfigLoaderAdvanced:
    """Advanced tests for MCPConfigLoader."""

    def test_load_from_file(self, tmp_path):
        """Test loading config from file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "mcp_servers": {
                "server1": {
                    "type": "http",
                    "url": "https://api.example.com/mcp"
                }
            }
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        servers = MCPConfigLoader.load_from_file(str(config_file))
        assert len(servers) == 1
        assert "server1" in servers

    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file."""
        servers = MCPConfigLoader.load_from_file("/nonexistent/path.json")
        assert len(servers) == 0

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON."""
        config_file = tmp_path / "invalid.json"
        with open(config_file, 'w') as f:
            f.write("invalid json {{{")

        with pytest.raises(json.JSONDecodeError):
            MCPConfigLoader.load_from_file(str(config_file))

    def test_multiple_auth_types(self):
        """Test different authentication types in config."""
        config_data = {
            "mcp_servers": {
                "bearer_server": {
                    "type": "http",
                    "url": "https://bearer.example.com/mcp",
                    "auth": {
                        "type": "bearer",
                        "headers": {"Authorization": "Bearer token"}
                    }
                },
                "apikey_server": {
                    "type": "http",
                    "url": "https://apikey.example.com/mcp",
                    "auth": {
                        "type": "apikey",
                        "headers": {"X-API-Key": "key123"}
                    }
                },
                "custom_server": {
                    "type": "http",
                    "url": "https://custom.example.com/mcp",
                    "auth": {
                        "type": "custom",
                        "headers": {
                            "Custom-Header": "value",
                            "Another-Header": "another"
                        }
                    }
                }
            }
        }

        servers = MCPConfigLoader.load_from_dict(config_data)
        assert len(servers) == 3
        assert servers["bearer_server"].auth.type == "bearer"
        assert servers["apikey_server"].auth.type == "apikey"
        assert servers["custom_server"].auth.type == "custom"


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @patch('agentu.mcp_transport.requests.post')
    @pytest.mark.asyncio
    async def test_complete_workflow(self, mock_post):
        """Test complete workflow from config to tool execution."""
        # Mock initialize
        init_response = Mock()
        init_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"protocolVersion": "2024-11-05"},
            "id": 1
        }

        # Mock list_tools
        tools_response = Mock()
        tools_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {
                        "name": "calculator",
                        "description": "Calculator tool",
                        "inputSchema": {
                            "properties": {
                                "operation": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"}
                            },
                            "required": ["operation", "x", "y"]
                        }
                    }
                ]
            },
            "id": 2
        }

        # Mock call_tool
        call_response = Mock()
        call_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"content": [{"text": "42"}]},
            "id": 3
        }

        mock_post.side_effect = [init_response, tools_response, call_response]

        # Create agent and add MCP server
        agent = Agent(name="test_agent")
        agent.with_mcp([{
            "url": "https://math.example.com/mcp",
            "headers": {"Authorization": "Bearer test_token"},
            "name": "math_server"
        }])

        # Verify tools loaded
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "math_server_calculator"

        # Execute tool
        result = await agent.call("math_server_calculator", {
            "operation": "add",
            "x": 40,
            "y": 2
        })

        assert result == "42"
        agent.close_mcp_connections()

    @patch('agentu.mcp_config.load_mcp_servers')
    @patch('agentu.mcp_transport.requests.post')
    def test_auto_load_from_config(self, mock_post, mock_load_servers):
        """Test auto-loading tools from config file."""
        # Mock config loader
        mock_load_servers.return_value = {
            "server1": MCPServerConfig(
                name="server1",
                transport_type=TransportType.HTTP,
                url="https://example.com/mcp"
            )
        }

        # Mock MCP responses
        init_response = Mock()
        init_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"protocolVersion": "2024-11-05"},
            "id": 1
        }

        tools_response = Mock()
        tools_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"tools": []},
            "id": 2
        }

        mock_post.side_effect = [init_response, tools_response]

        # Create agent with auto-load
        agent = Agent(name="auto_agent", load_mcp_tools=True, mcp_config_path="config.json")

        # Verify load was called
        mock_load_servers.assert_called_once()
        agent.close_mcp_connections()
