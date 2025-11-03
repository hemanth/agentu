"""MCP Tool adapter to wrap remote MCP tools as local Tool objects."""
import logging
from typing import Dict, Any, List, Callable
from dataclasses import dataclass

from .tools import Tool
from .mcp_transport import MCPTransport, MCPServerConfig, create_transport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """Adapter to convert MCP remote tools into local Tool objects."""

    def __init__(self, server_config: MCPServerConfig):
        """Initialize the MCP tool adapter.

        Args:
            server_config: Configuration for the MCP server
        """
        self.server_config = server_config
        self.transport = create_transport(server_config)
        self.tools_cache: List[Dict[str, Any]] = []

    def _create_tool_function(self, tool_name: str) -> Callable:
        """Create a callable function that invokes the remote MCP tool.

        Args:
            tool_name: Name of the MCP tool

        Returns:
            Callable function that accepts kwargs and calls the remote tool
        """
        def tool_function(**kwargs) -> Any:
            """Wrapper function that calls the remote MCP tool."""
            try:
                logger.info(f"Calling remote MCP tool: {tool_name} with args: {kwargs}")
                result = self.transport.call_tool(tool_name, kwargs)
                return result
            except Exception as e:
                logger.error(f"Error calling remote tool {tool_name}: {str(e)}")
                raise

        # Set function name for debugging
        tool_function.__name__ = f"mcp_{self.server_config.name}_{tool_name}"

        return tool_function

    def _convert_mcp_schema_to_parameters(self, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MCP JSON schema to agentu parameter format.

        Args:
            input_schema: MCP tool input schema (JSON Schema format)

        Returns:
            Dictionary of parameters in agentu format
        """
        parameters = {}

        if not input_schema or 'properties' not in input_schema:
            return parameters

        properties = input_schema.get('properties', {})
        required_fields = set(input_schema.get('required', []))

        for param_name, param_schema in properties.items():
            param_type = param_schema.get('type', 'string')
            description = param_schema.get('description', '')
            is_required = param_name in required_fields

            # Format: "type: description [required/optional]"
            param_str = f"{param_type}: {description}"
            if is_required:
                param_str += " (required)"
            else:
                param_str += " (optional)"

            parameters[param_name] = param_str

        return parameters

    def load_tools(self) -> List[Tool]:
        """Load all tools from the MCP server and convert them to Tool objects.

        Returns:
            List of Tool objects representing remote MCP tools
        """
        try:
            # Fetch tools from remote server
            self.tools_cache = self.transport.list_tools()
            logger.info(f"Loaded {len(self.tools_cache)} tools from MCP server: {self.server_config.name}")

            # Convert to Tool objects
            tools = []
            for mcp_tool in self.tools_cache:
                tool_name = mcp_tool.get('name', 'unknown')
                description = mcp_tool.get('description', '')
                input_schema = mcp_tool.get('inputSchema', {})

                # Create callable function for this tool
                tool_function = self._create_tool_function(tool_name)

                # Convert schema to parameters
                parameters = self._convert_mcp_schema_to_parameters(input_schema)

                # Create Tool object
                # Prefix tool name with server name to avoid conflicts
                prefixed_name = f"{self.server_config.name}_{tool_name}"

                tool = Tool(
                    name=prefixed_name,
                    description=f"[MCP:{self.server_config.name}] {description}",
                    function=tool_function,
                    parameters=parameters
                )

                tools.append(tool)
                logger.info(f"Created tool: {prefixed_name}")

            return tools

        except Exception as e:
            logger.error(f"Error loading tools from MCP server {self.server_config.name}: {str(e)}")
            raise

    def close(self):
        """Close the transport connection."""
        if self.transport:
            self.transport.close()


class MCPToolManager:
    """Manager for multiple MCP servers and their tools."""

    def __init__(self):
        """Initialize the MCP tool manager."""
        self.adapters: Dict[str, MCPToolAdapter] = {}

    def add_server(self, server_config: MCPServerConfig) -> MCPToolAdapter:
        """Add an MCP server and create an adapter for it.

        Args:
            server_config: Configuration for the MCP server

        Returns:
            The created MCPToolAdapter
        """
        if server_config.name in self.adapters:
            logger.warning(f"MCP server {server_config.name} already exists, replacing")

        adapter = MCPToolAdapter(server_config)
        self.adapters[server_config.name] = adapter
        logger.info(f"Added MCP server: {server_config.name}")

        return adapter

    def load_all_tools(self) -> List[Tool]:
        """Load tools from all registered MCP servers.

        Returns:
            List of all Tool objects from all MCP servers
        """
        all_tools = []

        for server_name, adapter in self.adapters.items():
            try:
                tools = adapter.load_tools()
                all_tools.extend(tools)
                logger.info(f"Loaded {len(tools)} tools from {server_name}")
            except Exception as e:
                logger.error(f"Failed to load tools from {server_name}: {str(e)}")

        return all_tools

    def get_adapter(self, server_name: str) -> MCPToolAdapter:
        """Get the adapter for a specific server.

        Args:
            server_name: Name of the MCP server

        Returns:
            The MCPToolAdapter for the server

        Raises:
            KeyError: If the server is not registered
        """
        if server_name not in self.adapters:
            raise KeyError(f"MCP server not found: {server_name}")

        return self.adapters[server_name]

    def close_all(self):
        """Close all transport connections."""
        for adapter in self.adapters.values():
            adapter.close()

        logger.info("Closed all MCP connections")
