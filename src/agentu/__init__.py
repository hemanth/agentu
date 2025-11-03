"""AgentU - A flexible Python package for creating AI agents with customizable tools."""

from .agent import Agent
from .tools import Tool
from .search import SearchAgent, search_tool
from .mcp_config import MCPConfigLoader, load_mcp_servers
from .mcp_transport import MCPServerConfig, AuthConfig, TransportType
from .mcp_tool import MCPToolAdapter, MCPToolManager

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "Tool",
    "SearchAgent",
    "search_tool",
    "MCPConfigLoader",
    "load_mcp_servers",
    "MCPServerConfig",
    "AuthConfig",
    "TransportType",
    "MCPToolAdapter",
    "MCPToolManager",
]